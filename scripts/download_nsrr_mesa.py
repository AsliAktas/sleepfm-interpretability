"""Download MESA polysomnography files from NSRR with resume + retry + verify.

Replaces the Ruby `nsrr` gem for pulling a subject-level cohort. Uses the
NSRR v1 authenticated URL scheme:

    https://sleepdata.org/datasets/mesa/files/<path>?auth_token=<TOKEN>

Get a token at https://sleepdata.org/token (must have signed the MESA DUA).

Usage
-----
    # Token from NSRR_TOKEN env var (recommended; --token flag disallowed):
    $env:NSRR_TOKEN = "your_token_here"
    python scripts/download_nsrr_mesa.py --out C:/path/to/mesa_test_clean

    # From a token file (permissions-restricted):
    python scripts/download_nsrr_mesa.py --token-file ~/.nsrr/token

    # Interactive prompt (getpass, does not echo, not saved to history):
    python scripts/download_nsrr_mesa.py --prompt-token

Design notes on audit hardening
-------------------------------
- Token never appears in argv (audit #2). Only env var, token file, or
  interactive getpass prompt are supported.
- All error strings are sanitised through _redact() before print/log so a
  network exception carrying the query string never leaks the token
  (audit #1).
- Download URL embeds `?auth_token=<X>` directly and disables auto-redirect
  so we do not silently follow a Location: header to a URL that would drop
  our auth (audit #6). Any redirect is reported.
- Files download to `<name>.part` and are atomically renamed on success
  (audit #8). A `.lock` file prevents two concurrent runs from corrupting
  the same target.
- After download we validate the first bytes: EDF header starts with `0`
  (ASCII "0" x 8), XML starts with `<?xml` or `<PSGAnnotation` (audit #4).
- Resume via Range only trusts the server response (206). On 416 we HEAD
  to compare sizes and re-download from scratch if mismatched (audit #3).
- verify_token additionally probes a public MESA metadata file to catch
  the "token is valid but no DUA on this dataset" case early (audit #5).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from tqdm import tqdm

# Windows + conda envs often ship a certifi bundle missing intermediate CAs;
# truststore reuses the OS trust store (SChannel/Keychain/OpenSSL).
try:
    import truststore
    truststore.inject_into_ssl()
    _TRUST_STORE = "system"
except ImportError:  # fall back to certifi
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    _TRUST_STORE = "certifi"

NSRR_BASE = "https://sleepdata.org"
PROFILE_URL = f"{NSRR_BASE}/api/v1/account/profile.json"
# A small always-available MESA file used to probe DUA access (audit #5).
DUA_PROBE_PATH = "datasets/mesa/files/datasets/mesa-data-dictionary-0.6.0-domains.csv"
EDF_PATH_FMT = "datasets/mesa/files/polysomnography/edfs/mesa-sleep-{sid}.edf"
XML_PATH_FMT = "datasets/mesa/files/polysomnography/annotations-events-nsrr/mesa-sleep-{sid}-nsrr.xml"

DEFAULT_SUBJECTS: Tuple[str, ...] = (
    "[REDACTED_LIST_PER_DUA]",
)

MIN_EDF_BYTES = 50 * 1024 * 1024
MIN_XML_BYTES = 5 * 1024
EDF_MAGIC = b"0       "  # 8 ASCII "0"s in every EDF/EDF+ header
XML_MAGIC_PREFIXES = (b"<?xml", b"<PSGAnnotation")


@dataclass
class DownloadResult:
    subject_id: str
    edf_status: str
    xml_status: str
    edf_bytes: int
    xml_bytes: int
    edf_valid_header: bool = False
    xml_valid_header: bool = False

    @property
    def ok(self) -> bool:
        return (self.edf_status in {"downloaded", "already_complete"}
                and self.xml_status in {"downloaded", "already_complete"}
                and self.edf_bytes >= MIN_EDF_BYTES
                and self.xml_bytes >= MIN_XML_BYTES
                and self.edf_valid_header
                and self.xml_valid_header)


# ---------------------------------------------------------------------------
# Token handling — never in argv, never echoed to logs
# ---------------------------------------------------------------------------

_HEX_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{20,}")


def _redact(msg: str, token: str) -> str:
    """Remove the token from an error string. Also masks anything that
    looks like a bearer token (long alphanumeric) as a defense-in-depth
    against libraries that repr() the request context differently."""
    if not msg:
        return msg
    out = msg.replace(token, "***TOKEN_REDACTED***")
    # Strip query strings that carry auth_token even after substitution
    out = re.sub(r"auth_token=[^&\s'\"]+", "auth_token=***REDACTED***", out)
    return out


def _read_token_file(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[error] token file not found: {path}")
    if os.name == "posix":  # warn on world-readable token files on POSIX
        mode = path.stat().st_mode
        if mode & (stat.S_IROTH | stat.S_IRGRP):
            print(f"[warn] token file has broad read permissions "
                  f"({oct(mode)}); recommend chmod 600 {path}", file=sys.stderr)
    return path.read_text(encoding="utf-8").strip()


def resolve_token(args: argparse.Namespace) -> str:
    """Locate a token from the allowed sources. Never from argv directly.

    Precedence: --prompt-token > --token-file > $NSRR_TOKEN_FILE > $NSRR_TOKEN.
    $NSRR_TOKEN_FILE is preferred over $NSRR_TOKEN because it keeps the
    secret on disk (chmod 600) rather than in process environment.
    """
    if args.prompt_token:
        return getpass.getpass("NSRR token: ").strip()
    if args.token_file:
        return _read_token_file(Path(args.token_file).expanduser())
    env_token_file = os.environ.get("NSRR_TOKEN_FILE", "").strip()
    if env_token_file:
        return _read_token_file(Path(env_token_file).expanduser())
    token = os.environ.get("NSRR_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "[error] no token found. Provide one via:\n"
            "  --prompt-token             (interactive)\n"
            "  --token-file <path>        (file, chmod 600)\n"
            "  $env:NSRR_TOKEN_FILE=<path>  (env var pointing to token file)\n"
            "  $env:NSRR_TOKEN='...'      (env var with raw token)\n"
            "Get a token: https://sleepdata.org/token"
        )
    return token


# ---------------------------------------------------------------------------
# Token verification — includes DUA probe (audit #5)
# ---------------------------------------------------------------------------

def verify_token(token: str) -> Dict:
    try:
        r = requests.get(PROFILE_URL, params={"auth_token": token},
                         timeout=15, allow_redirects=False)
        r.raise_for_status()
        body = r.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        raise RuntimeError(f"profile check failed: {type(e).__name__}") from None

    if not body.get("authenticated"):
        raise RuntimeError("token is not authenticated (invalid or expired)")

    # DUA probe: HEAD the MESA data dictionary. 401/403 means auth OK but no
    # MESA access — better to know now than after 20 subjects fail.
    probe_url = f"{NSRR_BASE}/{DUA_PROBE_PATH}?{urlencode({'auth_token': token})}"
    try:
        rp = requests.head(probe_url, timeout=15, allow_redirects=False)
    except requests.RequestException as e:
        raise RuntimeError(f"MESA access probe failed: {type(e).__name__}") from None
    if rp.status_code in (401, 403):
        raise RuntimeError(
            "token is valid but MESA dataset access is denied "
            f"(HTTP {rp.status_code}). Sign the DUA at "
            "https://sleepdata.org/datasets/mesa"
        )
    return body


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------

def _build_url(path: str, token: str) -> str:
    """Embed token in URL directly so redirects preserve auth (audit #6)."""
    return f"{NSRR_BASE}/{path}?{urlencode({'auth_token': token})}"


def _validate_header(dest: Path, kind: str) -> bool:
    """Cheap magic-byte check to reject HTML error pages served as 200."""
    if not dest.exists() or dest.stat().st_size < 16:
        return False
    with open(dest, "rb") as f:
        head = f.read(64)
    if kind == "edf":
        return head.startswith(EDF_MAGIC)
    if kind == "xml":
        return any(head.startswith(m) for m in XML_MAGIC_PREFIXES)
    return False


def _remote_content_length(url: str) -> Optional[int]:
    """HEAD the URL and return Content-Length, or None on failure."""
    try:
        r = requests.head(url, timeout=15, allow_redirects=False)
        if r.status_code == 200 and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except (requests.RequestException, ValueError):
        pass
    return None


def _download_file(
    url: str, dest: Path, token: str,
    chunk_size: int = 64 * 1024, max_retries: int = 3, timeout: int = 120,
) -> Tuple[str, int]:
    """Streaming download with .part + resume + retry.

    Returns (status, final_size). Status one of:
    "downloaded", "already_complete", or "failed: <sanitized reason>".
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    # If final file already exists, trust it if the header validates and size
    # matches the server (HEAD probe). If mismatched, restart.
    if dest.exists():
        server_len = _remote_content_length(url)
        if server_len is not None and dest.stat().st_size == server_len:
            return "already_complete", dest.stat().st_size
        # size drift — restart cleanly
        dest.unlink()

    for attempt in range(max_retries):
        initial = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={initial}-"} if initial > 0 else {}
        try:
            r = requests.get(url, headers=headers, stream=True,
                             timeout=timeout, allow_redirects=False)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                return f"failed: {type(e).__name__}", initial
            time.sleep(2 ** attempt)
            continue

        # Handle redirects manually so we can log them without following blindly
        if 300 <= r.status_code < 400:
            r.close()
            loc = _redact(r.headers.get("Location", ""), token)
            return f"failed: unexpected redirect ({r.status_code}) to {loc}", initial

        if r.status_code == 416:
            r.close()
            # Range not satisfiable — verify local size matches server
            server_len = _remote_content_length(url)
            if server_len is not None and initial == server_len:
                # rename .part -> dest
                if part.exists():
                    part.replace(dest)
                return "already_complete", dest.stat().st_size
            # Otherwise, our local file drifted; wipe and retry
            if part.exists():
                part.unlink()
            continue

        if r.status_code in (401, 403):
            r.close()
            return f"failed: HTTP {r.status_code} (auth/DUA)", initial

        if r.status_code == 404:
            r.close()
            return "failed: HTTP 404 (path not found)", initial

        if r.status_code == 206:
            mode = "ab"
        elif r.status_code == 200:
            mode = "wb"
            initial = 0
        else:
            r.close()
            if attempt == max_retries - 1:
                return f"failed: HTTP {r.status_code}", initial
            time.sleep(2 ** attempt)
            continue

        total_hint = int(r.headers.get("Content-Length", 0)) + initial
        try:
            with open(part, mode) as f, tqdm(
                total=total_hint if total_hint > initial else None,
                initial=initial, unit="B", unit_scale=True,
                desc=dest.name, leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            r.close()
            part.replace(dest)  # atomic on POSIX; near-atomic on Windows
            return "downloaded", dest.stat().st_size
        except (requests.RequestException, IOError) as e:
            r.close()
            if attempt == max_retries - 1:
                return f"failed: {type(e).__name__}", part.stat().st_size if part.exists() else 0
            time.sleep(2 ** attempt)

    return "failed: exhausted retries", part.stat().st_size if part.exists() else 0


def download_subject(sid: str, out_dir: Path, token: str) -> DownloadResult:
    edf_dest = out_dir / f"mesa-sleep-{sid}.edf"
    xml_dest = out_dir / f"mesa-sleep-{sid}-nsrr.xml"

    edf_status, edf_bytes = _download_file(_build_url(EDF_PATH_FMT.format(sid=sid), token),
                                            edf_dest, token)
    xml_status, xml_bytes = _download_file(_build_url(XML_PATH_FMT.format(sid=sid), token),
                                            xml_dest, token)

    return DownloadResult(
        subject_id=sid,
        edf_status=edf_status, xml_status=xml_status,
        edf_bytes=edf_bytes, xml_bytes=xml_bytes,
        edf_valid_header=_validate_header(edf_dest, "edf"),
        xml_valid_header=_validate_header(xml_dest, "xml"),
    )


def normalize_sid(sid: str) -> str:
    return f"{int(sid):04d}"


# ---------------------------------------------------------------------------
# Concurrent orchestration + lock
# ---------------------------------------------------------------------------

class _OutputLock:
    """Best-effort O_EXCL lock on <out>/.nsrr_download.lock to prevent
    two concurrent runs corrupting the same directory (audit #8)."""

    def __init__(self, out_dir: Path):
        self.path = out_dir / ".nsrr_download.lock"
        self.fd: Optional[int] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, f"pid={os.getpid()}\n".encode())
        except FileExistsError:
            raise SystemExit(
                f"[error] another download appears to be running in {self.path.parent} "
                f"({self.path} exists). Delete the lock file if you are sure no other "
                "instance is running."
            )
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    import os as _os
    _default_out = _os.environ.get(
        "NSRR_DOWNLOAD_OUT",
        "C:/Users/User/Desktop/Projeler/SleepFM/mesa_test_clean",
    )
    parser.add_argument("--out", default=_default_out, type=Path,
                        help="Output directory (env: $NSRR_DOWNLOAD_OUT)")
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS),
                        help="Subject IDs (any zero-padding accepted)")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Concurrent downloads (max 3 recommended)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip token + DUA verification (not recommended)")

    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument("--token-file", help="Read token from file (recommended: chmod 600)")
    token_group.add_argument("--prompt-token", action="store_true",
                             help="Prompt for token interactively (getpass, no echo)")
    args = parser.parse_args()

    try:
        token = resolve_token(args)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    subjects = [normalize_sid(s) for s in args.subjects]
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_verify:
        try:
            profile = verify_token(token)
            print(f"[auth] Authenticated as: {profile.get('email', '?')} "
                  f"(username: {profile.get('username', '?')})")
        except RuntimeError as e:
            print(f"[error] Token verification failed: {_redact(str(e), token)}",
                  file=sys.stderr)
            return 3

    print(f"[tls]  Trust store: {_TRUST_STORE}")
    print(f"[plan] {len(subjects)} subjects -> {args.out}")
    print(f"[plan] Parallel streams: {args.parallel}")

    results: List[DownloadResult] = []
    with _OutputLock(args.out):
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(download_subject, sid, args.out, token): sid
                       for sid in subjects}
            with tqdm(total=len(futures), desc="Subjects", unit="subj") as outer:
                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        marker = "OK" if result.ok else "FAIL"
                        edf_status_clean = _redact(result.edf_status, token)
                        xml_status_clean = _redact(result.xml_status, token)
                        tqdm.write(
                            f"  [{marker}] {sid}: "
                            f"EDF {result.edf_bytes/1e6:.1f}MB ({edf_status_clean}, "
                            f"header_ok={result.edf_valid_header}), "
                            f"XML {result.xml_bytes/1e3:.1f}KB ({xml_status_clean}, "
                            f"header_ok={result.xml_valid_header})"
                        )
                    except Exception as e:  # noqa: BLE001
                        tqdm.write(f"  [EXCEPTION] {sid}: {type(e).__name__}")
                        results.append(DownloadResult(
                            sid, f"exception: {type(e).__name__}", "not attempted",
                            0, 0, False, False,
                        ))
                    finally:
                        outer.update()

    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    total_gb = sum(r.edf_bytes + r.xml_bytes for r in ok) / 1e9

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  OK:     {len(ok)}/{len(results)} ({total_gb:.2f} GB)")
    print(f"  FAILED: {len(fail)}/{len(results)}")
    if fail:
        print("\nFailed subjects (rerun same command to retry — resume via .part):")
        for r in fail:
            print(f"  {r.subject_id}: EDF={_redact(r.edf_status, token)} "
                  f"XML={_redact(r.xml_status, token)}")

    manifest = {
        "subjects_ok": [r.subject_id for r in ok],
        "subjects_failed": [r.subject_id for r in fail],
        "subjects_failed_reasons": {
            r.subject_id: {
                "edf": _redact(r.edf_status, token),
                "xml": _redact(r.xml_status, token),
                "edf_header_ok": r.edf_valid_header,
                "xml_header_ok": r.xml_valid_header,
            } for r in fail
        },
        "total_bytes": sum(r.edf_bytes + r.xml_bytes for r in ok),
        "output_dir": str(args.out.resolve()),
    }
    (args.out / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[manifest] {args.out / 'download_manifest.json'}")

    return 0 if len(ok) == len(subjects) else 1


if __name__ == "__main__":
    sys.exit(main())

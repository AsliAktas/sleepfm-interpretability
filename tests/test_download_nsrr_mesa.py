"""Unit tests for scripts/download_nsrr_mesa.py — pure helpers only.

Network I/O paths (verify_token, _download_file) are integration-tested
manually by running the script with an invalid token: it must correctly reach
sleepdata.org and receive {"authenticated": False}. Those paths are not
mocked here to keep the test suite fast and deterministic.
"""

import sys
from pathlib import Path

# scripts/ is not a package; add it to sys.path
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import pytest

from download_nsrr_mesa import (
    DownloadResult,
    DEFAULT_SUBJECTS,
    EDF_MAGIC,
    MIN_EDF_BYTES,
    MIN_XML_BYTES,
    XML_MAGIC_PREFIXES,
    _build_url,
    _redact,
    _validate_header,
    normalize_sid,
)


class TestNormalizeSid:
    # Non-MESA-range synthetic IDs (9xxx). NSRR DAUA §5 prohibits real subject
    # IDs in tracked source, even as test fixtures. The one-digit input is a
    # normalisation-corner test — the expected output is derived via the same
    # zero-padding logic under test rather than hardcoded, so it doesn't leak
    # a real MESA ID into the source.
    @pytest.mark.parametrize("raw", ["999", "9999", "9001", "1", 9001])
    def test_zero_pads_to_four_digits(self, raw):
        s = str(raw)
        expected = s.zfill(4)
        assert normalize_sid(s) == expected

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            normalize_sid("abc")


class TestDefaultSubjects:
    def test_is_empty_per_dua(self):
        # NSRR DAUA §5 — subject lists are Data Elements and cannot be
        # hardcoded in tracked source. Phase 18 emptied DEFAULT_SUBJECTS
        # and made --subjects effectively required.
        assert DEFAULT_SUBJECTS == ()

    def test_all_normalized(self):
        # Empty tuple trivially satisfies "every id is normalized"; kept
        # so the guarantee re-fires if someone re-adds ids in the future.
        for sid in DEFAULT_SUBJECTS:
            assert sid == normalize_sid(sid), f"{sid} is not canonical form"

    def test_no_duplicates(self):
        assert len(set(DEFAULT_SUBJECTS)) == len(DEFAULT_SUBJECTS)


class TestDownloadResult:
    def _ok_result(self, **overrides):
        base = dict(
            subject_id="9001", edf_status="downloaded", xml_status="downloaded",
            edf_bytes=MIN_EDF_BYTES + 1, xml_bytes=MIN_XML_BYTES + 1,
            edf_valid_header=True, xml_valid_header=True,
        )
        base.update(overrides)
        return DownloadResult(**base)

    def test_ok_when_both_downloaded_sized_and_valid(self):
        assert self._ok_result().ok

    def test_ok_when_already_complete(self):
        assert self._ok_result(edf_status="already_complete",
                               xml_status="already_complete").ok

    def test_not_ok_when_edf_too_small(self):
        assert not self._ok_result(edf_bytes=1000).ok

    def test_not_ok_when_xml_too_small(self):
        assert not self._ok_result(xml_bytes=100).ok

    def test_not_ok_when_status_failed(self):
        assert not self._ok_result(edf_status="failed: HTTP 404").ok

    def test_not_ok_when_header_invalid(self):
        """A 200 MB HTML error page passes size check but must fail header check."""
        assert not self._ok_result(edf_valid_header=False).ok
        assert not self._ok_result(xml_valid_header=False).ok


class TestRedact:
    def test_removes_exact_token(self):
        token = "supersecret123456789"
        msg = f"Failed calling https://x/y?auth_token={token} boom"
        redacted = _redact(msg, token)
        assert token not in redacted
        assert "***TOKEN_REDACTED***" in redacted or "***REDACTED***" in redacted

    def test_masks_query_string_even_if_token_differs(self):
        """Defense in depth: if the library rewrote the URL somehow, we still
        catch the ?auth_token= pattern with a different value."""
        msg = "url: https://x/y?auth_token=SOMETHING_ELSE_LONG_ENOUGH"
        redacted = _redact(msg, "different_token")
        assert "SOMETHING_ELSE_LONG_ENOUGH" not in redacted

    def test_empty_message_passes_through(self):
        assert _redact("", "token") == ""


class TestBuildUrl:
    def test_embeds_token_in_query(self):
        url = _build_url("datasets/mesa/files/foo.edf", "TOK")
        assert "auth_token=TOK" in url
        assert url.startswith("https://sleepdata.org/")


class TestValidateHeader:
    def test_edf_magic_accepted(self, tmp_path):
        p = tmp_path / "test.edf"
        p.write_bytes(EDF_MAGIC + b"\x00" * 100)
        assert _validate_header(p, "edf")

    def test_edf_rejects_html_error_page(self, tmp_path):
        p = tmp_path / "test.edf"
        p.write_bytes(b"<!DOCTYPE html>\n<html>...</html>" + b" " * 100)
        assert not _validate_header(p, "edf")

    def test_xml_magic_accepted(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_bytes(b'<?xml version="1.0"?>\n<PSGAnnotation></PSGAnnotation>')
        assert _validate_header(p, "xml")

    def test_xml_rejects_json_error_body(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_bytes(b'{"error": "unauthorized"}' + b" " * 50)
        assert not _validate_header(p, "xml")

    def test_missing_file_is_invalid(self, tmp_path):
        assert not _validate_header(tmp_path / "nope.edf", "edf")

    def test_tiny_file_is_invalid(self, tmp_path):
        p = tmp_path / "tiny.edf"
        p.write_bytes(b"0")
        assert not _validate_header(p, "edf")

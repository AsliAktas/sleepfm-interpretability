"""Load real SleepFM embeddings from HDF5 output files.

Bridges the SleepFM inference output to this project's mock_data-compatible
interface. Each subject's embedding HDF5 contains 4 modality datasets (BAS, RESP,
EKG, EMG), each shaped (n_chunks, 128). This module aggregates chunk-level
embeddings into per-subject vectors and exposes both single-modality (128-dim,
mock-compatible) and multi-modality (4x128 concatenated, 512-dim) views.

Aggregation modes:
- "spherical_mean" (default): normalize each chunk to unit length, average, then
  re-normalize. Chunk-count-invariant on the unit sphere — a subject with
  167 chunks does not systematically get a shorter mean vector than one with
  107, so the L2 renormalization step does not amplify differently across
  subjects. Recommended for downstream metric-based analysis (KNN, HDBSCAN).
- "mean": arithmetic mean of raw chunk vectors. Chunk vectors point in
  different directions and partially cancel, so subjects with more chunks get
  systematically smaller means. Kept for backward compatibility.
- "median": elementwise median. Robust to outlier chunks (wake, artifact).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from paths import resolve_cohort
from utils import normalize_l2

logger = logging.getLogger(__name__)

SLEEPFM_MODALITIES: Tuple[str, ...] = ("BAS", "RESP", "EKG", "EMG")

_VALID_AGGREGATES = {"spherical_mean", "mean", "median"}

# MESA subject IDs are 4-5 digit zero-padded integers; the pattern requires
# the id to be immediately followed by a known token ("_embeddings",
# "-nsrr", or end of stem) so ambiguous / versioned filenames like
# "mesa-sleep-REDACTED-v2_embeddings" fail loudly (audit finding 2). Single
# canonical parser — do not reinvent split()-based extraction elsewhere.
_MESA_SID_RE = re.compile(r"mesa-sleep-(\d{4,5})(?:_embeddings|-nsrr|\Z)")


def _resolved_embedding_dir(embedding_dir: Optional[Path]) -> Path:
    if embedding_dir is not None:
        return Path(embedding_dir)
    return resolve_cohort().embedding_dir


def _read_subject_hdf5(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {m: f[m][()] for m in SLEEPFM_MODALITIES if m in f}


def _subject_id_from_filename(path: Path) -> str:
    """Extract the MESA subject id from a SleepFM output filename.

    Uses a strict regex so retry / versioned dumps like
    'mesa-sleep-REDACTED-v2_embeddings.hdf5' fail loudly instead of quietly
    returning 'v2' as the subject id (audit finding 2).
    """
    m = _MESA_SID_RE.search(path.stem)
    if not m:
        raise ValueError(
            f"cannot parse MESA subject id from {path.name!r}; "
            "expected 'mesa-sleep-<4-5 digits>...'"
        )
    return m.group(1)


def _aggregate_chunks(chunks: np.ndarray, mode: str) -> np.ndarray:
    """Reduce a (n_chunks, dim) array to (dim,) using the named strategy."""
    if mode == "spherical_mean":
        norms = np.linalg.norm(chunks, axis=1, keepdims=True) + 1e-8
        unit = chunks / norms
        m = unit.mean(axis=0)
        return m / (np.linalg.norm(m) + 1e-8)
    if mode == "mean":
        return chunks.mean(axis=0)
    if mode == "median":
        return np.median(chunks, axis=0)
    raise ValueError(f"unknown aggregate '{mode}', expected one of {sorted(_VALID_AGGREGATES)}")


def load_subject_embeddings(
    embedding_dir: Optional[Path] = None,
    modality: str = "BAS",
    aggregate: str = "spherical_mean",
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Load per-subject embeddings for a single modality (mock-compatible 128-dim).

    Returns (X, subject_ids) where X has shape (n_subjects, 128).
    The default "spherical_mean" aggregation is chunk-count-invariant on the
    unit sphere; use "mean" or "median" only when you have a specific reason.
    normalize=True L2-normalizes the aggregated vectors (redundant for
    spherical_mean, applied for mean/median).
    """
    if modality not in SLEEPFM_MODALITIES:
        raise ValueError(f"unknown modality '{modality}', expected one of {SLEEPFM_MODALITIES}")
    if aggregate not in _VALID_AGGREGATES:
        raise ValueError(f"unknown aggregate '{aggregate}', expected one of {sorted(_VALID_AGGREGATES)}")

    embedding_dir = _resolved_embedding_dir(embedding_dir)
    if not embedding_dir.exists():
        raise FileNotFoundError(f"embedding_dir does not exist: {embedding_dir}")

    files = sorted(embedding_dir.glob("*_embeddings.hdf5"))
    if not files:
        raise FileNotFoundError(f"no *_embeddings.hdf5 files found in {embedding_dir}")

    subject_ids: List[str] = []
    vectors: List[np.ndarray] = []
    for path in files:
        data = _read_subject_hdf5(path)
        if modality not in data or data[modality].size == 0:
            continue
        vectors.append(_aggregate_chunks(data[modality], aggregate))
        subject_ids.append(_subject_id_from_filename(path))

    X = np.stack(vectors).astype(np.float32)
    if normalize:
        X = normalize_l2(X)
    return X, subject_ids


def load_subject_embeddings_multimodal(
    embedding_dir: Optional[Path] = None,
    aggregate: str = "spherical_mean",
    normalize_per_modality: bool = True,
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
    """Load per-subject 4-modality concatenated embeddings (512-dim).

    Returns (X, subject_ids, skipped) where X has shape (n_subjects, 512) as
    [BAS(128) | RESP(128) | EKG(128) | EMG(128)]. Each modality is L2-normalized
    before concatenation so no single modality dominates the joint distance.

    `skipped` lists (filename, missing_modality) pairs for subjects dropped due
    to missing data — surfaced so callers can compare cohort sizes across
    single-modality vs multimodal analyses.
    """
    if aggregate not in _VALID_AGGREGATES:
        raise ValueError(f"unknown aggregate '{aggregate}', expected one of {sorted(_VALID_AGGREGATES)}")

    embedding_dir = _resolved_embedding_dir(embedding_dir)
    files = sorted(embedding_dir.glob("*_embeddings.hdf5"))
    if not files:
        raise FileNotFoundError(f"no *_embeddings.hdf5 files found in {embedding_dir}")

    subject_ids: List[str] = []
    vectors: List[np.ndarray] = []
    skipped: List[Tuple[str, str]] = []
    for path in files:
        data = _read_subject_hdf5(path)
        missing = None
        parts = []
        for m in SLEEPFM_MODALITIES:
            if m not in data or data[m].size == 0:
                missing = m
                break
            v = _aggregate_chunks(data[m], aggregate).astype(np.float32)
            if normalize_per_modality:
                v = v / (np.linalg.norm(v) + 1e-8)
            parts.append(v)
        if missing is not None:
            skipped.append((path.name, missing))
            continue
        vectors.append(np.concatenate(parts))
        subject_ids.append(_subject_id_from_filename(path))

    if skipped:
        logger.warning(
            "load_subject_embeddings_multimodal: %d hasta atlandi (eksik modalite): %s",
            len(skipped), skipped,
        )
    X = np.stack(vectors).astype(np.float32)
    return X, subject_ids, skipped


def load_chunk_embeddings(
    embedding_dir: Optional[Path] = None,
    modality: str = "BAS",
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Load per-chunk embeddings for a single modality (no aggregation).

    Returns (X, subject_ids) where X has shape (n_chunks_total, 128) and
    subject_ids has length n_chunks_total (one id per chunk, repeated). Useful
    for cluster-vs-subject purity analysis at 5-min resolution.
    """
    if modality not in SLEEPFM_MODALITIES:
        raise ValueError(f"unknown modality '{modality}', expected one of {SLEEPFM_MODALITIES}")

    embedding_dir = _resolved_embedding_dir(embedding_dir)
    files = sorted(embedding_dir.glob("*_embeddings.hdf5"))
    if not files:
        raise FileNotFoundError(f"no *_embeddings.hdf5 files found in {embedding_dir}")

    subject_ids: List[str] = []
    chunks: List[np.ndarray] = []
    for path in files:
        data = _read_subject_hdf5(path)
        if modality not in data or data[modality].size == 0:
            continue
        arr = data[modality].astype(np.float32)
        chunks.append(arr)
        sid = _subject_id_from_filename(path)
        subject_ids.extend([sid] * arr.shape[0])

    X = np.concatenate(chunks, axis=0)
    if normalize:
        X = normalize_l2(X)
    return X, subject_ids

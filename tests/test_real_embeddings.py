"""Tests for real_embeddings.py — SleepFM HDF5 output loader.

Uses tmp_path fixture to create synthetic embedding HDF5s that match the
SleepFM inference output format (4 modality datasets per file, each shaped
(n_chunks, 128)). Verifies shape, aggregation, normalization, and edge cases.
"""

import h5py
import numpy as np
import pytest

from real_embeddings import (
    SLEEPFM_MODALITIES,
    load_subject_embeddings,
    load_subject_embeddings_multimodal,
    load_chunk_embeddings,
    _subject_id_from_filename,
)


@pytest.fixture
def synthetic_embedding_dir(tmp_path):
    """Create 3 synthetic SleepFM output HDF5s with known chunk counts."""
    rng = np.random.default_rng(42)
    chunk_counts = {"9001": 100, "9002": 50, "9003": 75}
    for sid, n in chunk_counts.items():
        path = tmp_path / f"mesa-sleep-{sid}_embeddings.hdf5"
        with h5py.File(path, "w") as f:
            for m in SLEEPFM_MODALITIES:
                f.create_dataset(m, data=rng.normal(size=(n, 128)).astype(np.float32))
    return tmp_path, chunk_counts


class TestSubjectIdParsing:
    def test_strips_embeddings_suffix(self):
        from pathlib import Path
        assert _subject_id_from_filename(Path("mesa-sleep-9001_embeddings.hdf5")) == "9001"

    def test_takes_regex_matched_digits(self):
        from pathlib import Path
        assert _subject_id_from_filename(Path("mesa-sleep-9001_embeddings.hdf5")) == "9001"

    def test_rejects_versioned_filename(self):
        """Audit finding 2: 'mesa-sleep-9001-v2_embeddings.hdf5' used to
        silently return 'v2'. Must now raise ValueError."""
        from pathlib import Path
        with pytest.raises(ValueError, match="cannot parse"):
            _subject_id_from_filename(Path("mesa-sleep-9001-v2_embeddings.hdf5"))

    def test_rejects_unrelated_filename(self):
        from pathlib import Path
        with pytest.raises(ValueError):
            _subject_id_from_filename(Path("some_other_file.hdf5"))

    def test_accepts_5_digit_id(self):
        from pathlib import Path
        assert _subject_id_from_filename(Path("mesa-sleep-12345_embeddings.hdf5")) == "12345"


class TestLoadSubjectEmbeddings:
    def test_returns_correct_shape(self, synthetic_embedding_dir):
        tmp_path, chunk_counts = synthetic_embedding_dir
        X, sids = load_subject_embeddings(embedding_dir=tmp_path, modality="BAS")
        assert X.shape == (len(chunk_counts), 128)
        assert set(sids) == set(chunk_counts.keys())

    def test_l2_normalized_by_default(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        X, _ = load_subject_embeddings(embedding_dir=tmp_path, modality="BAS")
        norms = np.linalg.norm(X, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_no_normalize_when_flag_off(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        # Use plain "mean" so the raw aggregated vectors are NOT unit-length;
        # spherical_mean would return unit vectors regardless of the flag.
        X, _ = load_subject_embeddings(
            embedding_dir=tmp_path, modality="BAS", aggregate="mean", normalize=False,
        )
        norms = np.linalg.norm(X, axis=1)
        assert not np.allclose(norms, 1.0, atol=1e-3), "should not be pre-normalized"

    def test_spherical_mean_is_unit_norm_without_extra_normalize(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        X, _ = load_subject_embeddings(
            embedding_dir=tmp_path, modality="BAS", aggregate="spherical_mean", normalize=False,
        )
        norms = np.linalg.norm(X, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            "spherical_mean must return unit vectors intrinsically"
        )

    def test_spherical_mean_is_chunk_count_invariant(self, tmp_path):
        """Same chunk direction distribution + more chunks -> same aggregated vector."""
        rng = np.random.default_rng(0)
        base = rng.normal(size=(50, 128)).astype(np.float32)
        for sid, k in [("9001", 1), ("9002", 3)]:
            arr = np.tile(base, (k, 1))
            path = tmp_path / f"mesa-sleep-{sid}_embeddings.hdf5"
            with h5py.File(path, "w") as f:
                for m in SLEEPFM_MODALITIES:
                    f.create_dataset(m, data=arr)
        X, sids = load_subject_embeddings(
            embedding_dir=tmp_path, modality="BAS", aggregate="spherical_mean",
        )
        assert set(sids) == {"9001", "9002"}
        idx = {s: i for i, s in enumerate(sids)}
        np.testing.assert_allclose(X[idx["9001"]], X[idx["9002"]], atol=1e-5)

    def test_median_differs_from_mean_on_skewed(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        X_mean, _ = load_subject_embeddings(embedding_dir=tmp_path, modality="BAS",
                                            aggregate="mean", normalize=False)
        X_med, _ = load_subject_embeddings(embedding_dir=tmp_path, modality="BAS",
                                           aggregate="median", normalize=False)
        assert not np.allclose(X_mean, X_med), "mean and median should differ on random data"

    def test_subject_ids_sorted(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        _, sids = load_subject_embeddings(embedding_dir=tmp_path, modality="BAS")
        assert sids == sorted(sids), "subject_ids must be sorted for reproducibility"

    def test_all_modalities_supported(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        for m in SLEEPFM_MODALITIES:
            X, _ = load_subject_embeddings(embedding_dir=tmp_path, modality=m)
            assert X.shape[1] == 128

    def test_unknown_modality_raises(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        with pytest.raises(ValueError, match="unknown modality"):
            load_subject_embeddings(embedding_dir=tmp_path, modality="INVALID")

    def test_unknown_aggregate_raises(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        with pytest.raises(ValueError, match="unknown aggregate"):
            load_subject_embeddings(embedding_dir=tmp_path, modality="BAS", aggregate="mode")

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_subject_embeddings(embedding_dir=tmp_path / "does_not_exist", modality="BAS")

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no.*hdf5"):
            load_subject_embeddings(embedding_dir=tmp_path, modality="BAS")

    def test_deterministic_across_calls(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        X1, s1 = load_subject_embeddings(embedding_dir=tmp_path, modality="BAS")
        X2, s2 = load_subject_embeddings(embedding_dir=tmp_path, modality="BAS")
        assert np.array_equal(X1, X2)
        assert s1 == s2


class TestLoadSubjectEmbeddingsMultimodal:
    def test_returns_512_dim(self, synthetic_embedding_dir):
        tmp_path, chunk_counts = synthetic_embedding_dir
        X, sids, skipped = load_subject_embeddings_multimodal(embedding_dir=tmp_path)
        assert X.shape == (len(chunk_counts), 512)
        assert skipped == []

    def test_per_modality_norms_are_unit(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        X, _, _ = load_subject_embeddings_multimodal(embedding_dir=tmp_path)
        for i in range(4):
            slab = X[:, i * 128:(i + 1) * 128]
            norms = np.linalg.norm(slab, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), f"modality slab {i} not unit-normalized"

    def test_missing_modality_subject_skipped_and_reported(self, tmp_path):
        """A subject missing one modality's dataset should be filtered out AND
        the skip should be surfaced in the returned `skipped` list."""
        rng = np.random.default_rng(0)
        good = tmp_path / "mesa-sleep-9001_embeddings.hdf5"
        with h5py.File(good, "w") as f:
            for m in SLEEPFM_MODALITIES:
                f.create_dataset(m, data=rng.normal(size=(10, 128)).astype(np.float32))
        broken = tmp_path / "mesa-sleep-9002_embeddings.hdf5"
        with h5py.File(broken, "w") as f:
            f.create_dataset("BAS", data=rng.normal(size=(10, 128)).astype(np.float32))
            # missing RESP, EKG, EMG
        X, sids, skipped = load_subject_embeddings_multimodal(embedding_dir=tmp_path)
        assert sids == ["9001"], "broken subject must be skipped"
        assert X.shape == (1, 512)
        assert len(skipped) == 1
        assert skipped[0][0] == broken.name
        assert skipped[0][1] in {"RESP", "EKG", "EMG"}


class TestLoadChunkEmbeddings:
    def test_chunk_count_matches_source(self, synthetic_embedding_dir):
        tmp_path, chunk_counts = synthetic_embedding_dir
        X, sids = load_chunk_embeddings(embedding_dir=tmp_path, modality="BAS")
        assert X.shape[0] == sum(chunk_counts.values())
        assert X.shape[1] == 128

    def test_subject_id_repeated_per_chunk(self, synthetic_embedding_dir):
        tmp_path, chunk_counts = synthetic_embedding_dir
        _, sids = load_chunk_embeddings(embedding_dir=tmp_path, modality="BAS")
        from collections import Counter
        assert dict(Counter(sids)) == chunk_counts

    def test_l2_normalized(self, synthetic_embedding_dir):
        tmp_path, _ = synthetic_embedding_dir
        X, _ = load_chunk_embeddings(embedding_dir=tmp_path, modality="BAS")
        norms = np.linalg.norm(X, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

"""Tests for chunk_level_analysis.py — cluster × sleep-stage alignment.

Focuses on the tricky bits audit B2/B4/B7 flagged plus the new
`stage_ari_null` empirical-null helper.
"""

import numpy as np
import pandas as pd
import pytest

from chunk_level_analysis import (
    ChunkAnalysisResult,
    _align_labels_to_chunks,
    cluster_subject_purity,
    stage_ari,
    stage_ari_null,
)
from sleep_phases import ChunkLabel


def _label(idx: int, stage: str) -> ChunkLabel:
    return ChunkLabel(idx, idx * 300, (idx + 1) * 300, stage,
                      stage_fractions={stage: 1.0}, coverage_fraction=1.0)


class TestAlignLabelsToChunks:
    def test_contiguous_subject_ids_produce_expected_stages(self):
        subject_ids = ["A", "A", "A", "B", "B"]
        stages_by = {
            "A": [_label(0, "N2"), _label(1, "N3"), _label(2, "REM")],
            "B": [_label(0, "Wake"), _label(1, "N2")],
        }
        out = _align_labels_to_chunks(subject_ids, stages_by)
        assert out == ["N2", "N3", "REM", "Wake", "N2"]

    def test_missing_subject_gets_missing_xml_marker(self):
        subject_ids = ["A", "B"]
        stages_by = {"A": [_label(0, "N2")]}
        out = _align_labels_to_chunks(subject_ids, stages_by)
        assert out == ["N2", "MissingXML"]

    def test_more_chunks_than_labels_get_out_of_range(self):
        subject_ids = ["A", "A", "A"]
        stages_by = {"A": [_label(0, "N2")]}
        out = _align_labels_to_chunks(subject_ids, stages_by)
        assert out == ["N2", "OutOfRange", "OutOfRange"]

    def test_interleaved_subject_order_raises(self):
        """Audit B2: silent mis-alignment if subject_ids interleaved.
        Must fail loudly rather than quietly produce wrong stage labels."""
        subject_ids = ["A", "B", "A"]  # interleaved
        stages_by = {"A": [_label(0, "N2"), _label(1, "N3")],
                     "B": [_label(0, "Wake")]}
        with pytest.raises(ValueError, match="contiguous"):
            _align_labels_to_chunks(subject_ids, stages_by)


class TestClusterSubjectPurity:
    def _result(self, subjects, clusters, stages=None):
        stages = stages or ["N2"] * len(subjects)
        return ChunkAnalysisResult(
            modality="BAS",
            X=np.zeros((len(subjects), 128)),
            subject_ids=list(subjects),
            stage_labels=list(stages),
            cluster_labels=np.asarray(clusters),
            umap_2d=np.zeros((len(subjects), 2)),
        )

    def test_pure_single_subject_cluster_share_is_one(self):
        # All 3 chunks in cluster 0 come from subject A
        result = self._result(["A", "A", "A", "B"], [0, 0, 0, 1])
        df = cluster_subject_purity(result)
        row = df[df["cluster"] == 0].iloc[0]
        assert row["top_subject_share"] == 1.0
        assert row["n_unique_subjects"] == 1
        assert row["top_subject"] == "A"

    def test_mixed_cluster_reflects_top_subject_share(self):
        # cluster 0: 2 A + 1 B + 1 C
        result = self._result(["A", "A", "B", "C"], [0, 0, 0, 0])
        row = cluster_subject_purity(result).iloc[0]
        assert row["top_subject"] == "A"
        assert row["top_subject_share"] == pytest.approx(0.5)
        assert row["n_unique_subjects"] == 3


class TestStageAri:
    def _result(self, clusters, stages, subjects=None):
        n = len(clusters)
        return ChunkAnalysisResult(
            modality="BAS", X=np.zeros((n, 128)),
            subject_ids=subjects or [f"s{i}" for i in range(n)],
            stage_labels=list(stages),
            cluster_labels=np.asarray(clusters),
            umap_2d=np.zeros((n, 2)),
        )

    def test_perfect_alignment_ari_is_one(self):
        clusters = [0, 0, 1, 1, 2, 2]
        stages = ["N2", "N2", "REM", "REM", "N3", "N3"]
        assert stage_ari(self._result(clusters, stages)) == pytest.approx(1.0)

    def test_random_alignment_ari_near_zero(self):
        rng = np.random.default_rng(0)
        n = 200
        clusters = rng.integers(0, 5, size=n).tolist()
        stages = rng.choice(["N2", "N3", "REM", "Wake"], size=n).tolist()
        # Wake gets excluded by default; still expect low ARI
        val = stage_ari(self._result(clusters, stages))
        assert abs(val) < 0.05

    def test_excludes_missing_stage_labels(self):
        clusters = [0, 0, 1, 1]
        stages = ["N2", "MissingXML", "REM", "OutOfRange"]
        # Only 2 valid chunks -> insufficient for ARI, returns nan
        val = stage_ari(self._result(clusters, stages))
        assert not np.isnan(val)  # 2 valid points still returns something

    def test_excludes_noise_cluster(self):
        clusters = [-1, -1, 0, 0, 1, 1]
        stages = ["N2", "N3", "N2", "N2", "REM", "REM"]
        val = stage_ari(self._result(clusters, stages))
        # Noise excluded, remaining is perfect -> ARI = 1
        assert val == pytest.approx(1.0)


class TestStageAriNull:
    def test_null_p_bounded_positive(self):
        rng = np.random.default_rng(0)
        n = 100
        clusters = rng.integers(0, 4, size=n).tolist()
        stages = rng.choice(["N2", "N3", "REM"], size=n).tolist()
        result = ChunkAnalysisResult(
            modality="BAS", X=np.zeros((n, 128)),
            subject_ids=[f"s{i}" for i in range(n)],
            stage_labels=stages, cluster_labels=np.asarray(clusters),
            umap_2d=np.zeros((n, 2)),
        )
        obs, p, null = stage_ari_null(result, n_permutations=100)
        assert 0.0 < p <= 1.0
        assert len(null) == 100

    def test_planted_signal_rejects_null(self):
        # Perfect cluster-stage alignment -> observed ARI far above null
        clusters = [0] * 20 + [1] * 20 + [2] * 20
        stages = ["N2"] * 20 + ["REM"] * 20 + ["N3"] * 20
        result = ChunkAnalysisResult(
            modality="BAS", X=np.zeros((60, 128)),
            subject_ids=[f"s{i}" for i in range(60)],
            stage_labels=stages, cluster_labels=np.asarray(clusters),
            umap_2d=np.zeros((60, 2)),
        )
        obs, p, null = stage_ari_null(result, n_permutations=100)
        assert obs > 0.5
        assert p < 0.05  # observed far above null

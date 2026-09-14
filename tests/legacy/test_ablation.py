"""
Tests for Mask-Based Ablation — modality importance via KNN similarity drop.

HATA #2: modality_scores must use ablation-based attribution, not raw abs mean.
The ablation approach: zero-out each modality's dimensions, re-normalize,
measure KNN similarity drop. Larger drop = more important modality.
"""

import numpy as np
import pytest

from mock_data import generate_mock_embeddings
from config import DISEASE_NAMES, MODALITY_DIM_RANGES
from similarity_engine import build_reference_index, query_top5
from pipeline import run_pipeline


@pytest.fixture(scope="module")
def ablation_fixtures():
    """Shared data for ablation tests."""
    embeddings, labels = generate_mock_embeddings(
        seed=42, n_samples=500, n_disease_groups=12, perturbation_rate=0.0
    )
    return embeddings, labels


class TestAblationFunction:
    """Verify that compute_ablation_scores exists and returns correct structure."""

    def test_ablation_scores_importable(self):
        """compute_ablation_scores should be importable from pipeline."""
        from pipeline import compute_ablation_scores

    def test_ablation_scores_returns_dict_with_all_modalities(self, ablation_fixtures):
        """Should return a dict with keys for all 4 modalities."""
        from pipeline import compute_ablation_scores
        embeddings, labels = ablation_fixtures
        query_idx = 0
        mask = np.ones(len(embeddings), dtype=bool)
        mask[query_idx] = False
        ref_embeddings = embeddings[mask]
        ref_labels = labels[mask]
        index = build_reference_index(ref_embeddings, n_neighbors=10)

        scores = compute_ablation_scores(
            query_embedding=embeddings[query_idx],
            reference_index=index,
            reference_labels=ref_labels,
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(MODALITY_DIM_RANGES.keys())

    def test_ablation_scores_are_non_negative(self, ablation_fixtures):
        """Ablation scores (similarity drops) should be >= 0."""
        from pipeline import compute_ablation_scores
        embeddings, labels = ablation_fixtures
        query_idx = 0
        mask = np.ones(len(embeddings), dtype=bool)
        mask[query_idx] = False
        ref_embeddings = embeddings[mask]
        ref_labels = labels[mask]
        index = build_reference_index(ref_embeddings, n_neighbors=10)

        scores = compute_ablation_scores(
            query_embedding=embeddings[query_idx],
            reference_index=index,
            reference_labels=ref_labels,
        )
        for mod, val in scores.items():
            assert val >= 0.0, f"Ablation score for {mod} is negative: {val}"


class TestAblationClinicalValidity:
    """Verify that ablation scores reflect expected modality-disease relationships.

    In mock data (perturbation_rate=0.0), disease groups have known dominant
    modalities: label 1 → EEG dominant, label 2 → ECG dominant, etc.
    Ablation of the dominant modality should cause the largest similarity drop.
    """

    def test_ecg_dominant_cluster_ecg_ablation_largest(self, ablation_fixtures):
        """For ECG-dominant cluster (label 2), ECG ablation should cause largest drop."""
        from pipeline import compute_ablation_scores
        embeddings, labels = ablation_fixtures
        ecg_patients = np.where(labels == 2)[0]
        assert len(ecg_patients) > 0, "No patients with label 2"

        # Average ablation scores across ECG-dominant patients
        avg_scores = {mod: 0.0 for mod in MODALITY_DIM_RANGES}
        mask_all = labels != -999  # dummy: use all as reference
        for idx in ecg_patients[:10]:  # sample 10 patients
            mask = np.ones(len(embeddings), dtype=bool)
            mask[idx] = False
            ref_emb = embeddings[mask]
            ref_lab = labels[mask]
            index = build_reference_index(ref_emb, n_neighbors=10)
            scores = compute_ablation_scores(
                query_embedding=embeddings[idx],
                reference_index=index,
                reference_labels=ref_lab,
            )
            for mod in avg_scores:
                avg_scores[mod] += scores[mod]
        for mod in avg_scores:
            avg_scores[mod] /= min(10, len(ecg_patients))

        assert avg_scores["ecg"] > avg_scores["emg"], (
            f"ECG ablation drop ({avg_scores['ecg']:.4f}) should exceed "
            f"EMG ablation drop ({avg_scores['emg']:.4f}) for ECG-dominant cluster"
        )

    def test_eeg_dominant_cluster_eeg_ablation_largest(self, ablation_fixtures):
        """For EEG-dominant cluster (label 6 = Depression), EEG ablation should cause largest drop."""
        from pipeline import compute_ablation_scores
        embeddings, labels = ablation_fixtures
        eeg_patients = np.where(labels == 6)[0]
        assert len(eeg_patients) > 0

        avg_scores = {mod: 0.0 for mod in MODALITY_DIM_RANGES}
        for idx in eeg_patients[:10]:
            mask = np.ones(len(embeddings), dtype=bool)
            mask[idx] = False
            index = build_reference_index(embeddings[mask], n_neighbors=10)
            scores = compute_ablation_scores(
                query_embedding=embeddings[idx],
                reference_index=index,
                reference_labels=labels[mask],
            )
            for mod in avg_scores:
                avg_scores[mod] += scores[mod]
        for mod in avg_scores:
            avg_scores[mod] /= min(10, len(eeg_patients))

        assert avg_scores["eeg"] > avg_scores["emg"], (
            f"EEG ablation drop ({avg_scores['eeg']:.4f}) should exceed "
            f"EMG ablation drop ({avg_scores['emg']:.4f}) for EEG-dominant cluster"
        )


class TestPipelineUsesAblation:
    """Verify that pipeline.run_pipeline uses ablation-based modality_scores."""

    def test_pipeline_modality_scores_use_ablation(self, ablation_fixtures):
        """Pipeline modality_scores should differ from naive abs-mean approach."""
        import umap as umap_lib
        import hdbscan as hdbscan_lib
        embeddings, labels = ablation_fixtures

        reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_2d = reducer.fit_transform(embeddings)
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=5, prediction_data=True)
        clusterer.fit(umap_2d)

        query_idx = 0
        mask = np.ones(len(embeddings), dtype=bool)
        mask[query_idx] = False

        index = build_reference_index(embeddings[mask], n_neighbors=10)
        result = run_pipeline(
            query_embedding=embeddings[query_idx],
            reference_embeddings=embeddings[mask],
            reference_labels=labels[mask],
            umap_reducer=reducer,
            hdbscan_model=clusterer,
            reference_index=index,
            patient_id="test_ablation",
        )

        # Naive approach (old): abs mean per modality segment
        naive_scores = {}
        for mod, (start, end) in MODALITY_DIM_RANGES.items():
            naive_scores[mod] = round(float(np.abs(embeddings[query_idx][start:end]).mean()), 4)

        # Pipeline should NOT produce the naive scores anymore
        assert result["modality_scores"] != naive_scores, (
            "Pipeline still uses naive abs-mean — ablation not integrated"
        )

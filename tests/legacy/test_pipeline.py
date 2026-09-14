"""
Tests for pipeline.py — full patient analysis pipeline.

HATA #3: KNN index should be passed as parameter, not rebuilt every call.
HATA #4: Self-reference detection (to be added later).
"""

import numpy as np
import pytest
import umap
import hdbscan

from mock_data import generate_mock_embeddings
from similarity_engine import build_reference_index
from pipeline import run_pipeline


@pytest.fixture(scope="module")
def pipeline_fixtures():
    """Shared fixtures for pipeline tests: embeddings, UMAP, HDBSCAN."""
    embeddings, labels = generate_mock_embeddings(seed=42, n_samples=100, n_disease_groups=12)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    clusterer.fit(umap_2d)
    return embeddings, labels, reducer, clusterer


class TestPipelineIndexParameter:
    """Verify that run_pipeline accepts a pre-built KNN index."""

    def test_run_pipeline_accepts_reference_index(self, pipeline_fixtures):
        """run_pipeline should accept a reference_index parameter and use it."""
        embeddings, labels, reducer, clusterer = pipeline_fixtures
        index = build_reference_index(embeddings, n_neighbors=10)
        result = run_pipeline(
            query_embedding=embeddings[0],
            reference_embeddings=embeddings,
            reference_labels=labels,
            umap_reducer=reducer,
            hdbscan_model=clusterer,
            reference_index=index,
            patient_id="test_000",
        )
        assert "top5_diseases" in result
        assert len(result["top5_diseases"]) > 0

    def test_prebuilt_index_same_result_as_internal(self, pipeline_fixtures):
        """Pre-built index should produce identical results to internal build."""
        embeddings, labels, reducer, clusterer = pipeline_fixtures
        index = build_reference_index(embeddings, n_neighbors=10)
        result_with_index = run_pipeline(
            query_embedding=embeddings[0],
            reference_embeddings=embeddings,
            reference_labels=labels,
            umap_reducer=reducer,
            hdbscan_model=clusterer,
            reference_index=index,
            patient_id="test_000",
        )
        result_without_index = run_pipeline(
            query_embedding=embeddings[0],
            reference_embeddings=embeddings,
            reference_labels=labels,
            umap_reducer=reducer,
            hdbscan_model=clusterer,
            patient_id="test_000",
        )
        assert result_with_index["top5_diseases"] == result_without_index["top5_diseases"]


class TestPipelineOutput:
    """Verify pipeline output structure and JSON serializability."""

    def test_output_has_required_keys(self, pipeline_fixtures):
        """Pipeline output must contain all expected keys."""
        embeddings, labels, reducer, clusterer = pipeline_fixtures
        result = run_pipeline(
            query_embedding=embeddings[0],
            reference_embeddings=embeddings,
            reference_labels=labels,
            umap_reducer=reducer,
            hdbscan_model=clusterer,
            patient_id="test_000",
        )
        required_keys = {"patient_id", "timestamp", "embedding_norm", "umap_coords",
                         "cluster_id", "top5_diseases", "modality_scores"}
        assert required_keys.issubset(result.keys())

    def test_output_json_serializable(self, pipeline_fixtures):
        """Pipeline output must be JSON-serializable."""
        import json
        embeddings, labels, reducer, clusterer = pipeline_fixtures
        result = run_pipeline(
            query_embedding=embeddings[0],
            reference_embeddings=embeddings,
            reference_labels=labels,
            umap_reducer=reducer,
            hdbscan_model=clusterer,
            patient_id="test_000",
        )
        serialized = json.dumps(result)
        loaded = json.loads(serialized)
        assert loaded["patient_id"] == "test_000"


class TestPipelineSelfExclusion:
    """HATA #4: Caller must exclude query from reference to avoid self-reference."""

    def test_pipeline_with_excluded_query(self, pipeline_fixtures):
        """Pipeline should work correctly when query is excluded from reference."""
        embeddings, labels, reducer, clusterer = pipeline_fixtures
        query_idx = 0
        mask = np.ones(len(embeddings), dtype=bool)
        mask[query_idx] = False
        ref_embeddings = embeddings[mask]
        ref_labels = labels[mask]
        index = build_reference_index(ref_embeddings, n_neighbors=10)
        result = run_pipeline(
            query_embedding=embeddings[query_idx],
            reference_embeddings=ref_embeddings,
            reference_labels=ref_labels,
            umap_reducer=reducer,
            hdbscan_model=clusterer,
            reference_index=index,
            patient_id="test_000",
        )
        # All similarities should be < 1.0 (no self-match with distance=0)
        for r in result["top5_diseases"]:
            assert r["similarity"] < 1.0, (
                f"Similarity 1.0 detected — possible self-reference: {r}"
            )

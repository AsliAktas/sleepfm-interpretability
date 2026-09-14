"""
Tests for similarity_engine.py — KNN-based disease similarity.

HATA #1 regression: query_top5 should never return "Unknown(...)" when
labels come from generate_mock_embeddings with n_disease_groups=12.

HATA #4 regression: self-reference detection (to be added later).
"""

import numpy as np
import pytest

from mock_data import generate_mock_embeddings
from config import DISEASE_NAMES
from similarity_engine import build_reference_index, query_top5


class TestQueryTop5LabelAlignment:
    """Verify that query_top5 returns valid disease names, never 'Unknown(...)'."""

    @pytest.fixture
    def mock_data(self):
        embeddings, labels = generate_mock_embeddings(seed=42, n_disease_groups=12)
        index = build_reference_index(embeddings, n_neighbors=10)
        return embeddings, labels, index

    def test_no_unknown_in_top5(self, mock_data):
        """No result should contain 'Unknown(' in the disease name."""
        embeddings, labels, index = mock_data
        for i in range(0, len(embeddings), 50):  # sample every 50th patient
            results = query_top5(
                query_embedding=embeddings[i],
                index=index,
                reference_labels=labels,
                disease_names=DISEASE_NAMES,
            )
            for r in results:
                assert "Unknown(" not in r["disease"], (
                    f"Patient {i}: got '{r['disease']}' — label not in DISEASE_NAMES"
                )

    def test_all_returned_diseases_are_known(self, mock_data):
        """Every disease name in results must be in DISEASE_NAMES.values()."""
        embeddings, labels, index = mock_data
        valid_names = set(DISEASE_NAMES.values())
        results = query_top5(
            query_embedding=embeddings[0],
            index=index,
            reference_labels=labels,
            disease_names=DISEASE_NAMES,
        )
        for r in results:
            assert r["disease"] in valid_names, (
                f"Disease '{r['disease']}' not in DISEASE_NAMES"
            )


class TestSelfReferenceExclusion:
    """HATA #4: Query patient must NOT appear in its own KNN results.

    Approach B: Caller excludes the query from reference before calling.
    These tests verify that when reference does NOT contain the query,
    the nearest neighbor distance is > 0 (no self-match).
    """

    def test_no_zero_distance_when_query_excluded(self):
        """First neighbor distance must be > 0 when query is excluded from reference."""
        embeddings, labels = generate_mock_embeddings(seed=42, n_disease_groups=12)
        query_idx = 0
        # Exclude query from reference
        mask = np.ones(len(embeddings), dtype=bool)
        mask[query_idx] = False
        ref_embeddings = embeddings[mask]
        ref_labels = labels[mask]
        index = build_reference_index(ref_embeddings, n_neighbors=10)
        distances, _ = index.kneighbors(embeddings[query_idx].reshape(1, -1))
        assert distances[0, 0] > 0.0, "First neighbor distance is 0 — self-reference detected"

    def test_self_included_has_zero_distance(self):
        """Sanity check: when query IS in reference, first distance IS ~0 (the bug)."""
        embeddings, labels = generate_mock_embeddings(seed=42, n_disease_groups=12)
        index = build_reference_index(embeddings, n_neighbors=10)
        distances, _ = index.kneighbors(embeddings[0].reshape(1, -1))
        assert distances[0, 0] < 1e-6, (
            f"Expected near-zero self-reference distance, got {distances[0, 0]}"
        )

    def test_excluded_query_gets_10_unique_neighbors(self):
        """With query excluded, all 10 neighbors should be distinct patients."""
        embeddings, labels = generate_mock_embeddings(seed=42, n_disease_groups=12)
        query_idx = 0
        mask = np.ones(len(embeddings), dtype=bool)
        mask[query_idx] = False
        ref_embeddings = embeddings[mask]
        ref_labels = labels[mask]
        index = build_reference_index(ref_embeddings, n_neighbors=10)
        results = query_top5(
            query_embedding=embeddings[query_idx],
            index=index,
            reference_labels=ref_labels,
            disease_names=DISEASE_NAMES,
        )
        total_votes = sum(r["n_votes"] for r in results)
        assert total_votes == 10, f"Expected 10 votes total, got {total_votes}"

"""Tests for clinical_bridge_adapter.py — SleepFM embedding -> risk payload.

Network / clinical-bridge integration test is at the end and skips if the
clinical-bridge repo is not available on disk.
"""

from pathlib import Path

import numpy as np
import pytest

from clinical_bridge_adapter import (
    DEFAULT_CONDITION,
    NeighbourMatch,
    RiskPayload,
    _cosine_similarity,
    build_risk_payload,
    estimate_risk_from_neighbours,
    find_similar_patients,
    translate_to_fhir,
)


import os

CLINICAL_BRIDGE_DIR = Path(os.environ.get(
    "CLINICAL_BRIDGE_DIR",
    "/nonexistent/set-CLINICAL_BRIDGE_DIR-env-var",
))


class TestCosineSimilarity:
    def test_identical_vector_is_one(self):
        q = np.random.RandomState(0).normal(size=128)
        sims = _cosine_similarity(q, q[None, :])
        assert sims[0] == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        q = np.zeros(128); q[0] = 1
        r = np.zeros((1, 128)); r[0, 1] = 1
        sims = _cosine_similarity(q, r)
        assert sims[0] == pytest.approx(0.0)


class TestFindSimilarPatients:
    def test_returns_top_k_ordered_by_similarity(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=128)
        # Build a reference where subject B is *related* but not identical
        # (small perturbation still triggers the exclude_self > 0.9999 guard).
        # Use a moderate mix so B is clearly most similar but not "self".
        ref = np.stack([
            rng.normal(size=128),                       # A: random
            0.5 * query + 0.5 * rng.normal(size=128),   # B: half query + half noise
            rng.normal(size=128),                       # C: random
        ])
        neighbours = find_similar_patients(
            query, ref, ["A", "B", "C"], reference_ahi=[10.0, 40.0, 5.0], k=2,
            exclude_self=False,
        )
        assert neighbours[0].subject_id == "B"
        assert neighbours[0].similarity > neighbours[1].similarity

    def test_exclude_self_drops_near_identical(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=128)
        ref = np.stack([query, rng.normal(size=128), rng.normal(size=128)])
        neighbours = find_similar_patients(
            query, ref, ["self", "A", "B"], reference_ahi=[20.0, 15.0, 5.0], k=2,
            exclude_self=True,
        )
        assert "self" not in [n.subject_id for n in neighbours]

    def test_ahi_propagates_to_neighbour_match(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=128)
        ref = rng.normal(size=(3, 128))
        neighbours = find_similar_patients(
            query, ref, ["A", "B", "C"], reference_ahi=[10.0, 40.0, 5.0], k=3,
        )
        ahis = {n.subject_id: n.ahi for n in neighbours}
        assert ahis == {"A": 10.0, "B": 40.0, "C": 5.0}


class TestEstimateRiskFromNeighbours:
    def test_all_severe_gives_high_risk(self):
        neighbours = [NeighbourMatch("A", 0.9, 40.0),
                      NeighbourMatch("B", 0.8, 50.0)]
        assert estimate_risk_from_neighbours(neighbours) == pytest.approx(1.0)

    def test_all_normal_gives_low_risk(self):
        neighbours = [NeighbourMatch("A", 0.9, 2.0),
                      NeighbourMatch("B", 0.8, 4.0)]
        val = estimate_risk_from_neighbours(neighbours)
        assert 0.0 <= val <= 0.15

    def test_similarity_weighted_matters(self):
        """A high-AHI neighbour with low similarity should weigh less than a
        low-AHI neighbour with high similarity."""
        neighbours = [NeighbourMatch("A", 0.99, 5.0),   # normal, very similar
                      NeighbourMatch("B", 0.01, 60.0)]  # severe, barely similar
        val = estimate_risk_from_neighbours(neighbours)
        assert val < 0.4  # dominated by A

    def test_no_ahi_returns_neutral(self):
        neighbours = [NeighbourMatch("A", 0.9, None),
                      NeighbourMatch("B", 0.8, None)]
        assert estimate_risk_from_neighbours(neighbours) == 0.5

    def test_empty_returns_neutral(self):
        assert estimate_risk_from_neighbours([]) == 0.5


class TestBuildRiskPayload:
    def test_returns_valid_payload(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=128)
        ref = rng.normal(size=(4, 128))
        payload = build_risk_payload(
            query_subject_id="P-102",
            query_embedding=query,
            reference_embeddings=ref,
            reference_subject_ids=["A", "B", "C", "D"],
            reference_ahi=[5.0, 25.0, 45.0, 10.0],
            k=3,
        )
        assert payload.patient_id == "P-102"
        assert payload.condition == DEFAULT_CONDITION
        assert 0.0 <= payload.risk_score <= 1.0
        assert len(payload.neighbours) == 3


class TestRiskPayloadClinicalBridgeInput:
    def test_clamps_and_serialises(self):
        payload = RiskPayload("P-102", "Atrial Fibrillation", 1.5, [])
        d = payload.to_clinical_bridge_input()
        assert d == {"patient_id": "P-102", "condition": "Atrial Fibrillation",
                     "risk_score": 1.0}


@pytest.mark.skipif(not (CLINICAL_BRIDGE_DIR / "main.py").exists(),
                    reason="clinical-bridge repo not found alongside this project")
class TestClinicalBridgeIntegration:
    def test_translates_to_fhir_and_attaches_neighbours(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=128)
        ref = rng.normal(size=(3, 128))
        payload = build_risk_payload(
            query_subject_id="P-102",
            query_embedding=query,
            reference_embeddings=ref,
            reference_subject_ids=["A", "B", "C"],
            reference_ahi=[5.0, 25.0, 45.0],
            k=3,
        )
        fhir_resource = translate_to_fhir(payload)
        assert fhir_resource["resourceType"] == "RiskAssessment"
        # our extension must be attached
        extension_urls = [ext.get("url") for ext in fhir_resource.get("extension", [])]
        assert "http://sleepfm-interpretability/similar-patients" in extension_urls

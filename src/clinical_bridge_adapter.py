"""Bridge between SleepFM embeddings and the clinical-bridge FHIR project.

Turns a SleepFM patient embedding into a clinical-bridge-compatible payload:
    {patient_id: str, condition: str, risk_score: float 0-1}

The risk_score is a proxy: because we do not have a fine-tuned CoxPH head
locally, we estimate risk by k-nearest-neighbour lookup against a reference
cohort whose AHI (Apnea-Hypopnea Index) is known. The proxy is:

    risk_score = weighted mean of neighbours' normalised AHI severity

where the weight is the cosine similarity between the query embedding and
each neighbour, and severity is `min(AHI / 30, 1.0)` (AHI ≥ 30 = severe,
0 = normal). This is a *demonstration* of the interpretability -> clinical
handoff, not a validated risk model — see README caveats.

The `similar_patients` list is attached to the FHIR RiskAssessment via
`build_fhir_input`'s optional `metadata` field so the clinician can see the
neighbours the risk was derived from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# AUDIT REVIEW FIX — semantic mismatch: previously defaulted to "Atrial
# Fibrillation" (SNOMED 71908006) even though the risk we compute is a
# similarity-weighted AHI (a sleep-apnea metric). Downstream FHIR consumers
# would have filed our payloads under the wrong disease bucket. Default
# now matches the underlying metric; callers can still pick "Atrial
# Fibrillation" explicitly when driving a real AFib CoxPH head.
DEFAULT_CONDITION = "Obstructive Sleep Apnea"
SEVERE_AHI_THRESHOLD = 30.0                 # AHI≥30 = severe OSA (AASM)


@dataclass
class NeighbourMatch:
    subject_id: str
    similarity: float
    ahi: Optional[float]


@dataclass
class RiskPayload:
    patient_id: str
    condition: str
    risk_score: float
    neighbours: List[NeighbourMatch]

    def to_clinical_bridge_input(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "condition": self.condition,
            "risk_score": float(np.clip(self.risk_score, 0.0, 1.0)),
        }


def _cosine_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query (128,) and reference (N, 128)."""
    q = query / (np.linalg.norm(query) + 1e-8)
    r = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8)
    return r @ q


def find_similar_patients(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_subject_ids: List[str],
    reference_ahi: Optional[List[float]] = None,
    k: int = 5,
    exclude_self: bool = True,
) -> List[NeighbourMatch]:
    """Return top-k reference subjects most similar to `query_embedding`.

    reference_ahi is optional; when provided it is attached to each match
    for downstream risk calculation.
    """
    if reference_embeddings.shape[0] == 0:
        return []
    sims = _cosine_similarity(query_embedding, reference_embeddings)
    order = np.argsort(-sims)  # descending

    kept: List[NeighbourMatch] = []
    for idx in order:
        sid = reference_subject_ids[idx]
        sim = float(sims[idx])
        if exclude_self and sim > 0.9999:
            # exact match — either the same subject or a duplicate embedding
            continue
        ahi = None
        if reference_ahi is not None:
            ahi_val = reference_ahi[idx]
            ahi = None if ahi_val is None or (isinstance(ahi_val, float) and np.isnan(ahi_val)) else float(ahi_val)
        kept.append(NeighbourMatch(sid, sim, ahi))
        if len(kept) >= k:
            break
    return kept


def estimate_risk_from_neighbours(
    neighbours: List[NeighbourMatch],
    severe_threshold: float = SEVERE_AHI_THRESHOLD,
) -> float:
    """Similarity-weighted mean of neighbour AHI severity, mapped to [0, 1].

    severity(patient) = min(AHI / severe_threshold, 1.0)
    weight(patient)   = max(similarity, 0)
    risk_score        = sum(weight * severity) / sum(weight)

    Ignores neighbours whose AHI is unknown. Returns 0.5 when no neighbours
    carry AHI (unknown risk).
    """
    known = [n for n in neighbours if n.ahi is not None]
    if not known:
        return 0.5
    weights = np.array([max(n.similarity, 0.0) for n in known], dtype=float)
    if weights.sum() == 0:
        return 0.5
    severities = np.array([min(n.ahi / severe_threshold, 1.0) for n in known])
    return float((weights * severities).sum() / weights.sum())


def build_risk_payload(
    query_subject_id: str,
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_subject_ids: List[str],
    reference_ahi: Optional[List[float]] = None,
    condition: str = DEFAULT_CONDITION,
    k: int = 5,
) -> RiskPayload:
    neighbours = find_similar_patients(
        query_embedding, reference_embeddings, reference_subject_ids,
        reference_ahi=reference_ahi, k=k,
    )
    risk = estimate_risk_from_neighbours(neighbours)
    return RiskPayload(
        patient_id=query_subject_id, condition=condition,
        risk_score=risk, neighbours=neighbours,
    )


def _default_clinical_bridge_dir() -> Path:
    """Resolve clinical-bridge sibling repo location.

    Precedence: $CLINICAL_BRIDGE_DIR -> ../clinical-bridge-main (sibling of
    this repo) -> ../clinical-bridge -> ../../clinical-bridge-main. Returns
    the first candidate that exists, or the env-var value if unset (which
    then errors loudly if not present at import time).
    """
    import os
    env = os.environ.get("CLINICAL_BRIDGE_DIR", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve().parents[1]  # sleepfm_interpretability/
    for cand in [
        here.parent / "clinical-bridge-main",
        here.parent / "clinical-bridge",
        here.parent.parent / "clinical-bridge-main",
    ]:
        if cand.exists():
            return cand
    # Last-resort fallback — matches historical default so nothing breaks
    # for the original developer; other machines should set $CLINICAL_BRIDGE_DIR.
    return here.parent / "clinical-bridge-main"


def translate_to_fhir(
    payload: RiskPayload,
    clinical_bridge_dir: Optional[Path] = None,
    attach_neighbours: bool = True,
) -> Dict:
    """Call the clinical-bridge adapter and (optionally) attach neighbour
    references as a custom extension to the returned RiskAssessment.

    Both projects have top-level modules named `config`, `models` etc., so a
    naive `sys.path.insert` would let sleepfm_interpretability's `config`
    win when clinical-bridge tries to import its own. We snapshot the full
    module set before the import and drop *every* newly loaded module
    afterwards — a whitelist would silently miss any new top-level module
    the sibling project adds later (audit finding 3).
    """
    if clinical_bridge_dir is None:
        clinical_bridge_dir = _default_clinical_bridge_dir()
    import sys
    cb_str = str(clinical_bridge_dir)
    orig_path = list(sys.path)
    modules_before = set(sys.modules)
    conflicting_before = {
        name: sys.modules[name] for name in tuple(sys.modules)
        if name.split(".")[0] in {"config", "models", "main", "risk_engine",
                                  "fhir_builder", "utils"}
    }
    # Drop conflicting top-level modules so `from main import ...` inside
    # clinical-bridge triggers a fresh resolution from cb_str.
    for name in conflicting_before:
        sys.modules.pop(name, None)
    try:
        if cb_str in sys.path:
            sys.path.remove(cb_str)
        sys.path.insert(0, cb_str)
        from main import SleepFMToFHIRAdapter  # type: ignore
        adapter = SleepFMToFHIRAdapter()
        fhir_resource = adapter.translate(payload.to_clinical_bridge_input())
    finally:
        sys.path[:] = orig_path
        # Remove every module clinical-bridge loaded, regardless of name.
        added = set(sys.modules) - modules_before
        for name in added:
            sys.modules.pop(name, None)
        # Restore the sleepfm_interpretability originals we shadowed.
        for name, module in conflicting_before.items():
            sys.modules[name] = module

    if attach_neighbours and payload.neighbours:
        fhir_resource.setdefault("extension", []).append({
            "url": "http://sleepfm-interpretability/similar-patients",
            "extension": [
                {
                    "url": "neighbour",
                    "extension": [
                        {"url": "subject_id", "valueString": n.subject_id},
                        {"url": "cosine_similarity", "valueDecimal": n.similarity},
                        {"url": "ahi", "valueDecimal": n.ahi if n.ahi is not None else -1.0},
                    ],
                }
                for n in payload.neighbours
            ],
        })
    return fhir_resource

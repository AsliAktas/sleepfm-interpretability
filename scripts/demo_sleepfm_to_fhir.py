"""End-to-end demo: SleepFM embedding -> similar-patient risk -> FHIR RiskAssessment.

Runs a single-query demonstration of the interpretability pipeline plugged
into the clinical-bridge FHIR adapter:

    1. Load per-subject embeddings for a cohort (uses spherical_mean over
       5-min chunks; audit-hardened aggregation).
    2. Pick one subject as the "query"; the remaining subjects form the
       reference cohort.
    3. Cosine-similarity KNN -> top-5 similar patients.
    4. Similarity-weighted mean of neighbours' AHI severity -> proxy risk
       score in [0, 1].
    5. Hand off {patient_id, condition, risk_score} to clinical-bridge's
       SleepFMToFHIRAdapter -> validated FHIR R4 RiskAssessment (JSON).
    6. Attach neighbour references as a custom extension on the resource
       so the clinician sees what the risk was derived from.

    !!! This is a *method demonstration*, not a validated risk model.
        Risk score is a proxy: no fine-tuned CoxPH head is used here, and
        the current cohort is contaminated (~19/20 in SleepFM's pretraining
        split). See reports/phase8*/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_analysis import load_metadata
from clinical_bridge_adapter import build_risk_payload, translate_to_fhir
from real_embeddings import DEFAULT_EMBEDDING_DIR, load_subject_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", default="MULTI",
                        help="BAS/RESP/EKG/EMG/MULTI (default: MULTI, 4x128 concat)")
    parser.add_argument("--query-subject", default="REDACTED",
                        help="Subject ID to use as query (default: REDACTED, the one clean-cohort ID)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbours")
    parser.add_argument("--embedding-dir", type=Path, default=None,
                        help="Directory of *_embeddings.hdf5 files")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reports" / "phase9g_demo",
                        help="Output directory for FHIR JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("SleepFM Interpretability -> Clinical Bridge Demo")
    print("=" * 70)
    print(f"modality={args.modality}  query={args.query_subject}  k={args.k}\n")

    if args.modality == "MULTI":
        from real_embeddings import load_subject_embeddings_multimodal
        X, subject_ids, skipped = load_subject_embeddings_multimodal(
            embedding_dir=args.embedding_dir,
        )
    else:
        X, subject_ids = load_subject_embeddings(
            embedding_dir=args.embedding_dir, modality=args.modality,
        )

    if args.query_subject not in subject_ids:
        print(f"[error] query subject {args.query_subject!r} not in cohort. "
              f"Available: {subject_ids}", file=sys.stderr)
        return 2

    q_idx = subject_ids.index(args.query_subject)
    query_embedding = X[q_idx]
    ref_mask = np.ones(len(subject_ids), dtype=bool)
    ref_mask[q_idx] = False
    ref_X = X[ref_mask]
    ref_ids = [s for i, s in enumerate(subject_ids) if i != q_idx]

    print(f"[cohort] {len(subject_ids)} subjects, embedding dim={X.shape[1]}")
    print(f"[query]  subject {args.query_subject} vs {len(ref_ids)} reference subjects\n")

    meta = load_metadata(subject_ids)
    ahi_by_sid = dict(zip(meta["subject_id"], meta["ahi"]))
    ref_ahi = [ahi_by_sid.get(sid) for sid in ref_ids]
    query_ahi = ahi_by_sid.get(args.query_subject)

    payload = build_risk_payload(
        query_subject_id=args.query_subject,
        query_embedding=query_embedding,
        reference_embeddings=ref_X,
        reference_subject_ids=ref_ids,
        reference_ahi=ref_ahi,
        k=args.k,
    )

    print(f"[risk]   query true AHI  = {query_ahi}")
    print(f"[risk]   proxy risk score = {payload.risk_score:.3f}\n")
    print(f"[neighbours] top-{args.k} similar patients:")
    for n in payload.neighbours:
        print(f"   {n.subject_id}  sim={n.similarity:+.3f}  ahi={n.ahi}")

    fhir_resource = translate_to_fhir(payload)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"risk_assessment_{args.query_subject}_{args.modality}.json"
    out_path.write_text(json.dumps(fhir_resource, indent=2), encoding="utf-8")

    print(f"\n[fhir] FHIR RiskAssessment written to {out_path}")
    print(f"[fhir] Resource summary:")
    print(f"   resourceType : {fhir_resource.get('resourceType')}")
    print(f"   status       : {fhir_resource.get('status')}")
    print(f"   subject      : {fhir_resource.get('subject', {}).get('reference')}")
    preds = fhir_resource.get("prediction", [])
    if preds:
        pred = preds[0]
        print(f"   prediction[0]:")
        outcome = pred.get("outcome", {}).get("coding", [{}])[0]
        print(f"     outcome    : {outcome.get('display')} ({outcome.get('code')})")
        print(f"     qual risk  : {pred.get('qualitativeRisk', {}).get('coding', [{}])[0].get('display')}")
        print(f"     rationale  : {pred.get('rationale')}")
    exts = fhir_resource.get("extension", [])
    if exts:
        neighbours_ext = next(
            (e for e in exts if e.get("url", "").endswith("similar-patients")),
            None,
        )
        if neighbours_ext:
            print(f"   extension    : {len(neighbours_ext.get('extension', []))} similar-patient references")

    return 0


if __name__ == "__main__":
    sys.exit(main())

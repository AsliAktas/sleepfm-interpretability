# Phase 25 — Modality-transfer sanity check

**Purpose.** Phase 22 reported that the RESP modality × AHI and × ODI3
pass cross-modality FWER (Bonferroni p ≈ 0.034, 0.035), while BAS, EKG,
EMG, and MULTI show no FWER-significant association with any clinical
variable. This report asks the inverse question: is the BAS null a
**genuine negative control** (model preserved modality separation), or
an artefact (collapsed embeddings, failed clustering, underpowered test,
seed-unstable labels, or channel mislabel)?

Six layers were tested. All results are produced by `scripts/run_modality_transfer_sanity.py`
from the committed Phase 22 artefacts plus the local (gitignored) n=100 embeddings.

---

## A. Embedding collapse check (relative across modalities)

| modality   |   n_subjects |   dim |   per_dim_var_mean |   per_dim_var_std |   per_dim_var_cv |   pairwise_cos_mean |   pairwise_cos_std |   pairwise_cos_p05 |   pairwise_cos_p95 |   participation_ratio |   max_eig_ratio | verdict                                      |
|:-----------|-------------:|------:|-------------------:|------------------:|-----------------:|--------------------:|-------------------:|-------------------:|-------------------:|----------------------:|----------------:|:---------------------------------------------|
| BAS        |          100 |   128 |             0.0011 |            0.0004 |           0.3631 |              0.8624 |             0.0777 |             0.7027 |             0.9583 |               11.5876 |          0.1702 | HIGHER_EFFECTIVE_DIM (pr=11.6 vs median 6.5) |
| RESP       |          100 |   128 |             0.0003 |            0.0002 |           0.6504 |              0.9583 |             0.0307 |             0.8986 |             0.9889 |                3.6738 |          0.4853 | LOWER_EFFECTIVE_DIM (pr=3.7 vs median 6.5)   |
| EKG        |          100 |   128 |             0.0007 |            0.0005 |           0.7373 |              0.9116 |             0.0645 |             0.7846 |             0.9848 |                3.4194 |          0.5043 | LOWER_EFFECTIVE_DIM (pr=3.4 vs median 6.5)   |
| EMG        |          100 |   128 |             0.0014 |            0.0006 |           0.4055 |              0.8139 |             0.0778 |             0.6841 |             0.9339 |                9.3991 |          0.2183 | NEAR_MEDIAN                                  |

**Methodology note.** An earlier draft of this script used absolute
thresholds (`pairwise_cos > 0.9` → COLLAPSED). That is wrong for
contrastive-trained foundation models: SleepFM naturally produces
tightly packed embeddings on the unit sphere, and RESP (which DOES
produce a FWER-significant signal in Phase 22) sits at
pairwise_cos=0.958. The verdict here is now RELATIVE: a modality is
flagged only if it is dramatically more concentrated or lower-dim
than the median across the four modalities. For the BAS null to be
a degeneracy artefact, BAS would need to be MORE concentrated than
RESP — but it is in fact LESS concentrated.

## B. Cluster structure (from Phase 22 stability files)

| modality   | available   |   n_seeds |   n_clusters_mean |   n_clusters_std |   n_clusters_min |   n_clusters_max | verdict   |
|:-----------|:------------|----------:|------------------:|-----------------:|-----------------:|-----------------:|:----------|
| BAS        | True        |        10 |            29.200 |            2.040 |               25 |               32 | HEALTHY   |
| RESP       | True        |        10 |            26.200 |            2.750 |               21 |               29 | HEALTHY   |
| EKG        | True        |        10 |            30.300 |            2.492 |               25 |               34 | HEALTHY   |
| EMG        | True        |        10 |            26.300 |            2.052 |               22 |               29 | HEALTHY   |

## C. Broader statistical scan (BAS × all 6 clinical vars + KS vs uniform)

| modality   |   n_tests |   raw_p_mean |   raw_p_min |   raw_p_max |   raw_p_near_one_frac |   ks_vs_uniform_stat |   ks_vs_uniform_p | any_bonferroni_hit   | any_fdr_hit   | verdict                                                     |
|:-----------|----------:|-------------:|------------:|------------:|----------------------:|---------------------:|------------------:|:---------------------|:--------------|:------------------------------------------------------------|
| BAS        |         6 |       0.4256 |      0.0603 |      0.8796 |                0.1667 |               0.3961 |            0.2337 | False                | False         | HEALTHY_NULL: raw p's consistent with uniform (honest null) |
| EKG        |         6 |       0.1915 |      0.0235 |      0.8114 |                0.1667 |               0.6445 |            0.0059 | False                | False         | AMBIGUOUS: non-uniform but no FWER hit                      |
| EMG        |         6 |       0.0791 |      0.0187 |      0.1437 |                0.0000 |               0.8563 |            0.0000 | False                | False         | AMBIGUOUS: non-uniform but no FWER hit                      |
| MULTI      |         6 |       0.0359 |      0.0054 |      0.1190 |                0.0000 |               0.8810 |            0.0000 | False                | True          | AMBIGUOUS: non-uniform but no FWER hit                      |
| RESP       |         6 |       0.2177 |      0.0011 |      0.6505 |                0.0000 |               0.6565 |            0.0046 | True                 | True          | ACTIVE: FWER hit present (modality finds structure)         |

KS test asks whether the 6 raw p-values per modality are consistent with
`U(0,1)` (honest null). A p-value < 0.1 would suggest the raw-p's are
non-uniform (either enriched for low p's → real signal, or clumped high
→ degenerate modality). Uniform-null is the expected shape for a healthy
modality that genuinely finds nothing.

## D. Channel assignment audit

See `channel_audit.md`. BAS = EEG + EOG; RESP = thorax/abdomen/SpO2/nasal
pressure/airflow; EKG = ECG leads; EMG = chin/leg leads. No channel appears
in more than one group.

## E. Effect-gap analysis (Kruskal-Wallis, empirical)

| modality   | variable   |   raw_p |   alpha_fw |   margin_vs_alpha_fw |   fold_above_best_in_family | interpretation     |
|:-----------|:-----------|--------:|-----------:|---------------------:|----------------------------:|:-------------------|
| RESP       | ahi        | 0.00114 |    0.00167 |              0.68240 |                     1.00000 | DETECTS_AT_FWER    |
| MULTI      | ahi        | 0.01434 |    0.00167 |              8.60348 |                    12.60778 | NEAR_THRESHOLD     |
| EMG        | ahi        | 0.02484 |    0.00167 |             14.90542 |                    21.84280 | FAR_FROM_THRESHOLD |
| EKG        | ahi        | 0.03927 |    0.00167 |             23.56416 |                    34.53154 | FAR_FROM_THRESHOLD |
| BAS        | ahi        | 0.79827 |    0.00167 |            478.96404 |                   701.88667 | FAR_FROM_THRESHOLD |
| RESP       | odi3       | 0.00117 |    0.00167 |              0.70053 |                     1.00000 | DETECTS_AT_FWER    |
| MULTI      | odi3       | 0.00904 |    0.00167 |              5.42634 |                     7.74610 | NEAR_THRESHOLD     |
| EMG        | odi3       | 0.02082 |    0.00167 |             12.49386 |                    17.83497 | FAR_FROM_THRESHOLD |
| EKG        | odi3       | 0.04523 |    0.00167 |             27.13513 |                    38.73535 | FAR_FROM_THRESHOLD |
| BAS        | odi3       | 0.87959 |    0.00167 |            527.75224 |                   753.36558 | FAR_FROM_THRESHOLD |

**Methodology note.** Phase 22's actual test is Kruskal-Wallis on a
continuous clinical variable grouped by cluster assignment, not a
chi-square on a categorical×categorical contingency table — so an
analytical minimum-detectable-Cramér's-V formula does not apply. An
earlier draft of this script reported a theoretical MDE anyway and
flagged every modality (including the detected RESP) as
'UNDERPOWERED' — that output was wrong by construction and has
been removed. The replacement is empirical: rank modalities by raw
p against α = 0.05/30 ≈ 0.00167, and report `fold_above_best` to
quantify the gap between the detecting modality and the rest.

## F. Seed stability (ARI across HDBSCAN seeds, from Phase 22)

| modality   | available   |   n_seeds |   ari_mean |   ari_std |   ari_min |   ari_max | verdict                                                    |
|:-----------|:------------|----------:|-----------:|----------:|----------:|----------:|:-----------------------------------------------------------|
| BAS        | True        |        10 |      0.558 |     0.085 |     0.396 |     0.757 | MODERATE: partial stability; some seed-dependent variation |
| RESP       | True        |        10 |      0.559 |     0.116 |     0.378 |     0.858 | MODERATE: partial stability; some seed-dependent variation |
| EKG        | True        |        10 |      0.548 |     0.078 |     0.421 |     0.781 | MODERATE: partial stability; some seed-dependent variation |
| EMG        | True        |        10 |      0.551 |     0.113 |     0.368 |     0.826 | MODERATE: partial stability; some seed-dependent variation |

---

## Overall verdict

- **Collapse:** HIGHER_EFFECTIVE_DIM (pr=11.6 vs median 6.5)
- **Cluster structure:** HEALTHY
- **Broader scan:** HEALTHY_NULL: raw p's consistent with uniform (honest null)
- **Seed stability:** MODERATE: partial stability; some seed-dependent variation
- **Effect-gap (BAS × ahi):** FAR_FROM_THRESHOLD (p=0.7983, 702× best in family)
- **Effect-gap (BAS × odi3):** FAR_FROM_THRESHOLD (p=0.8796, 753× best in family)

If BAS shows (i) non-degenerate embedding (comparable or more spread than RESP), (ii) non-trivial cluster structure, (iii) uniform-null raw-p distribution across clinical variables, (iv) seed stability on par with other modalities, and (v) a raw-p for AHI/ODI3 that is many fold above the family-wise threshold, then the BAS null for these respiratory metrics is interpretable as modality-specific information retention by SleepFM. If any layer fails these checks, the interpretation is weakened accordingly.

## Change log

| Date       | Author      | Change               |
|------------|-------------|----------------------|
| 2026-10-08 | Aslı Aktaş  | İlk yazım (Phase 25) |

# SleepFM Interpretability

[![tests](https://github.com/AsliAktas/sleepfm-interpretability/actions/workflows/tests.yml/badge.svg)](https://github.com/AsliAktas/sleepfm-interpretability/actions/workflows/tests.yml)

SleepFM foundation model'inin embedding uzayının **klinik veriyle**
denetlenebilir analizi. Gerçek MESA PSG kayıtları → SleepFM embeddings →
istatistiksel rigor pass (chance baselines, cross-modality FWER, bootstrap
CI, subject-level permutation) → FHIR RiskAssessment dışa aktarımı.

---

## Durum (Eylül 2026)

Proje iki paradigma taşıyor. Aktif olan **gerçek MESA cohort** paradigmasıdır.

| Paradigma | Konum | Durum |
|---|---|---|
| **Real MESA cohort** (Phase 13c → Phase 22) | `src/real_embeddings.py`, `src/chunk_level_analysis.py`, `src/rigor_analysis.py`, `src/clinical_bridge_adapter.py` | **Aktif** |
| Mock disease demo (Nisan 2026, ilk fazlar) | `src/legacy/` (Phase 16s'te izole edildi) | Legacy — metodoloji öğrenme çerçevesi; bilimsel çıkarım için kullanılmaz. Bkz. [`src/legacy/README.md`](src/legacy/README.md) |

**Son bilimsel özet:** [`reports/phase22_defensibility_n100/README.md`](reports/phase22_defensibility_n100/README.md)
— n=100 clean cohort'ta cross-modality Bonferroni'yi geçen ilk pozitif
bulgular (RESP × AHI, RESP × ODI3; positive-control düzeyinde, bkz.
raporun "Circularity Kaveati" bölümü).

Önceki fazlar: [`reports/phase15_defensibility/`](reports/phase15_defensibility/README.md)
(n=20, "sağlam pozitif bulgu YOK" verdikti — Phase 22'de revize edildi).

Pretrain-independence kanıtı: [`reports/pretrain_independence/README.md`](reports/pretrain_independence/README.md)

---

## Bilimsel Çerçeve

**Amaç:** SleepFM'in ürettiği embedding'lerin ne kadarı gerçek fizyolojik
sinyal, ne kadarı memorization/biometric bias — bunu istatistiksel olarak
sınanabilir hale getirmek.

**Veri:** 20 MESA hastası, SleepFM'in resmi `test` split'inden
(pretrain'de 0 overlap — bkz. [pretrain_independence](reports/pretrain_independence/README.md)).
NSRR üzerinden indirilir; ham EDF/XML repoya commit edilmez.

**Yöntem:**
- 4 modalite × 128-dim embedding (BAS/RESP/EKG/EMG) + concatenated MULTI (512-dim)
- Yüksek-boyutlu UMAP → HDBSCAN kümeleme (silhouette original space'te ölçülür)
- Kruskal-Wallis + Bonferroni + Benjamini-Hochberg FDR (cross-modality family)
- Subject-level Westfall-Young step-down permutation (inter-değişken korelasyonu korur)
- Phipson-Smyth bounded p-values
- Label-shuffle chance baseline (empirical null)
- 10-seed ARI stability + 2000-iter bootstrap CI

**Ölçek sınırı:** n=20 çoğu klinik iddia için yetersiz. Phase 15
"sağlam pozitif bulgu YOK" verdikti — n=100 clean cohort planlı.

---

## Kurulum

Python 3.10, pinned dependencies:

```bash
git clone https://github.com/AsliAktas/sleepfm-interpretability.git
cd sleepfm-interpretability
pip install -r requirements.txt
```

SleepFM checkpoint (ayrı repo, `sleepFMoriginal/sleepfm-clinical/`) ve
NSRR data access (DUA gerekli) ayrıca kurulmalı — bkz.
[`NSRR_DOWNLOAD_GUIDE.md`](NSRR_DOWNLOAD_GUIDE.md).

---

## Proje Yapısı

```
sleepfm-interpretability/
├── src/                              # Aktif analiz modülleri
│   ├── real_embeddings.py            # HDF5 → spherical mean → subject-level table
│   ├── chunk_level_analysis.py       # Chunk-level HDBSCAN + purity
│   ├── rigor_analysis.py             # İstatistiksel rigor katmanı
│   ├── clinical_analysis.py          # UMAP+HDBSCAN + Kruskal-Wallis
│   ├── clinical_bridge_adapter.py    # KNN-proxy → FHIR RiskAssessment
│   ├── sleep_phases.py               # MESA XML parse → sleep-stage aware chunks
│   ├── paths.py, utils.py            # Ortak yardımcılar
│   └── legacy/                       # (legacy) Nisan 2026 mock evren — bkz. src/legacy/README.md
├── scripts/
│   ├── download_nsrr_mesa.py         # NSRR authenticated download (token-file, redaction)
│   ├── demo_sleepfm_to_fhir.py       # 3-hasta FHIR demo (method demo — klinik değil)
│   ├── verify_cohort_pretrain_independence.py  # Cohort × SleepFM splits cross-check
│   ├── run_phase15_analysis.py       # Phase 15 defensibility headline üretici
│   ├── run_bootstrap_stability.py    # Subject-dropout robustness
│   └── regen_reproduce_tables.py     # REPRODUCE.md tablolarını CSV'den üretir
├── tests/
│   ├── (aktif suite — 12 dosya)
│   └── legacy/                       # (legacy) mock evren testleri — bkz. tests/legacy/README.md
├── reports/
│   ├── phase15_defensibility/        # En güncel bilimsel özet
│   ├── phase13c_rigor_clean/         # Clean cohort ilk analizi (Phase 15'te revize)
│   ├── pretrain_independence/        # Pretrain-independence kanıtı
│   └── phase9g_demo/                 # FHIR RiskAssessment demo JSONs
├── data/
│   ├── clean_cohort_run/embeddings/  # 20 hasta × SleepFM embeddings (HDF5)
│   └── smoke_run/                    # 3-hasta smoke pipeline output
├── requirements.txt
└── README.md
```

---

## Testler

```bash
python -m pytest tests/ -v
```

12 dosya, **194 test** (parametrize sonrası; grep ile `def test_`
sayarsanız 190 görürsünüz — 4 parametrized case fazlası). En son
Phase 20 koşumu: **191 passed + 3 skipped** (skipped = real-MESA-XML
integration testleri, DAUA gereği synthetic path'e çevrildiğinden CI
+ fresh clone'da graceful skip). rigor katmanı için 36 test, NSRR
download script'i için 25 test (magic byte, redaction, atomic rename,
token precedence).

---

## Not — Klinik Kullanım

Repodaki hiçbir çıktı klinik karar için kullanılamaz. FHIR RiskAssessment
JSON'ları **method demo**dur: KNN-proxy risk skoru, foundation model
çıktısı değil. n=20 kohort istatistiksel güç sağlamaz. Ayrıntı için
[`reports/phase15_defensibility/README.md`](reports/phase15_defensibility/README.md)
§6 (RESP-only demo) ve [`reports/phase9g_demo/README.md`](reports/phase9g_demo/README.md).

---

## Referanslar

- **SleepFM**: [A multimodal sleep foundation model for disease prediction](https://doi.org/10.1038/s41591-025-04133-4) — *Nature Medicine, 2026*
- **Upstream repo**: [zou-group/sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical)
- **MESA / NSRR**: [sleepdata.org/datasets/mesa](https://sleepdata.org/datasets/mesa)

---

## Acknowledgment

This work uses data obtained through the National Sleep Research Resource.

> NSRR R24 HL114473: NHLBI National Sleep Research Resource.

Required under Section 16 of the NSRR Data Access and Use Agreement, and to
be reproduced in any publication or presentation arising from this work. The
obligation survives expiry of the agreement (Section 17).

Data access is governed by a signed DAUA (3 April 2026). No NSRR data is
included in this repository: raw recordings, derived embeddings and
subject-level identifiers are excluded from version control, and clinical
values in the demonstration outputs are binned rather than exact. The NSRR
confirmed on 1 October 2026 that this analysis falls within the approved
Specific Purpose and that sharing code is permitted where no data is shared.
Extending the work or using MESA data for additional projects requires a new
proposal.

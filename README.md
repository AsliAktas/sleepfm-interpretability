# SleepFM Interpretability

SleepFM foundation model'inin embedding uzayının **klinik veriyle**
denetlenebilir analizi. Gerçek MESA PSG kayıtları → SleepFM embeddings →
istatistiksel rigor pass (chance baselines, cross-modality FWER, bootstrap
CI, subject-level permutation) → FHIR RiskAssessment dışa aktarımı.

---

## Durum (Eylül 2026)

Proje iki paradigma taşıyor. Aktif olan **gerçek MESA cohort** paradigmasıdır.

| Paradigma | Konum | Durum |
|---|---|---|
| **Real MESA cohort** (Phase 13c → Phase 16) | `src/real_embeddings.py`, `src/chunk_level_analysis.py`, `src/rigor_analysis.py`, `src/clinical_bridge_adapter.py` | **Aktif** |
| Mock disease demo (Nisan 2026, ilk fazlar) | `src/mock_data.py`, `src/similarity_engine.py`, `src/pipeline.py`, `run_ablation_demo.py` | Legacy — metodoloji öğrenme çerçevesi; bilimsel çıkarım için kullanılmaz |

Son bilimsel özet: [`reports/phase15_defensibility/README.md`](reports/phase15_defensibility/README.md)
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
git clone https://github.com/AsliAktas/sleepfm_interpretability.git
cd sleepfm_interpretability
pip install -r requirements.txt
```

SleepFM checkpoint (ayrı repo, `sleepFMoriginal/sleepfm-clinical/`) ve
NSRR data access (DUA gerekli) ayrıca kurulmalı — bkz.
[`NSRR_DOWNLOAD_GUIDE.md`](NSRR_DOWNLOAD_GUIDE.md).

---

## Proje Yapısı

```
sleepfm_interpretability/
├── src/                              # Analiz modülleri
│   ├── real_embeddings.py            # (aktif) HDF5 → spherical mean → subject-level table
│   ├── chunk_level_analysis.py       # (aktif) Chunk-level HDBSCAN + purity
│   ├── rigor_analysis.py             # (aktif) İstatistiksel rigor katmanı
│   ├── clinical_analysis.py          # (aktif) UMAP+HDBSCAN + Kruskal-Wallis
│   ├── clinical_bridge_adapter.py    # (aktif) KNN-proxy → FHIR RiskAssessment
│   ├── sleep_phases.py               # (aktif) MESA XML parse → sleep-stage aware chunks
│   ├── mock_data.py                  # (legacy) 500 sentetik hasta, 12 hastalık
│   ├── similarity_engine.py          # (legacy) KNN Top-5 disease recommendation
│   └── pipeline.py                   # (legacy) mock end-to-end
├── scripts/
│   ├── download_nsrr_mesa.py         # NSRR authenticated download (token-file, redaction)
│   ├── demo_sleepfm_to_fhir.py       # 3-hasta FHIR demo (method demo — klinik değil)
│   └── verify_cohort_pretrain_independence.py  # Cohort × SleepFM splits cross-check
├── tests/                            # ~189 test tanımı, 12 dosya
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

12 dosya, **193 test** (`pytest.parametrize` sonrası tam sayı; grep ile
`def test_` sayarsanız 189 görürsünüz — 4 parametrized case fazlası). En
son çalıştırma: **193/193 passed** (Phase 16, `sleepfm` conda env, 125 sn).
rigor katmanı için 36 test, NSRR download script'i için 21 test (magic
byte, redaction, atomic rename).

---

## Not — Klinik Kullanım

Repodaki hiçbir çıktı klinik karar için kullanılamaz. FHIR RiskAssessment
JSON'ları **method demo**dur: KNN-proxy risk skoru, foundation model
çıktısı değil. n=20 kohort istatistiksel güç sağlamaz. Ayrıntı için
[`SONUC_ANALIZI.md`](SONUC_ANALIZI.md).

---

## Referanslar

- **SleepFM**: [A multimodal sleep foundation model for disease prediction](https://doi.org/10.1038/s41591-025-04133-4) — *Nature Medicine, 2026*
- **Upstream repo**: [zou-group/sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical)
- **MESA / NSRR**: [sleepdata.org/datasets/mesa](https://sleepdata.org/datasets/mesa)

# Phase 9 — Consolidated Report

## Nihai Durum

**Kod tabanı**
- 5 yeni `src/` modülü, 5 yeni `tests/` dosyası, 172/172 test geçiyor.
- 1 yeni script (`scripts/download_nsrr_mesa.py`), 1 demo script (`scripts/demo_sleepfm_to_fhir.py`).
- İki bağımsız Python projesi (`sleepfm_interpretability` ↔ `clinical-bridge-main`) end-to-end bir pipeline oluşturur:

  **PSG signal → SleepFM embedding → similar-patient KNN → proxy risk → FHIR R4 RiskAssessment**

**Bilimsel bulgu durumu**
- Tüm sayısal sonuçlar kontamine kohort (17/20 SleepFM pretrain) üzerinde üretildi ve method demonstration olarak korunuyor.
- Temiz cohort için tam mekanizma hazır (NSRR download + pipeline swap-in).

## Phase 9'da Tamamlananlar

### 9a — NSRR Python indirme scripti
`scripts/download_nsrr_mesa.py` — 20 stratified MESA hastasını NSRR'dan authenticated indirir. Resume + retry + parallel + verify + magic-byte + atomic-rename dahil.

**Audit sonrası fixler (8 finding):**
- Token asla argv'de değil (getpass / env / file), asla log'a sızmıyor (`_redact`).
- URL'de `?auth_token=X` gömülü + `allow_redirects=False` (redirect'te auth kaybı önleme).
- `.part` + atomic rename + `.lock` (concurrent-write koruma).
- EDF/XML magic-byte doğrulama (HTML error page kabul edilmiyor).
- DUA-specific verify (token valid ama MESA erişimi yoksa erken abort).
- Phipson-Smyth benzeri lower-bound-inspired file-size sanity.

### 9c — Sleep-phase preprocessing
`src/sleep_phases.py` — MESA XML annotation'larını parse edip 5-dk embedding chunk'larına dominant sleep stage atar.

**Audit sonrası fixler (8 finding):**
- `coverage_fraction` alanı — kısmi kayıt chunk'larının bias'ı elimine.
- `recording_offset_sec` parametresi — SleepFM preprocessing calibration offset'i için.
- `parse_stages` corrupt XML'de crash yerine `[]` döner.
- `load_chunk_labels_for_cohort` tuple `(labels, skipped)` döner — silent skip yok.
- N1-friendly default threshold (0.5 → 0.4) — N1 chunk'ları kurtar.
- Wake/Unsure/Unknown distinction docstring'te.

### 9e — Chunk-level analiz
`src/chunk_level_analysis.py` — 20 hasta × ~120 chunk = ~2464 nokta üzerinde HDBSCAN + cluster×stage confusion + subject purity.

**Audit sonrası fixler (7 finding + null baseline + sensitivity):**
- `_align_labels_to_chunks` subject-ordering assert (silent mis-alignment koruma).
- Hardcoded MESA path → env fallback.
- `stage_ari_null` — Westfall-Young tarzı permutation null; observed ARI karşılaştırma için baseline.
- `sensitivity_sweep` — HDBSCAN `min_cluster_size` sweep.
- Docstring'deki "typically 0.05-0.25" iddiası kaldırıldı (kaynaksız).

**Ana bulgu:**
| Modalite | ARI(stage) | Subject purity | Yorum |
|---|---:|---:|---|
| BAS | 0.0003 | **0.74** | EEG hasta imzasını yakalıyor, uyku fazını değil |
| RESP | 0.034 | 0.37 | Daha global |
| EKG | 0.024 | 0.55 | Orta |
| EMG | 0.038 | 0.48 | Orta |

Subagent yorumu: *"BAS embedding kontrastif pretraining objektifinin subject-discrimination'ı stage-discrimination'a örtük olarak tercih ettiğinin işareti. Temiz cohort'ta subject purity ~0.05'e düşerse etki tamamen memorization; ≥0.3 kalırsa gerçek bir EEG hasta-varyansı sinyali."*

### 9g — Clinical-bridge entegrasyonu MVP
`src/clinical_bridge_adapter.py` + `scripts/demo_sleepfm_to_fhir.py` — SleepFM embedding'inden KNN-based proxy risk üretip clinical-bridge'in `SleepFMToFHIRAdapter`'ına verir.

**Demo çıktısı (query pseudonymized, MULTI 512-dim):**
```
[cohort] 20 subjects, embedding dim=512
[query]  subject SUBJ_X (AHI bin: <5) vs 19 reference subjects
[risk]   proxy risk score = 0.416
[neighbours] top-5:
   SUBJ_*  sim=~0.95  ahi_bin=<5
   SUBJ_*  sim=~0.95  ahi_bin=5-15
   SUBJ_*  sim=~0.95  ahi_bin=5-15
   SUBJ_*  sim=~0.95  ahi_bin=15-30
   SUBJ_*  sim=~0.95  ahi_bin=5-15
[fhir]   RiskAssessment/Patient/SUBJ_X, Low likelihood, 5 similar-patient references
```

FHIR R4 çıktısı pseudonymized formda `reports/phase9g_demo/` altında.
Subject ID'ler ve tam AHI değerleri NSRR DAUA §5 gereği belgelenmedi.

**Önemli:** Risk score bir proxy (KNN-weighted AHI severity); gerçek CoxPH head lokalde yok. Method demonstration.

**Ek olarak:** İki projenin `config.py`, `models.py`, `main.py` gibi top-level modül isimleri çakışıyor. `translate_to_fhir` `sys.modules` + `sys.path` snapshot/restore ile isolated import yapıyor.

## Test Suite Özeti

| Test dosyası | Test sayısı | Kapsam |
|---|---:|---|
| test_embedding_generation.py | 20 | mock embedding üretimi |
| test_similarity_engine.py | 5 | KNN Top-5 label alignment |
| test_pipeline.py | 5 | pipeline orchestration |
| test_ablation.py | 12 | modality ablation |
| test_real_embeddings.py | 21 | SleepFM HDF5 loader + spherical mean |
| test_clinical_analysis.py | 11 | UMAP+HDBSCAN + stats |
| test_rigor_analysis.py | 27 | BH-FDR, Westfall-Young, ARI |
| test_download_nsrr_mesa.py | 25 | NSRR script (magic bytes, redaction) |
| test_sleep_phases.py | 25 | XML parse + chunk labelling |
| test_chunk_level_analysis.py | 12 | subject-ordering + null baseline |
| test_clinical_bridge_adapter.py | 13 | KNN + risk + FHIR integration |
| **Toplam** | **172** | 0 fail, 0 regression |

## Bilimsel Katkı Şu Anda

**Yayınlanabilir bulgu yok.** Kontamine kohort tüm sayısal sonuçları invalidate ediyor. Ama:
- **Yayınlanabilir yöntem:** rigor pass (Bonferroni + Westfall-Young + multi-seed stability + hyperparam sensitivity) + null baseline.
- **Yayınlanabilir mimari:** SleepFM embedding → similar-patient KNN → FHIR RiskAssessment köprüsü — SleepFM makalesinin (Nature Medicine 2025) klinik hattaki eksiğini adresliyor.
- **Yayınlanabilir hipotez:** BAS embedding subject-discrimination bias'ı; temiz cohort ile empirik olarak test edilecek.

## Sıradaki Adımlar (Temiz Cohort Sonrası)

1. NSRR'dan 20 hasta indir (`python scripts/download_nsrr_mesa.py`).
2. `preprocess_and_embed_smoke.py`'ın `MESA_DIR` ve `OUT_ROOT`'unu yeni dizine bakacak şekilde güncelle.
3. Chunk analizi + rigor pass + demo scripti tekrar çalıştır.
4. **Kritik karşılaştırma:** temiz cohort'ta
   - BAS subject purity 0.05 civarı → memorization artefaktı
   - BAS subject purity ≥0.3 → EEG hasta-varyansı gerçek sinyal
   - MULTI-sex Bonferroni geçerse → gerçek klinik-demografik bulgu
5. Sonuçları Phase 8e/9e ile diff'le, karşılaştırmalı rapor yaz.

## Dosya Envanteri

**Yeni `src/`:**
- `real_embeddings.py` — SleepFM HDF5 loader (spherical_mean, mock, chunk, multimodal)
- `clinical_analysis.py` — UMAP+HDBSCAN + stats + metadata join
- `rigor_analysis.py` — Bonferroni/FDR + Westfall-Young permutation + stability
- `sleep_phases.py` — XML parse + chunk labelling
- `chunk_level_analysis.py` — chunk clustering + stage confusion + null
- `clinical_bridge_adapter.py` — SleepFM → FHIR köprüsü

**Yeni `scripts/`:**
- `download_nsrr_mesa.py` — NSRR authenticated download
- `demo_sleepfm_to_fhir.py` — end-to-end demo

**Yeni `tests/`:**
- 6 test dosyası, 172/172 geçiyor

**Yeni `reports/`:**
- `phase7/` — ilk cluster+test (contaminated banner)
- `phase8c_rigor_on_contaminated/` — rigor pass v1
- `phase8e_rigor_v2_on_contaminated/` — rigor v2 after fixes
- `phase9e_chunk_level_v2/` — chunk × sleep-stage analiz
- `phase9g_demo/risk_assessment_SUBJ_*_MULTI.json` — FHIR demo çıktısı (pseudonymized)
- `phase9h_final/README.md` — bu rapor

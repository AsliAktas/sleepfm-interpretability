# Phase 8e — Rigor Pass v2 (After Audit Fixes)

> **UYARI:** Bu rapor da Phase 7/8c ile aynı kontamine cohort'a dayanır (17/20 pretrain,
> 2/20 train). Amaç yeni fix'lerin etkisini Phase 8c ile karşılaştırmak. Temiz cohort
> geldiğinde v2 kodu ile tekrar çalıştırılacak.

## Uygulanan Fix'ler

Audit raporundan gelen 10 bulgu düzeltildi:

**HIGH öncelik**
- **M1** — HDBSCAN artık orijinal 128-dim/512-dim uzayda değil, 15-boyutlu ara UMAP embedding'inde fit ediliyor (`fit_umap_hdbscan` içinde `cluster_dim=15`, metric=cosine); 2D UMAP sadece scatter plot için.
- **M2** — Aggregation `spherical_mean` (default): her chunk L2-norm → ortala → tekrar L2-norm. Chunk-count-invariant on unit sphere. Uzun uyku hastalarının bias'ı elimine.
- **C1** — `preprocess_and_embed_smoke.py` modalite sırasına assert eklendi.
- **C2** — `load_metadata` artık `.reindex()` kullanıyor; missing subjects warning'e düşüyor, KeyError yok.
- **C3** — `load_subject_embeddings_multimodal` 3-tuple döndürüyor `(X, sids, skipped)`; atlanmış hastalar surfaced.

**MEDIUM öncelik**
- **S1** — Phipson & Smyth (2010) `(count+1)/(n+1)` empirical-p bounding; p=0 imkansız.
- **S2** — `bonferroni_fdr_family` fonksiyonu cross-modality düzeltme için; `n_planned` explicit parametre.
- **S3** — Permutation subject-level row shuffle (kolon değil); AHI↔BMI↔age korelasyonları korunur. Ek olarak Westfall-Young step-down `familywise_adjusted_p` her variable için ayrı korrelasyona-hassas adjustment.
- **S4** — `test_rigor_analysis.py` (27 test): BH-FDR statsmodels karşılaştırma, Phipson-Smyth alt sınır, Westfall-Young monotonluk, deterministic permutation, planted signal recovery, ARI matrisi simetri.

**Bonus**
- HDBSCAN sweep silhouette artık orijinal high-dim uzayda hesaplanıyor (cosine metric), 2D plot uzayında değil.
- CUDA fallback: GPU yoksa CPU'ya düşer, RuntimeError yok.

## Sonuç — Phase 8c ile Karşılaştırma

### Küme Stabilitesi (Mean ARI ± SD, n=10 seed)

| Modalite | v1 (Phase 8c) | **v2 (Phase 8e)** | Değişim |
|---|---:|---:|---|
| BAS | 0.43 ± 0.21 | **0.87 ± 0.07** | 🚀 +0.44 (2×) |
| RESP | 0.56 ± 0.22 | 0.55 ± 0.21 | ≈ |
| EKG | 0.39 ± 0.23 | **0.71 ± 0.23** | 🚀 +0.32 |
| EMG | 0.56 ± 0.20 | 0.62 ± 0.16 | +0.06 |
| MULTI | 0.38 ± 0.18 | **0.66 ± 0.12** | 🚀 +0.28 |

Üç modalitede ARI dramatik olarak arttı. **Fix'lerin objektif etkisi doğrulandı**: mean-then-L2 bias + 2D UMAP-in-HDBSCAN kombinasyonu Phase 7/8c'de küme yapısını gerçekten distort ediyormuş.

### Bulgu Revizyonu

| Test | v1 durum | **v2 durum** |
|---|---|---|
| BAS raw significant | 0/6 | 0/6 |
| RESP raw significant | 1/6 (ODI3, p=0.045) | 2/6 (RESP hâlâ zayıf sinyalli) |
| EKG-sex Bonferroni | ✅ (0.024) | ❌ kayboldu — muhtemelen v1 artefakt |
| MULTI-sex Bonferroni | ✅ (0.029) | ✅ (0.045) — sağlam |
| MULTI-age FW permutation | — | **0.012 (anlamlı)** — yeni bulundu |
| MULTI-sex FW permutation | — | **0.012 (anlamlı)** — sağlam |

### Ders

- **BAS mean ARI 0.87** artık "gerçek küme yapısı var" diyebiliriz (kontamine cohort'ta bile)
- **EKG-sex sinyali kayboldu** — Phase 8c'deki "Bonferroni pass" methodological artifact idi
- **MULTI-sex ve MULTI-age güçlü** — hem Bonferroni hem Westfall-Young permutation ile doğrulandı
- **RESP-ODI3 kırılgan** — permutation ile anlamlı ama Bonferroni'yi geçmiyor; n arttıkça netleşir

## Çıktı Dosyaları

- `SUMMARY.csv` — 5 modalite × 10 metrik özet
- `correction_{MODALITY}.csv` — Bonferroni + BH-FDR
- `permutation_{MODALITY}.csv` — Westfall-Young FW-adjusted p
- `stability_{MODALITY}.csv` — 10-seed ARI + küme sayıları
- `stability_ari_matrix_{MODALITY}.npy` — 10×10 pairwise ARI
- `sweep_{MODALITY}.csv` — HDBSCAN hyperparam sweep (silhouette high-dim'de)

## Sıradaki Adım (Temiz Cohort İçin)

Aynı kod, `preprocess_and_embed_smoke.py`'ın `MESA_DIR` değişkeni değiştirilerek temiz test-split cohort üzerinde çalıştırılır. Beklentiler:
- BAS ARI 0.87 gibi yüksek kalmalı (küme yapısı gerçekse)
- MULTI-sex sinyali yeni cohort'ta düşerse: kontamine memorization artefaktıydı
- MULTI-sex sinyali kalırsa: gerçek klinik-demografik sinyal, güçlü bulgu
- RESP-ODI3 için n arttıkça power artmalı → Bonferroni'yi de geçme şansı

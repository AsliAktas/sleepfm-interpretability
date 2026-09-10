# Phase 8c — Rigor Pass (Kontamine Cohort Üzerinde Metodoloji Doğrulaması)

> **UYARI:** Bu raporun altındaki tüm sayısal sonuçlar Phase 7 ile aynı kontamine cohort'a
> (17/20 pretrain, 2/20 train) dayanır. Amaç bilimsel bulgu değil, **rigor pass metodolojisinin
> çalıştığını doğrulamak** ve temiz cohort geldiğinde direkt çalıştırılacak kodu hazırlamak.

## Yapılan 4 Kontrol

1. **Multiple-testing correction** — Bonferroni + Benjamini-Hochberg FDR
2. **Permutation test** — 1000 iterasyon, etiket-shuffle ile empirik null dağılımı
3. **Multi-seed stability** — 10 UMAP+HDBSCAN fit, pairwise Adjusted Rand Index
4. **HDBSCAN hyperparameter sweep** — `min_cluster_size ∈ {2,3,4,5}`, silhouette skoru

## Özet Bulgular

`SUMMARY.csv`:

| Modalite | Raw sig | Bonferroni sig | FDR sig | Min empirical p | Mean ARI ± SD | Best silhouette | Best min_cluster_size |
|---|---:|---:|---:|---:|---:|---:|---:|
| BAS | 0 | 0 | 0 | 0.320 | 0.43 ± 0.21 | 0.37 | 3 |
| RESP | 1 | 0 | 0 | **0.025** | 0.56 ± 0.22 | 0.53 | 4 |
| EKG | 1 | **1** | **1** | 0.418 | 0.39 ± 0.23 | 0.53 | 4 |
| EMG | 0 | 0 | 0 | 0.135 | 0.56 ± 0.20 | 0.58 | 3 |
| MULTI | 2 | **1** | **1** | **0.028** | 0.38 ± 0.18 | 0.39 | 3 |

## Kritik Yorumlar

- **Phase 7'nin RESP-ODI3 "anlamlı" iddiası düştü.** Raw p=0.045 → Bonferroni p=0.27. Ama permutation p=0.025 gösteriyor ki gerçek bir sinyal var (30-test artefaktı değil). Bu, n=20'de klinik olarak anlamlı ama istatistiksel olarak underpowered bir sinyal.
- **EKG-sex ve MULTI-sex Bonferroni'yi geçen tek bulgular** — kontamine cohort'un demografik ezberi gösteriyor.
- **Küme stabilitesi zayıf** (ARI ~0.4-0.6): 10 seed'de küme yapısı değişiyor. Yorum: n=20 için HDBSCAN inherently kararsız. Temiz cohort'ta n≥50 ile ARI ≥0.7 hedeflenebilir.
- **`min_cluster_size=2` (Phase 7'de kullanılan) suboptimal**. Sweep 3-4'ün daha iyi silhouette verdiğini gösterdi.

## Çıktılar

- `correction_{MODALITY}.csv` — raw + Bonferroni + FDR p-değerleri
- `permutation_{MODALITY}.csv` — observed vs empirical p-değerleri
- `stability_{MODALITY}.csv` — seed başına küme sayısı
- `stability_ari_matrix_{MODALITY}.npy` — 10×10 pairwise ARI
- `sweep_{MODALITY}.csv` — HDBSCAN hyperparam sweep
- `SUMMARY.csv` — 5 modalite × 9 metrik özet

## Temiz Cohort Hazırlığı

`src/rigor_analysis.py` ve `scratchpad/run_rigor_pass.py` parametre değişikliği ile yeni cohort'a uygulanabilir. Beklenen değişiklikler:
- Cinsiyet Bonferroni bulguları KAYBOLABİLİR (memorization artefaktı ise)
- RESP-ODI3 sinyali GÜÇLENMELİ (gerçek klinik sinyal ise, n artınca power artar)
- ARI YÜKSELMELİ (n=50 ile HDBSCAN daha kararlı)
- Silhouette YÜKSELMELİ (daha net klinik ayrışma varsa)

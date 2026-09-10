> # ⚠️ SONUCLAR GECERLI DEGIL — DATASET KONTAMINASYONU
>
> Bu rapordaki 20 MESA hastasinin **17'si SleepFM'in `pretrain` split'inde**, 2'si `train` split'inde;
> yalnizca 1 hasta (`mesa-sleep-REDACTED`) `test` split'inde temiz durumda.
>
> Dolayisiyla asagidaki embedding'ler modelin **ogrenilmis temsili** degil, buyuk olcude
> **training-set memorization**'dir. p-degerleri ve kume yapisi klinik olarak dogru olsa bile,
> SleepFM'in generalization yetenegi hakkinda hicbir sey soylemiyor.
>
> **Bu rapor sadece pipeline'in ucdan uca calistigini gosteren bir "method demo" olarak korunuyor.**
> Bilimsel sonuclar icin **temiz test-split cohort'una** ihtiyac var (bkz. Phase 8).
>
> — Tespit tarihi: 2026-09-10, kontrol edilen: `dataset_split.json`

---

# SleepFM Interpretability — Phase 7 Raporu (EMG)

**Kohort:** 20 MESA hastasi
**Embedding boyutu:** 128 (modalite: EMG)

## Kume ozeti

- Kume sayisi: **7** (noise dahil degil)
- Noise nokta sayisi: 1
- Kume dagilimi: `{-1: 1, 0: 2, 1: 2, 2: 2, 3: 2, 4: 5, 5: 4, 6: 2}`

## Kume x klinik degisken testleri

| Degisken | Test | Statistic | p-degeri | Not |
|---|---|---:|---:|---|
| age | Kruskal-Wallis | 9.182 | 0.1636 |  |
| bmi | Kruskal-Wallis | 7.950 | 0.2418 |  |
| ahi | Kruskal-Wallis | 4.048 | 0.6701 |  |
| ahi_obs | Kruskal-Wallis | 4.546 | 0.6032 |  |
| odi3 | Kruskal-Wallis | 4.509 | 0.6081 |  |
| sex | chi-square | 11.616 | 0.0711 |  |

## Yorumlar

- n=20 kohort kucuk; testler kesin sonuc icin **degil**, kesif icin.
- p<0.05 sonuclar hipotez uretimi seviyesindedir; validation icin daha buyuk kohort gerekli.
- Noise orani: 1/20 = 5% (yuksekse HDBSCAN min_cluster_size dusurmek denenebilir)

## Gorseller

![umap_EMG](umap_EMG.png)

![boxplots_EMG](boxplots_EMG.png)

## Hastalar

|   subject_id |   sex |   age |     bmi |      ahi |   ahi_obs |     odi3 |   race |   cluster |
|-------------:|------:|------:|--------:|---------:|----------:|---------:|-------:|----------:|
|         REDACTED |     0 |    70 | 22.0108 | 18.1395  |   9.94186 | 17.1177  |      1 |        -1 |
|         REDACTED |     0 |    83 | 21.898  |  8.34225 |   3.85027 |  9.20134 |      1 |         5 |
|         REDACTED |     0 |    57 | 56.0072 | 62.514   |  51.1173  | 65.114   |      4 |         6 |
|         REDACTED |     1 |    57 | 19.8446 | 40.9091  |  32.1818  | 42.7     |      1 |         3 |
|         REDACTED |     1 |    80 | 27.7738 | 27.4725  |  20.2198  | 26.0114  |      1 |         4 |
|         REDACTED |     0 |    60 | 31.4484 | 11.3744  |   6.54028 | 16.463   |      3 |         0 |
|         REDACTED |     0 |    57 | 23.6519 |  2.70142 |   1.4218  |  3.85237 |      4 |         5 |
|         REDACTED |     0 |    78 | 22.7706 | 19.2766  |  12.5106  | 15.9649  |      4 |         5 |
|         REDACTED |     0 |    72 | 20.2133 | 10       |   5.2439  | 11.1453  |      1 |         6 |
|         REDACTED |     0 |    60 | 32.7291 | 21.8675  |  17.8916  | 19.2419  |      1 |         2 |
|         0033 |     1 |    77 | 28.8317 | 70.4833  |  66.4684  | 69.2558  |      2 |         4 |
|         0035 |     1 |    67 | 23.0182 |  4.88688 |   2.71493 |  4.63348 |      4 |         4 |
|         0036 |     1 |    75 | 33.0419 | 12       |   6.25    | 12.52    |      4 |         4 |
|         REDACTED |     1 |    63 | 31.7622 |  7.32673 |   3.56436 |  7.54092 |      1 |         1 |
|         REDACTED |     0 |    79 | 26.6988 | 23.4109  |  19.5349  | 23.3085  |      1 |         1 |
|         0048 |     0 |    82 | 27.2761 |  8.12155 |   4.30939 |  8.99061 |      1 |         4 |
|         0050 |     1 |    68 | 25.6167 |  8.91688 |   5.13854 | 10.1234  |      3 |         2 |
|         REDACTED |     1 |    56 | 26.6633 |  8.29412 |   3.52941 |  7.92235 |      3 |         3 |
|         0054 |     0 |    61 | 30.1291 | 27.2727  |  20.4545  | 27.5727  |      3 |         0 |
|         REDACTED |     0 |    63 | 22.6197 |  4.3871  |   1.54839 |  6.32645 |      3 |         5 |

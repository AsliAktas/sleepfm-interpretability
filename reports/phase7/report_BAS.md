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

# SleepFM Interpretability — Phase 7 Raporu (BAS)

**Kohort:** 20 MESA hastasi
**Embedding boyutu:** 128 (modalite: BAS)

## Kume ozeti

- Kume sayisi: **4** ()
- Noise nokta sayisi: 0
- Kume dagilimi: `{0: 3, 1: 12, 2: 2, 3: 3}`

## Kume x klinik degisken testleri

| Degisken | Test | Statistic | p-degeri | Not |
|---|---|---:|---:|---|
| age | Kruskal-Wallis | 3.631 | 0.3042 |  |
| bmi | Kruskal-Wallis | 1.093 | 0.7788 |  |
| ahi | Kruskal-Wallis | 3.624 | 0.3051 |  |
| ahi_obs | Kruskal-Wallis | 2.714 | 0.4378 |  |
| odi3 | Kruskal-Wallis | 2.762 | 0.4298 |  |
| sex | chi-square | 2.292 | 0.5141 |  |

## Yorumlar

- n=20 kohort kucuk; testler kesin sonuc icin **degil**, kesif icin.
- p<0.05 sonuclar hipotez uretimi seviyesindedir; validation icin daha buyuk kohort gerekli.
- Noise orani: 0/20 = 0% (yuksekse HDBSCAN min_cluster_size dusurmek denenebilir)

## Gorseller

![umap_BAS](umap_BAS.png)

![boxplots_BAS](boxplots_BAS.png)

## Hastalar

|   subject_id |   sex |   age |     bmi |      ahi |   ahi_obs |     odi3 |   race |   cluster |
|-------------:|------:|------:|--------:|---------:|----------:|---------:|-------:|----------:|
|         REDACTED |     0 |    70 | 22.0108 | 18.1395  |   9.94186 | 17.1177  |      1 |         1 |
|         REDACTED |     0 |    83 | 21.898  |  8.34225 |   3.85027 |  9.20134 |      1 |         1 |
|         REDACTED |     0 |    57 | 56.0072 | 62.514   |  51.1173  | 65.114   |      4 |         1 |
|         REDACTED |     1 |    57 | 19.8446 | 40.9091  |  32.1818  | 42.7     |      1 |         1 |
|         REDACTED |     1 |    80 | 27.7738 | 27.4725  |  20.2198  | 26.0114  |      1 |         1 |
|         REDACTED |     0 |    60 | 31.4484 | 11.3744  |   6.54028 | 16.463   |      3 |         0 |
|         REDACTED |     0 |    57 | 23.6519 |  2.70142 |   1.4218  |  3.85237 |      4 |         2 |
|         REDACTED |     0 |    78 | 22.7706 | 19.2766  |  12.5106  | 15.9649  |      4 |         1 |
|         REDACTED |     0 |    72 | 20.2133 | 10       |   5.2439  | 11.1453  |      1 |         1 |
|         REDACTED |     0 |    60 | 32.7291 | 21.8675  |  17.8916  | 19.2419  |      1 |         1 |
|         0033 |     1 |    77 | 28.8317 | 70.4833  |  66.4684  | 69.2558  |      2 |         0 |
|         0035 |     1 |    67 | 23.0182 |  4.88688 |   2.71493 |  4.63348 |      4 |         1 |
|         0036 |     1 |    75 | 33.0419 | 12       |   6.25    | 12.52    |      4 |         1 |
|         REDACTED |     1 |    63 | 31.7622 |  7.32673 |   3.56436 |  7.54092 |      1 |         1 |
|         REDACTED |     0 |    79 | 26.6988 | 23.4109  |  19.5349  | 23.3085  |      1 |         1 |
|         0048 |     0 |    82 | 27.2761 |  8.12155 |   4.30939 |  8.99061 |      1 |         3 |
|         0050 |     1 |    68 | 25.6167 |  8.91688 |   5.13854 | 10.1234  |      3 |         3 |
|         REDACTED |     1 |    56 | 26.6633 |  8.29412 |   3.52941 |  7.92235 |      3 |         0 |
|         0054 |     0 |    61 | 30.1291 | 27.2727  |  20.4545  | 27.5727  |      3 |         2 |
|         REDACTED |     0 |    63 | 22.6197 |  4.3871  |   1.54839 |  6.32645 |      3 |         3 |

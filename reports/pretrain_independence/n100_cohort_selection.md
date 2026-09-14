# n=100 Clean Cohort — Selection and Plan (Phase 17)

## Amaç

Phase 15 defensibility raporu n=20 için "sağlam pozitif bulgu YOK" verdi —
istatistiksel güç yetersiz. n=100 ile:
- Cross-modality FWER'i geçen ilk gerçek sinyal(ler) mümkün olabilir
- Bootstrap-stability CV değerlerinin subject sayısı arttıkça düşmesi test edilecek
- Cohort seçim biası daha az kritik hale gelir (denetim uyarısı)

## Seçim Stratejisi

**Random sampling**, AHI-stratified değil. Denetim gerekçesi (Phase 13c
README #5): AHI-stratified seçim AHI ile ilişki testlerinde circular risk
yaratıyor. n=100 için doğal AHI dağılımı yeterli varyans sağlar.

- Kaynak: SleepFM `dataset_split.json` → `test` split (150 hasta)
- Mevcut 20 hasta (Phase 13c clean cohort) dahil
- 80 yeni hasta rasgele seçildi (`random.Random(42).sample`)
- Toplam: 100 hasta

## Pretrain-Independence Pre-Flight

100 subject'in tümü SleepFM `test` split'inde, `pretrain`de sıfır overlap.
Doğrulama: `scripts/verify_cohort_pretrain_independence.py --cohort-dir
data/n100_cohort_run/embeddings` (indirme sonrası koşulacak).

## Kaynak Tahmini

| Kaynak | Tahmin |
|---|---|
| Yeni indirme (80 subject × ~200 MB EDF + XML) | ~16 GB |
| Toplam disk (n=100 için) | ~20 GB |
| NSRR indirme (~1.5 dk/subject, 3 paralel) | ~40-60 dk |
| Preprocess EDF→HDF5 (80 yeni) | ~40 dk |
| SleepFM inference (80 yeni, GPU) | ~4 dk |
| Full analysis (rigor pass + defensibility) | ~10-15 dk |
| **Toplam wall-clock** | **~2 saat** |

## Subject IDs (100)

### Mevcut 20 (Phase 13c'den, preprocess'li)
```
[REDACTED_LIST_PER_DUA]
```

### Yeni 80 (indirilecek)
```
[REDACTED_LIST_PER_DUA]
```

## Uygulama Adımları (DUA onayı sonrası)

```powershell
# 1. Yeni 80 hastayı indir (mevcut 20 zaten data/clean_cohort_run/)
python scripts/download_nsrr_mesa.py `
  --out data/n100_cohort_run/raw `
  --subjects [REDACTED_LIST_PER_DUA] `
             [REDACTED_LIST_PER_DUA] `
             [REDACTED_LIST_PER_DUA] `
             [REDACTED_LIST_PER_DUA] `
             [REDACTED_LIST_PER_DUA] `
             [REDACTED_LIST_PER_DUA] `
             [REDACTED_LIST_PER_DUA] `
             [REDACTED_LIST_PER_DUA]

# 2. Mevcut 20 clean_cohort_run/ embeddings'i n100_cohort_run/ altına kopyala
# (veya symlink); disk tasarrufu için symlink önerilir

# 3. Yeni 80 için preprocess + inference
# (upstream SleepFM pipeline; bkz. REPRODUCE.md §6-7)

# 4. Pretrain-independence sanity check (CI-gate)
python scripts/verify_cohort_pretrain_independence.py `
  --cohort-dir data/n100_cohort_run/embeddings `
  --cohort-label n100_cohort

# 5. Full rigor pass on n=100
python src/clinical_analysis.py `
  --output-dir reports/phase17_rigor_n100

# 6. Phase 15 defensibility on n=100
python scripts/run_phase15_analysis.py `
  --output-dir reports/phase17_defensibility_n100 `
  --phase13c-dir reports/phase17_rigor_n100 `
  --n-permutations 5000

# 7. Bootstrap stability
python scripts/run_bootstrap_stability.py `
  --cohort-dir data/n100_cohort_run/embeddings `
  --output-dir reports/phase17_defensibility_n100

# 8. Regen REPRODUCE.md tables + commit
python scripts/regen_reproduce_tables.py > /tmp/n100_snippets.md
```

## Hipotezler — n=100'ün Test Edeceği

| Hipotez | n=20'de durum | n=100 beklenti |
|---|---|---|
| Cross-modality FWER'i geçen sinyal | 0 | 1-3 gerçek sinyal olabilir (EMG-BMI, MULTI-BMI ilk aday) |
| Subject-dropout CV | %20-35 | Düşer (%10-20 beklenir); n arttıkça |
| Chance baseline null_mean | ~0.11 | Düşer (~0.05) — büyük n = daha küçük random overlap |
| Label-shuffle p-tavan | 0.002 (500 perm floor) | 5000 perm ile 0.REDACTED'ye düşer, daha keskin |
| Ham SleepFM + KNN klinik risk | Ters yönlü | Muhtemelen aynı — bu **mimari sınır**, ölçek çözmez |

## Öncelikli Kararlar (Kullanıcıdan)

1. **DUA yasal onay** — henüz beklemede. NSRR'a mail atıldı, cevap gelecek.
2. **Random seed=42 kabul mi?** Farklı seed = farklı 80 hasta. Bir kez seçilirse
   tekrar üretilebilirlik için sabit tutulur.
3. **`data/n100_cohort_run/` .gitignore'da tutulacak** — 20 GB embeddings zaten
   .gitignore'da; sadece pretrain_independence CSV commit edilir.
4. **n=100'den sonra ne?** — mesa test split'in 150'sinin tümüne çıkma seçeneği
   var (~50 daha, +10 GB); veya CoxPH fine-tuning ile downstream clinical.

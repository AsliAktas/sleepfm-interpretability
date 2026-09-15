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

## Subject IDs — Kayıt Yeri (NSRR DAUA Uyum)

NSRR Data Access and Use Agreement Bölüm 5 gereği per-subject ID listeleri
public repository'de yayınlanmıyor. Reproducibility için:

- **Deterministic seçim reçetesi** aşağıda; herhangi biri aynı split dosyası
  + aynı seed ile aynı listeyi yeniden üretebilir
- Fiili listeler `data/private/subject_lists/` altında (git-ignored, sadece
  DUA imzalayan kullanıcının makinesinde)

**Reçete:**

```python
import json, random, re
SPLIT = json.load(open("sleepfm-clinical/sleepfm/configs/dataset_split.json"))
test_ids = sorted({re.search(r"mesa-sleep-(\d{4})", x).group(1)
                   for x in SPLIT["test"] if re.search(r"mesa-sleep-(\d{4})", x)})
# Phase 13c'de seçilen 20 hasta — reproduce için sabit
CURRENT_20 = sorted({...})  # data/private/subject_lists/n20_clean_cohort_full.csv
remaining = sorted(set(test_ids) - set(CURRENT_20))
new_80 = sorted(random.Random(42).sample(remaining, 80))
n100 = sorted(set(CURRENT_20) | set(new_80))
```

DUA imzalanmış kullanıcılar aynı SleepFM `dataset_split.json` (SHA256
`57d5019a...`) ile aynı 100 hastayı üretir.

## Uygulama Adımları (DUA onayı sonrası)

```powershell
# 1. Yeni 80 hastayı indir (mevcut 20 zaten data/clean_cohort_run/)
#    Subject listesi private dosyadan okunur:
$ids = (Get-Content data/private/subject_lists/n100_selection.txt | Where-Object { $_ -notmatch '^#' }) -join ' '
python scripts/download_nsrr_mesa.py --out data/n100_cohort_run/raw --subjects $ids.Split()

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
| Label-shuffle p-tavan | 0.002 (500 perm floor) | 5000 perm ile 2e-4 civarına düşer, daha keskin |
| Ham SleepFM + KNN klinik risk | Ters yönlü | Muhtemelen aynı — bu **mimari sınır**, ölçek çözmez |

## Öncelikli Kararlar

1. **DUA yasal onay** — 3 April 2026'da imzalandı, `Effective Date` = data release
2. **Random seed=42** — sabit, reproducibility için değiştirilmez
3. **`data/n100_cohort_run/` .gitignore'da** — hiçbir derived/raw veri commit edilmez (DUA Bölüm 5)
4. **n=100'den sonra ne?** — mesa test split'in 150'sinin tümüne çıkma seçeneği
   var (~50 daha, +10 GB); veya CoxPH fine-tuning ile downstream clinical.

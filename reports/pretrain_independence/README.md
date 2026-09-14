# Pretrain-Independence Cross-Check

Bu rapor Phase 15'in "clean cohort SleepFM pretrain'de değil" iddiasının
kod seviyesinde tekrarlanabilir kanıtıdır. Phase 16'da bağımsız denetim
tarafından işaret edilen boşluk kapatıldı — iddia artık sözlü değil,
CSV ile doğrulanabilir.

## Yöntem

SleepFM'in resmi split dosyası (`configs/dataset_split.json`, upstream
repo içinden) altı kümeye ayrılır: `pretrain`, `train`, `validation`,
`test`, `temporal_test`, `external_validation`. Clean cohort'un 20
subject ID'si her split ile küme kesişimi olarak karşılaştırılır.

Kod: [`scripts/verify_cohort_pretrain_independence.py`](../../scripts/verify_cohort_pretrain_independence.py)

## Sonuç — Clean Cohort (n=20)

| Split | Boyut | Overlap | Sonuç |
|---|---:|---:|---|
| pretrain | 1747 | **0** | Bu cohort SleepFM pretraining sırasında görülmedi |
| train | 149 | 0 | Downstream fine-tuning'de de kullanılmadı |
| validation | 10 | 0 | Validasyon setine dahil değil |
| test | 150 | **20/20** | Cohort tamamen SleepFM test split'inden çekildi |
| temporal_test | 0 | 0 | — |
| external_validation | 0 | 0 | — |

**Verdict:** Cohort SleepFM pretraining'den `defensibly held out`.
Phase 15'in memorization argümanı bu ön koşulda meşru.

## Çıktı Dosyası

- `clean_cohort_n20_vs_sleepfm_splits.csv` — 6 split × overlap sayısı + ID listesi

## Ne Zaman Yeniden Koşulmalı

Her yeni cohort için (n=100 dahil) bu script tekrar koşulmalı ve çıktı
`{cohort_label}_vs_sleepfm_splits.csv` olarak commit edilmelidir:

```bash
python scripts/verify_cohort_pretrain_independence.py \
  --cohort-dir data/<yeni_cohort>/embeddings \
  --cohort-label <yeni_cohort_ismi>
```

`pretrain overlap > 0` durumunda script exit code 1 döner ve
"memorization argument INVALID" uyarısı verir — CI'ya bağlanabilir.

## Neden Bu Kontrol Gerekli

SleepFM contrastive pretraining objective'i özünde subject-discrimination
öğrenir (pozitif çift = aynı hastanın iki chunk'ı, negatif = başka hasta).
Pretrain kohortunda görülen bir hastayı test etmek, model açısından
"bildik yüz" tanımaktır — subject purity gibi metrikler yapay olarak
şişer. Bu ayrım analizin geçerliliği için kritik.

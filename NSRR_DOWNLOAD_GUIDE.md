# Temiz Cohort İndirme Rehberi (Phase 8b)

## Neden

Mevcut 20 MESA hastası kontamine (17'si SleepFM pretrain'inde). Test split'ten yeni bir cohort indirilmesi gerekiyor.

## Önerilen 20 Hasta

AHI şiddetine göre stratifiye edilmiş 20 MESA hastası (SleepFM test split'ten, `random_state=42`):

| Kategori | Sayı | Hasta ID'leri |
|---|---:|---|
| **Normal** (AHI<5) | 5 | [REDACTED_LIST_PER_DUA] |
| **Mild** (AHI 5-15) | 5 | [REDACTED_LIST_PER_DUA] |
| **Moderate** (AHI 15-30) | 5 | 620, [REDACTED_LIST_PER_DUA] |
| **Severe** (AHI>30) | 5 | [REDACTED_LIST_PER_DUA] |

Detaylı liste: `scratchpad/clean_cohort_20.csv`

## İndirme Seçenekleri

### Seçenek 1: Python indirme scripti — ÖNERİLEN

Repo'nun içindeki `scripts/download_nsrr_mesa.py` scripti Ruby gem'e ihtiyaç
duymadan authenticated download yapar. Resume + retry + parallel + verify
dahil, `truststore` ile Windows SSL sorunlarını da atlar.

**Kurulum** (bir kere):
```powershell
# sleepfm env aktif olmalı; requirements.txt zaten truststore + requests içerir
conda activate sleepfm
```

**Token** — https://sleepdata.org/token adresinden al (MESA DUA gerekli):
```powershell
$env:NSRR_TOKEN = "senin_tokenin"
```

**Çalıştır** (default 20 stratified hasta, ~4 GB):
```powershell
python scripts/download_nsrr_mesa.py --out "C:/Users/User/Desktop/Projeler/SleepFM/mesa_test_clean"
```

Ya da custom subject list:
```powershell
python scripts/download_nsrr_mesa.py --subjects 620 REDACTED REDACTED --out .
```

Script:
- Token'ı önce `api/v1/account/profile.json` ile doğrular (hızlı fail).
- 3 paralel stream (NSRR rate limit koruma).
- Range header ile resume (kesilirse aynı komut kaldığı yerden devam eder).
- 3 retry + exponential backoff.
- Dosya-boyut sanity check (EDF ≥50 MB, XML ≥5 KB → hata sayfası inmiş mi kontrolü).
- Sonda `download_manifest.json` yazılır (hangi hasta OK, hangisi fail).

### Seçenek 2: NSRR Ruby CLI (alternatif)

Ruby zaten kuruluysa: `gem install nsrr` sonra `nsrr download <path>` ile.
Detay: https://github.com/nsrr/nsrr-gem

### Seçenek 3: Manuel Web İndirme

1. https://sleepdata.org/datasets/mesa/files
2. `polysomnography/edfs/mesa-sleep-{ID}.edf` her ID için tıkla
3. `polysomnography/annotations-events-nsrr/mesa-sleep-{ID}-nsrr.xml` aynı şekilde
4. Hepsini `C:/Users/User/Desktop/Projeler/SleepFM/mesa_test_clean/` altına koy

## Boyut ve Zaman Tahmini

| Öğe | Adet | Boyut/adet | Toplam |
|---|---:|---:|---:|
| EDF sinyal | 20 | ~200 MB | ~4 GB |
| XML annotation | 20 | ~50 KB | ~1 MB |
| **Toplam indirme** | | | **~4 GB** |

Süre: 20-40 dk (bağlantı hızına bağlı).

## İndirdikten Sonra

`preprocess_and_embed_smoke.py` script'ini `MESA_DIR` değişkeni ve `OUT_ROOT` değişkeni değiştirilerek tekrar çalıştır — pipeline aynı, sadece kaynak dizin farklı. Toplam ~15 dk çalışacak (10 dk preprocess + 5 dk inference + analiz).

```python
# Değiştirilecek satırlar:
MESA_DIR = Path("C:/Users/User/Desktop/Projeler/SleepFM/mesa_test_clean")
OUT_ROOT = Path("C:/Users/User/Desktop/Projeler/sleepfm_interpretability/data/clean_cohort")
```

Sonra `clinical_analysis.py`'ı `DEFAULT_REPORT_DIR = ... / "phase8"` ile çalıştırıp Phase 8 raporlarını üret.

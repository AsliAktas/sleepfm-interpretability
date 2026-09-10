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

### Seçenek 1: NSRR CLI (Ruby gem) — ÖNERİLEN

**Kurulum** (bir kere):
```bash
# Ruby ve gem yoksa: https://rubyinstaller.org (Windows için)
gem install nsrr
```

**Token ayarı** (NSRR hesap → sleepdata.org/token):
```bash
export NSRR_TOKEN="senin_tokenin"
# Windows PowerShell: $env:NSRR_TOKEN = "senin_tokenin"
```

**Toplu indirme**:
```bash
cd C:/Users/User/Desktop/Projeler/SleepFM/mesa_test_clean

# EDF'ler (~4 GB toplam)
for id in [REDACTED_LIST_PER_DUA] 620 [REDACTED_LIST_PER_DUA]; do
    padded=$(printf "%04d" $id)
    nsrr download mesa/polysomnography/edfs/mesa-sleep-${padded}.edf
    nsrr download mesa/polysomnography/annotations-events-nsrr/mesa-sleep-${padded}-nsrr.xml
done
```

### Seçenek 2: Manuel Web İndirme

1. https://sleepdata.org/datasets/mesa/files adresine git
2. `polysomnography/edfs/mesa-sleep-{ID}.edf` her ID için tıkla
3. `polysomnography/annotations-events-nsrr/mesa-sleep-{ID}-nsrr.xml` aynı şekilde
4. Hepsini `C:/Users/User/Desktop/Projeler/SleepFM/mesa_test_clean/` altına koy

### Seçenek 3: Python + NSRR API (yazılırsa)

NSRR gem yoksa, `nsrr-download.py` diye bir Python script hazırlanabilir. İhtiyacın olursa söyle.

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

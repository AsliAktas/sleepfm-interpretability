# NSRR MESA Cohort İndirme Rehberi

Bu rehber DUA imzalayan kullanıcının SleepFM `test` split'inden pretrain-
independent cohort'u indirmesi içindir. Phase 13c'de n=20 için kullanıldı;
Phase 17 planına göre n=100 için de aynı akış geçerlidir. Subject listesi
DAUA §5 gereği `data/private/subject_lists/` altında lokal olarak yaşar
ve bu belgede listelenmez — bkz. [REPRODUCE.md §4](REPRODUCE.md).

## Neden

SleepFM pretrain'de görülmüş subject'ler üzerinde analiz yapmak
memorization ile fizyolojik sinyali karıştırır. Test split'ten kohort
çekmek bu tuzağı ortadan kaldırır (bkz. [reports/pretrain_independence/](reports/pretrain_independence/README.md)).

## Önerilen 20 Hasta

AHI şiddetine göre stratifiye edilmiş 20 MESA hastası (SleepFM test split'ten, `random_state=42`):

| Kategori | Sayı | Hasta ID'leri |
|---|---:|---|
| **Normal** (AHI<5) | 5 | [REDACTED_LIST_PER_DUA] |
| **Mild** (AHI 5-15) | 5 | [REDACTED_LIST_PER_DUA] |
| **Moderate** (AHI 15-30) | 5 | [REDACTED_PER_DUA] |
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
python scripts/download_nsrr_mesa.py --out "<your-download-dir>"
```

Ya da custom subject list:
```powershell
python scripts/download_nsrr_mesa.py --subjects <ID1> <ID2> <ID3> --out .
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
4. Hepsini `<your-download-dir>/` altına koy

## Boyut ve Zaman Tahmini

| Öğe | Adet | Boyut/adet | Toplam |
|---|---:|---:|---:|
| EDF sinyal | 20 | ~200 MB | ~4 GB |
| XML annotation | 20 | ~50 KB | ~1 MB |
| **Toplam indirme** | | | **~4 GB** |

Süre: 20-40 dk (bağlantı hızına bağlı).

## İndirdikten Sonra

Ham EDF → HDF5 preprocess için SleepFM upstream repo'daki
`sleepfm/preprocessing/` scriptleri kullanılır. Ayrıntı ve komut zinciri:
[`REPRODUCE.md §6-7`](REPRODUCE.md).

Preprocess + inference + rigor pass toplam ~15 dk (n=20 için).

Sonuçları belirli bir raporda toplamak için:
```bash
python src/clinical_analysis.py --output-dir reports/<yeni_phase_ismi>
```

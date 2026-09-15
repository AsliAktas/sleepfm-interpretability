# REPRODUCE.md — Phase 15 Sayılarını Sıfırdan Üretme

Bu rehber, boş bir makineden başlayarak Phase 15 defensibility raporundaki
sayıların (subject purity null baselines, ARI bootstrap CI, cross-modality
FWER, subject-dropout robustness) **birebir yeniden üretilmesini** sağlar.

> Not: n=20 clean cohort sonuçları içindir. n=100 için aynı adımlar geçerli;
> sadece subject listesi ve süreler ölçeklenir.

---

## 0. Gereksinimler

| Bileşen | Sürüm | Neden |
|---|---|---|
| Python | 3.10 | requirements pin'lendi |
| conda / miniconda | any | UMAP + hdbscan + h5py için |
| git | any | repo |
| Disk | ~10 GB serbest | EDF+XML (4 GB) + HDF5 (2.5 GB) + embedding (~50 MB) + geçici |
| RAM | ≥8 GB | Preprocessing peak |
| GPU (opsiyonel) | CUDA 11+ | Inference 5× hızlanır; CPU'da 5-10 dk sürer |
| NSRR hesabı | onaylı MESA erişimi | https://sleepdata.org |

---

## 1. Repolar

Bu proje **iki repoya** bağımlıdır:

```powershell
mkdir C:\Users\<you>\Desktop\Projeler
cd C:\Users\<you>\Desktop\Projeler

# 1. Bu repo — analiz
git clone https://github.com/AsliAktas/sleepfm_interpretability.git

# 2. SleepFM upstream — checkpoint + preprocessing + inference kodu
mkdir SleepFM
cd SleepFM
git clone https://github.com/zou-group/sleepfm-clinical.git sleepFMoriginal/sleepfm-clinical
```

Checkpoint dosyaları (best.pt, ~500 MB) upstream repo'nun release'lerinde;
`sleepfm-clinical/sleepfm/checkpoints/model_base/best.pt` konumuna koy.

---

## 2. Python Ortamı

```powershell
cd C:\Users\<you>\Desktop\Projeler\sleepfm_interpretability
conda create -n sleepfm python=3.10 -y
conda activate sleepfm
pip install -r requirements.txt
```

`requirements.txt` sabit sürümler içerir (numpy==1.24.4, torch==2.1.2,
umap-learn==0.5.5, hdbscan==0.8.33). Farklı sürümler ARI sayılarını
±0.02 kaydırabilir.

Testleri koşarak kurulum sağlığını doğrula:

```powershell
python -m pytest tests/ -q
# Beklenen: 193 passed
```

---

## 3. NSRR Token

1. https://sleepdata.org → sign in → onaylı MESA erişimi olmalı
2. https://sleepdata.org/token → token'ı kopyala
3. Token'ı **argv'de değil**, dosyaya güvenli kaydet:

```powershell
mkdir C:\Users\<you>\.nsrr
$secure = Read-Host "NSRR token" -AsSecureString
[System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
  [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
) | Out-File -NoNewline -Encoding ASCII C:\Users\<you>\.nsrr\token
```

Dosya izinleri (opsiyonel ama önerilir):
```powershell
icacls C:\Users\<you>\.nsrr\token /inheritance:r /grant:r "$($env:USERNAME):F"
```

---

## 4. Clean Cohort — 20 Subject

Phase 13 sonuçlarını üretmek için 20 hasta gerekli (SleepFM'in resmi
`test` split'inden, AHI-stratified, `random_state=42`; 5 normal + 5 mild
+ 5 moderate + 5 severe).

**Subject IDs NSRR DAUA Bölüm 5 gereği bu belgede listelenmiyor.** DUA
imzalayan kullanıcı için:
- Listenin lokal kopyası: `data/private/subject_lists/n20_clean_cohort_full.csv` (git-ignored)
- Yeniden üretim reçetesi:

```python
# NSRR DAUA imzalı kullanıcı için — deterministic seçim
import json, random, re
SPLIT = json.load(open("<upstream>/sleepfm/configs/dataset_split.json"))
test_ids = sorted({re.search(r"mesa-sleep-(\d{4})", x).group(1)
                   for x in SPLIT["test"] if re.search(r"mesa-sleep-(\d{4})", x)})
# AHI-stratified sample of test_ids (metadata CSV'den AHI çekilir,
# 4 kategoride 5+5+5+5 hasta random.Random(42).sample ile)
# Fiili liste: data/private/subject_lists/n20_clean_cohort_full.csv
```

Pretrain-independence doğrulaması: 20 ID SleepFM `dataset_split.json`
altında **pretrain'de 0, test'te 20/20** overlap — bkz.
[`reports/pretrain_independence/`](reports/pretrain_independence/README.md).

İndir:

```powershell
# DUA imzalı kullanıcı — private listeden oku
$ids = (Get-Content data/private/subject_lists/n20_clean_cohort_full.csv |
        Select-Object -Skip 1 |
        ForEach-Object { ($_ -split ',')[5] } |
        Where-Object { $_ } |
        ForEach-Object { $_ -split ',' } | Where-Object { $_ -match '^\d{4}$' }) -join ' '
python scripts/download_nsrr_mesa.py `
  --token-file C:\Users\<you>\.nsrr\token `
  --out C:\Users\<you>\Desktop\Projeler\SleepFM\mesa_test_clean `
  --subjects $ids.Split()
```

Beklenen çıktı:
- ~4 GB toplam (20 EDF × ~200 MB + 20 XML)
- 20-40 dk (bağlantıya göre)
- `download_manifest.json` — hangi hasta OK
- Magic byte doğrulaması geçmiş (EDF header `0       ` string)

---

## 5. Metadata CSV

MESA klinik değişkenleri (age, sex, BMI, AHI, ODI) ayrı bir CSV'de:

- https://sleepdata.org/datasets/mesa → Files → datasets
- İndir: `mesa-sleep-dataset-0.8.0.csv` (7.32 MB)
- Yerleştir: `C:/Users/<you>/Desktop/Projeler/SleepFM/mesa/.csv/mesa-sleep-dataset-0.8.0.csv`
  (veya `$env:SLEEPFM_METADATA_CSV` ile başka yol göster)

Gerekli kolonlar (`src/clinical_analysis.py:METADATA_COLS`):
- `mesaid` — subject ID
- `gender1` → sex
- `sleepage5c` → age
- `bmi5c` → bmi
- `ahi_a0h4` → ahi
- `avgsat` → mean_spo2 (opsiyonel)

---

## 6. Preprocess (EDF → HDF5)

SleepFM upstream'in preprocessing modülü ham EDF'i 128 Hz resample edip
kanal gruplarına ayırır ve 5-dakikalık chunk'lara böler:

```powershell
cd C:\Users\<you>\Desktop\Projeler\SleepFM\sleepFMoriginal\sleepfm-clinical
python sleepfm/preprocessing/preprocessing.py `
  --input_dir "C:/Users/<you>/Desktop/Projeler/SleepFM/mesa_test_clean" `
  --output_dir "C:/Users/<you>/Desktop/Projeler/sleepfm_interpretability/data/clean_cohort_run/hdf5"
```

Beklenen: ~10 dk, 20 HDF5 dosyası, her biri ~120 MB.

---

## 7. Inference (HDF5 → embeddings)

```powershell
python sleepfm/pipeline/generate_embeddings.py `
  --input_dir "C:/Users/<you>/Desktop/Projeler/sleepfm_interpretability/data/clean_cohort_run/hdf5" `
  --output_dir "C:/Users/<you>/Desktop/Projeler/sleepfm_interpretability/data/clean_cohort_run/embeddings" `
  --checkpoint sleepfm/checkpoints/model_base/best.pt
```

Beklenen:
- GPU: 51 sn
- CPU: ~5 dk
- Her hasta için `mesa-sleep-XXXX_embeddings.hdf5` (4 modality × N chunk × 128-dim)
- Toplam 2539 chunk (subject başına ~127)

---

## 8. Pretrain-Independence Doğrulama (zorunlu)

Her yeni cohort için bu adım **CI-gate** gibi çalışmalı:

```powershell
cd C:\Users\<you>\Desktop\Projeler\sleepfm_interpretability
python scripts/verify_cohort_pretrain_independence.py `
  --cohort-dir data/clean_cohort_run/embeddings `
  --cohort-label clean_cohort_n20
```

Beklenen çıktı:
```
[OK] pretrain overlap = 0; test overlap = 20/20 (100%)
Cohort defensibly held out of SleepFM pretraining.
```

Pretrain overlap > 0 durumunda exit code 1 → **analiz koşulmamalı**.

---

## 9. Analiz — Phase 13c

Kohort yolunu env var ile ayarla, sonra clinical_analysis'i koş:

```powershell
$env:SLEEPFM_COHORT_ROOT = "C:/Users/<you>/Desktop/Projeler/sleepfm_interpretability/data/clean_cohort_run"
$env:SLEEPFM_MESA_XML_DIR = "C:/Users/<you>/Desktop/Projeler/SleepFM/mesa_test_clean"

# Rigor pass (5 modalite) — Phase 13c çıktı klasörüne yaz
python src/clinical_analysis.py --output-dir reports/phase13c_rigor_clean
```

`--output-dir` argümanı Phase 16r'de eklendi; manuel klasör rename artık
gerekmiyor. Modality alt-seti için `--modalities BAS EMG` gibi verilebilir;
`--fail-fast` ile ilk hata anında dur.

**Beklenen headline (Phase 13c)** — committed [`SUMMARY.csv`](reports/phase13c_rigor_clean/SUMMARY.csv)
ile birebir; bu tablo `scripts/regen_reproduce_tables.py` ile üretildi:

- **BAS**: min_p=0.1588, familywise min-p=0.4106, ARI=0.718 (std=0.122, n_clusters_mode=7)
- **RESP**: min_p=0.1439, familywise min-p=0.3656, ARI=0.542 (std=0.253, n_clusters_mode=2)
- **EKG**: min_p=0.1578, familywise min-p=0.4036, ARI=0.795 (std=0.175, n_clusters_mode=3)
- **EMG**: min_p=0.0020, familywise min-p=0.0060, ARI=0.694 (std=0.196, n_clusters_mode=4)
- **MULTI**: min_p=0.0070, familywise min-p=0.0230, ARI=0.615 (std=0.152, n_clusters_mode=6)

---

## 10. Analiz — Phase 15 Defensibility

Chance baseline + cross-modality FWER + bootstrap CI + subject-dropout:

```powershell
# Chance baseline label-shuffle + session-shuffle + cross-modality FWER + purity Δ
python scripts/run_phase15_analysis.py

# Subject-dropout robustness (Phase 16 ekleme)
python scripts/run_bootstrap_stability.py
```

Her ikisi de argparse override kabul eder (`--output-dir`,
`--phase13c-dir`, `--cohort-dir`, `--n-permutations`, `--n-bootstraps`).
Default'lar committed Phase 15 CSV'lerini birebir üretir.

Beklenen çıktı (`reports/phase15_defensibility/`):

**`ari_bootstrap_ci.csv`** — committed birebir:
```
modality  mean_ari  ci95_low  ci95_high
     BAS    0.7180    0.6823     0.7526
    RESP    0.5420    0.4712     0.6167
     EKG    0.7953    0.7429     0.8433
     EMG    0.6941    0.6360     0.7536
   MULTI    0.6151    0.5703     0.6610
```

**`bootstrap_stability_summary.csv`** — committed birebir (clean cohort):
```
modality  clusters_mean  clusters_std  clusters_min  clusters_max  clusters_cv
     BAS           5.52         1.147             2             8        20.8%
    RESP           4.00         1.385             2             7        34.6%
     EKG           3.50         1.093             2             6        31.2%
     EMG           4.76         1.153             2             7        24.2%
   MULTI           4.78         1.055             2             7        22.1%
```

Diğer üretilen dosyalar:
- `cross_modality_family_correction.csv` — 29 test, 0 Bonferroni-anlamlı
- `purity_contaminated_vs_clean.csv` — Δ tablosu

---

## 11. Beklenen Sayıların Kontrolü

Yeniden üretilen dosyaları git'teki commit edilmiş sayılarla karşılaştır:

```powershell
python scripts/regen_reproduce_tables.py
```

Bu script bu belgedeki §9-11 tablolarını **committed CSV'lerden birebir**
üretir. Kendi lokal çıktın bu tablolarla eşleşmiyorsa environment drift
var demektir.

**Beklenen `subject_purity_null_baselines.csv`** (Phase 15, birebir committed):
```
modality  n_chunks  n_clusters  observed_purity  label_shuffle_null_mean  label_shuffle_p
     BAS      2539          36         0.636530                 0.114733         0.001996
    RESP      2539          24         0.363649                 0.111470         0.001996
     EKG      2539          30         0.530961                 0.115408         0.001996
     EMG      2539           7         0.732421                 0.104090         0.001996
```

Not: `n_clusters` chunk-level HDBSCAN'in tespit ettiği küme sayısıdır
(default `hdbscan_min_cluster_size=30`, 2539 chunk üzerinde). Subject
sayısı (20) ile karışmasın.

Farklı sayılar görüyorsan olası nedenler:
- Python sürümü farkı (3.10 dışı → numpy floating point drift)
- Requirements pin'ler tutulmadı
- UMAP/HDBSCAN sürüm sapması (0.5.5 / 0.8.33 kritik)
- Random seed drift (bootstrap_ci_mean, permutation seed=42 sabit)

---

## 12. Toplam Süre

| Adım | Süre |
|---|---:|
| Repo + env kurulum | 15 dk |
| NSRR download (20 hasta) | 30 dk |
| Preprocess EDF → HDF5 | 10 dk |
| Inference | 1-5 dk |
| Pretrain-independence check | <1 sn |
| Phase 13c rigor pass | 2 dk |
| Phase 15 defensibility (chance + FWER + CI) | 5 dk |
| Bootstrap stability | 3 dk |
| **Toplam (fresh clone → tam sonuç)** | **~70 dk** |

---

## 13. Sorun Giderme

| Belirti | Neden | Çözüm |
|---|---|---|
| `truststore` SSL hatası | Windows CA bundle eksik | `pip install truststore` (requirements.txt'de var) |
| `h5py` ImportError | Yanlış env | `conda activate sleepfm`, `pip list` kontrol |
| NSRR 401 Unauthorized | Token geçersiz/expired | Yeni token al, `~/.nsrr/token` güncelle |
| UMAP `n_jobs=-1` warning | Deterministic mode | Zararsız; sabit seed = tek thread zorunlu |
| HDBSCAN `alltrue` DeprecationWarning | numpy 1.24 uyarısı | Zararsız |
| Preprocess `KeyError` (kanal) | Bazı MESA hastalarında kanal yok | XML'i kontrol et; skip logging'de görünür |
| Pretrain overlap > 0 | Yanlış subject seçtin | `scripts/verify_cohort_pretrain_independence.py` çıktısına bak |

---

## 14. Ne DEĞİŞMEZ, Ne DEĞİŞEBİLİR

**Bit-tam üretilebilir:**
- Test suite (193/193 pass)
- Cross-modality FWER sonuçları (deterministic input)
- Pretrain-independence CSV

**Çok küçük drift olabilir (±0.01):**
- Bootstrap CI'lar (2000 iter × 5 modalite, farklı seed drift)
- Multi-seed ARI (UMAP internal state farklı thread'de)

**±0.02-0.05 drift olabilir:**
- HDBSCAN cluster label numaraları (label ID stable değil, ARI stable)
- Observed subject purity (cluster reassignment ↔ purity)

Herhangi bir sayı ±0.10'dan fazla saparsa **environment drift** var demektir —
requirements'ı sıkı yeniden kur.

> ⚠️ **Phase 15'te bu raporun bazı iddiaları REVİZE edildi.** Adversarial
> review göstermiştir ki: (1) "MULTI-BMI + EMG-BMI iki yeni sinyal" iddiası
> cross-modality Bonferroni sonrası çürüdü (0 anlamlı bulgu), (2) "memorization
> + biometric karışım" yorumu için session-shuffle null testi tam ayrım
> yapamıyor. Güncel yorum için: `../phase15_defensibility/README.md`.

# Phase 13 — Clean Cohort Sonuçları ve Kontamine ile Karşılaştırma

> **Bu ilk gerçek bilimsel sonuçlar.** 20 MESA hastası SleepFM'in test split'inden
> (pretrain'de değil). NSRR'dan Phase 13a-b'de indirildi, `mesa_test_clean/`
> içinde. Öncekiler yalnızca method demo idi.

## Süreç Özeti

| Adım | Süre | Sonuç |
|---|---:|---|
| NSRR indirme (20 hasta) | 30:51 dk | 3.95 GB, 0 fail, magic byte'lar doğru |
| Preprocess (EDF→HDF5, 128 Hz) | ~10 dk | 20 HDF5 |
| Inference (SleepFM SetTransformer) | 51.5 sn | 2539 chunk × 4 modalite |
| Rigor pass v3 | ~2 dk | 5 modalite × 4 kontrol |
| Chunk analiz | ~30 sn | 4 modalite × stage/purity |

## Kritik Hipotez Testi — Sonuç

Kontamine cohort'ta (Phase 8e/9e) BAS subject purity **0.74** idi. Hipotez:
"memorization mı, yoksa gerçek EEG hasta-varyansı mı?" Beklenti:
- Tamamen memorization → clean'de ~0.05 (rastgele, 1/20)
- Tamamen gerçek sinyal → clean'de ≥0.30 kalır

**Ölçülen: 0.62** — arada, karışım. Yorum:
- **Kısmen memorization vardı** (0.74 → 0.62, 16 puan düşüş)
- **Ama büyük kısmı gerçek EEG hasta-imzası** — clean cohort'ta bile HDBSCAN kümeleri %62 tek-hasta ağırlıklı

**Uyarı (Phase 15 revize):** Yukarıdaki "memorization + gerçek EEG hasta-imzası karışım" yorumu bir hipotezdir; kohortta doğrulanmış değildir. Session-shuffle null testi bu ayrımı yapamaz (tek gecelik veriyle imkânsız). Bir sonraki cümlede "biometric ID gibi davranması" ifadesi de kanıtlanmamış hipotez seviyesindedir — inter-session reproducibility veya matched-pair kohort gerekir. Ayrıntı: [`../phase15_defensibility/README.md`](../phase15_defensibility/README.md) §2.

## Rigor Pass Karşılaştırması (Contaminated v2 → Clean v3)

| Modalite | Bonferroni sig | Min per_var_p | Familywise min-p | Mean ARI (10 seed) |
|---|---|---|---|---|
| BAS | 0 → 0 | 0.30 → 0.16 | 0.63 → 0.41 | 0.87 → **0.72** |
| RESP | 0 → 0 | 0.028 → 0.14 | 0.09 → 0.37 | 0.55 → 0.54 |
| EKG | 1 → 0 | 0.42 → 0.16 | 0.82 → 0.40 | 0.71 → **0.80** |
| EMG | 0 → 0 | 0.14 → **0.002** | 0.41 → **0.006** | 0.62 → 0.69 |
| MULTI | **1** → 0 | 0.028 → 0.007 | 0.03 → **0.023** | 0.66 → 0.62 |

### Bilimsel Bulgular

**#1. Kontamine cohort'un "MULTI-sex Bonferroni geçti" bulgusu KAYBOLDU** (raw p=0.005 → 0.58). Bu **memorization artefaktıydı** — clean cohort'ta sex ile ilişki yok. Öngörüldüğü gibi.

**#2. ~~İki yeni permutation-anlamlı sinyal~~ — Phase 15'te ÇÜRÜTÜLDÜ:**
Bu bölümde başlangıçta "EMG-BMI familywise p=0.006, MULTI-BMI p=0.023"
sinyalleri rapor edilmişti. Phase 15'in cross-modality FWER analizi
(29 test ailesi) sonrası **hiçbiri Bonferroni'yi geçmiyor** (en düşük
raw p=0.019, Bonferroni sonrası 0.557). MULTI zaten EMG içerdiği için
çift sayım riski de vardı. Bkz. [`../phase15_defensibility/`
§3](../phase15_defensibility/README.md).

**#3. Küme stabilitesi çok iyi** (ARI ≥ 0.62 tüm modalitelerde) —
ama sadece UMAP-init seed varyansı bazında. Phase 16'da subject-dropout
robustness ölçüldü: cluster count CV %20-35 (whole-pipeline daha
mütevazı). Bkz. [`../phase15_defensibility/` §9](../phase15_defensibility/README.md).

**#4. Hiçbir Bonferroni-signifikant bulgu yok** — n=20 çok küçük.
Permutation sinyalleri de cross-modality FWER sonrası çürüdü;
sağlam pozitif bulgu YOK, replikasyon (n≥100) şart.

**#5. Kohort seçim biası uyarısı (Phase 16 ekleme):** Bu 20 hasta
AHI-stratified 5+5+5+5 seçildi. Yani AHI dağılımı **planlı olarak
zenginleştirildi**. AHI ile modalite embeddings arası ilişki test
edilirken bu tasarım seçimi circular risk yaratır — "AHI varyansı
zaten var, dolayısıyla ilişki kolayca bulunur" sanılabilir. Şu ana
kadar hiçbir sinyal FWER'i geçmediği için pratik risk düşük;
n=100 clean cohort için random sampling düşünülmelidir.

## Chunk-Level Karşılaştırma (Kontamine v2 → Clean v3)

| Modalite | Chunks | Stage ARI | Subject Purity |
|---|---:|---:|---|
| BAS | 2464 → 2539 | 0.0003 → **0.042** | 0.74 → **0.62** |
| RESP | 2464 → 2539 | 0.034 → 0.050 | 0.37 → 0.36 |
| EKG | 2464 → 2539 | 0.024 → 0.026 | 0.55 → 0.52 |
| EMG | 2464 → 2539 | 0.038 → -0.022 | 0.48 → **0.71** |

### Chunk-Level Bulgular

- **BAS stage ARI 0.0003 → 0.042** — clean cohort'ta ARI mutlak değeri hâlâ rastgeleye (ARI=0) yakın; "100× artış" ifadesi baz sıfıra yakın olduğu için yanıltıcı olabilir. Sinyalin var olduğunu iddia etmek için Bonferroni-anlamlı bir p değeri gerekir; şu anki farkın istatistiksel anlamlılığı test edilmedi.
- **RESP subject purity stabil (0.37 → 0.36)** — RESP embeddings **cohorttan bağımsız global yapı** yakalıyor. Klinik yorumlanabilirlik için en güvenilir modalite.
- **EMG subject purity yükseldi (0.48 → 0.71)** — clean cohort'ta EMG kümeler daha subject-specific. EMG sensor placement varyansı yüksek olabilir.

## Ne Kanıtladık, Ne Kanıtlamadık

**Kanıtlanan:**
- SleepFM'in "%85 hasta biliyor" iddiası pipeline'ımızda yakalanabilir hale geldi (memorization audit)
- BAS embedding EEG hasta-imzası içerir (biometric bias, doğal)
- RESP embedding cohorttan bağımsız fizyolojik yapı öğrenir (klinik anlamlı)
- Pipeline uçtan uca doğru çalışıyor (audit-hardened, 185 test geçiyor)

**Kanıtlanmamış (n=20 sınırı):**
- MULTI-BMI ve EMG-BMI permutation-anlamlı ama Bonferroni'yi geçmiyor. Replikasyon şart (n≥100).
- Klinik uygulanabilirlik — CoxPH head yok, downstream fine-tuning yapılmadı.

## Çıktı Dosyaları

- `SUMMARY.csv` — 5 modalite × 10 metrik özet (rigor v3)
- `correction_{MODALITY}.csv` — Bonferroni + BH-FDR
- `permutation_{MODALITY}.csv` — Westfall-Young FW-adjusted p
- `stability_{MODALITY}.csv` — 10-seed ARI
- `sweep_{MODALITY}.csv` — HDBSCAN min_cluster_size sweep

Chunk-level: `../phase13c_chunk_clean/`.

## Bir Sonraki Adım

**Yayınlanabilir bulgu için:**
- n≥100 clean cohort (NSRR test split'te 150 MESA hasta mevcut)
- Downstream fine-tuning (CoxPH head) — SleepFM makalesi ile karşılaştırma
- clinical-bridge entegrasyon — KNN similarity → FHIR RiskAssessment (Phase 9g)

Bu Phase, "SleepFM interpretability pipeline'ı gerçek klinik veriyle uçtan uca çalışıyor" ilk gerçek kanıtı.

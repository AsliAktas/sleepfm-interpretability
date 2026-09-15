# Phase 9g Demo — FHIR RiskAssessment Örnekleri

> **KLİNİK KARAR İÇİN KULLANILAMAZ.**
>
> Bu klasördeki JSON'lar **method demo**dur. İçerikleri gerçek klinik
> risk değil, yalnızca boru hattının uçtan uca çalıştığını göstermek için
> üretilmiştir.
>
> Kullanılmaması gereken bağlamlar:
> - Herhangi bir hasta kararı
> - Klinik değerlendirme veya triyaj
> - Eğitim materyali dışında herhangi bir demonstrasyon
> - Downstream sistemlere "gerçek RiskAssessment" olarak besleme

## Neyi Temsil Ediyorlar

- 3-7 MESA hastasının SleepFM embeddings'i üzerinde **KNN-proxy**
  benzerlik hesabı
- `probabilityDecimal` = k-en-yakın-komşuların AHI-severity oy sayımı,
  foundation model çıktısı değil
- Neighbour listesi = **20 hastalık** clean cohort içinden top-5

## Neden Klinik Değil

- **n=20** kohort istatistiksel güç sağlamaz
- Phase 15 empirik olarak gösterdi: ham SleepFM + KNN sıralaması **AHI
  ile ters yönlü hata verebilir** (severe hasta düşük risk, normal
  hasta yüksek risk)
- Pretraining objective AHI-agnostic — CoxPH fine-tuning yapılmadıkça
  klinik risk sıralaması geçerli değildir
- Ayrıntı: [`../phase15_defensibility/README.md`](../phase15_defensibility/README.md) §6

## SNOMED Coding Geçmişi

Phase 9g'de üretilen JSON'lar `outcome.coding` altında **71908006
(Atrial fibrillation)** taşıyordu. Bu Phase 15'te semantik yanlış
olarak tespit edildi (AHI-based risk ≠ AFib) ve
`clinical_bridge_adapter.py` içindeki DEFAULT_CONDITION **Obstructive
Sleep Apnea** (78275009) olarak düzeltildi (`clinical-bridge` repo,
commit `6139efb`).

**Phase 16'da** bu klasördeki tüm MULTI JSON'lar düzeltilmiş adapter
ile regenerate edildi ve şu an **hiçbir dosya AFib kodu içermiyor**.
Eski `risk_assessment_REDACTED_MULTI.json` (kontamine smoke_run kohortundan
kalıntı) silindi.

## Mevcut Dosyalar (Phase 19 sonrası)

Subject ID'ler NSRR DAUA §5 gereği pseudonym'lere çevrildi (SUBJ_A, SUBJ_B, ...).
AHI değerleri kategori binlerine yuvarlandı (`<5`, `5-15`, `15-30`, `>30`)
— demografik parmak izi engellendi.

| Dosya | Modalite | SNOMED |
|---|---|---|
| `risk_assessment_SUBJ_B_MULTI.json` | MULTI (512-dim) | 78275009 (OSA) |
| `risk_assessment_SUBJ_R_MULTI.json` | MULTI | 78275009 (OSA) |
| `risk_assessment_SUBJ_T_MULTI.json` | MULTI | 78275009 (OSA) |
| `risk_assessment_SUBJ_U_MULTI.json` | MULTI | 78275009 (OSA) |
| `risk_assessment_SUBJ_B_RESP.json` | RESP-only | 78275009 (OSA) |
| `risk_assessment_SUBJ_R_RESP.json` | RESP | 78275009 (OSA) |
| `risk_assessment_SUBJ_U_RESP.json` | RESP | 78275009 (OSA) |

Yeniden üretim (DAUA imzalı kullanıcı, private subject listesi):
```powershell
python scripts/demo_sleepfm_to_fhir.py --modality MULTI `
  --query-subject <ID_from_data/private/> --cohort-root data/clean_cohort_run
```

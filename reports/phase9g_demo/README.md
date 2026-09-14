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

## Eski SNOMED Kodu Uyarısı

Bu JSON'lar Phase 9g döneminde üretildi ve `outcome.coding` altında
**71908006 (Atrial fibrillation)** SNOMED kodu içerir. Bu Phase 15'te
yanlış tanı grubu olarak tespit edildi ve `clinical_bridge_adapter.py`
içindeki DEFAULT_CONDITION **Obstructive Sleep Apnea** (78275009) olarak
düzeltildi (bkz. `clinical-bridge` repo, commit `6139efb`).

Bu klasördeki JSON'lar **eski SNOMED ile bırakıldı** çünkü:
- Ne zaman üretildiklerinin tarihsel kaydı olarak değerliler
- Regenerate etmek yeni Phase (16+) kapsamında yapılmalı, method-demo
  değil

Yeni demo üretmek için `scripts/demo_sleepfm_to_fhir.py` düzeltilmiş
adapter ile koşulur; çıktı doğru SNOMED kodunu içerir.

# src/legacy — Mock Disease Demo Evren (LEGACY)

> ⚠️ Bu katman **bilimsel çıkarım için kullanılmıyor.**
> Aktif katman: `src/` (real_embeddings, chunk_level_analysis,
> rigor_analysis, clinical_analysis, clinical_bridge_adapter, sleep_phases).

## Ne İçin Vardı

Proje Nisan 2026'da bu mock evrenle başladı:
- 500 sentetik hasta, 12 hastalık, 128-dim L2-normalized embeddings
- KNN Top-5 disease recommendation
- Mask-based ablation ile modality importance
- End-to-end pipeline demo

Amaç: gerçek SleepFM inference'a bağlamadan önce mimari doğrulamak.

## Neden Legacy

- Eylül 2026'da gerçek MESA cohort'una (20 hasta, SleepFM test split)
  taşındık; mock evren metodolojik-yorum üretemez
- `ANALIZ_RAPORU.md` HATA #1-#4'te bu katmandaki bilinen bug'lar
  belgelenmiş (label 0-11 vs 1-12 mismatch, KNN self-reference,
  ablation circular design). Fix'lenmedi çünkü katman zaten
  bilimsel çıkarım için kullanılmıyor.

## Dosyalar

| Dosya | Amaç |
|---|---|
| `mock_data.py` | 500 sentetik hasta + 12 hastalık üretici |
| `similarity_engine.py` | KNN Top-5 disease recommendation |
| `pipeline.py` | End-to-end mock inference pipeline |
| `config.py` | Mock disease names + constants |
| `metrics.py` | CoxPH concordance metrics (hiçbir yerde kullanılmıyor) |
| `visualization.py` | UMAP scatter + heatmap (yalnızca notebook'lardan çağrılıyor) |
| `run_ablation_demo.py` | Mask-based ablation demo runner |
| `test_pipeline_quick.py` | Quick smoke test for mock pipeline |

## Import Etkisi

- `src/utils.py` **legacy değil** — `real_embeddings.py` `normalize_l2`
  fonksiyonunu bu modülden alıyor. Bu yüzden `src/legacy/` değil
  `src/` altında kaldı.
- `conftest.py` `src/` ve `src/legacy/` ikisini de `sys.path`'e ekler,
  yani `from mock_data import ...` gibi eski import satırları
  değiştirilmedi — çalışmaya devam ediyor.

## Silinmeli mi?

Silme kararı bilinçli olarak ertelendi:
- Metodoloji öğrenme aşamasının tarihsel kaydını saklamak değerli
- Testlerin çalışmaya devam etmesi genel test suite health'i açısından
  fayda sağlıyor
- Fakat isteyen `git rm -r src/legacy/ tests/legacy/` ile temizleyebilir

## Tests

`tests/legacy/` altında 4 dosya, ~42 test. Bkz. `tests/legacy/README.md`.

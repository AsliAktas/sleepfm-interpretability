# tests/legacy — Mock Disease Demo Testleri (LEGACY)

> ⚠️ Bu testler `src/legacy/` altındaki mock disease demo katmanı içindir.
> Aktif test suite: `tests/` (test_paths, test_real_embeddings,
> test_rigor_analysis, test_clinical_analysis, vs.).

## Dosyalar

| Dosya | Test sayısı | Test ettiği |
|---|---:|---|
| `test_embedding_generation.py` | 26 | mock_data — shape, L2-norm, cluster structure, perturbation, reproducibility |
| `test_similarity_engine.py` | 5 | similarity_engine — KNN Top-5 disease recommendation |
| `test_pipeline.py` | 5 | pipeline — end-to-end mock inference |
| `test_ablation.py` | 6 | mask-based ablation |

Toplam ~42 test. Full suite (`pytest tests/`) hepsini koşar.

## Neden Hâlâ Duruyor

- Test suite health metric (193/193 pass) mock katmanı da kapsıyor
- Mimari testler (shape invariants, reproducibility, cluster structure)
  hâlâ bir sanity katmanı — mock veri üzerinde bile
- Silme kararı ertelendi; `src/legacy/README.md` bkz.

## CI

GitHub Actions bu testleri de koşar (`.github/workflows/tests.yml`).
Mock katman silinirse hem `src/legacy/` hem `tests/legacy/` birlikte
kaldırılmalıdır.

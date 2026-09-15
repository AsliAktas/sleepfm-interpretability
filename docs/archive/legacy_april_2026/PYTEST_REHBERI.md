# PYTEST BAŞLANGIÇ REHBERİ

pytest ilk kez kullanacakların için kısa ve pratik bir rehber.

---

## KURULUM

```bash
pip install pytest
```

Kurulumu doğrula:
```bash
pytest --version
```

---

## TEMEL MANTIK

pytest çok basit bir prensiple çalışır:
1. `test_` ile başlayan dosyaları bulur
2. İçindeki `test_` ile başlayan fonksiyonları çalıştırır
3. `assert` ifadeleri doğru mu kontrol eder
4. Geçen/kalan testleri raporlar

---

## İLK TESTİN

```python
# tests/test_example.py

def test_addition():
    assert 1 + 1 == 2

def test_string():
    assert "hello".upper() == "HELLO"
```

Çalıştır:
```bash
# Proje kök dizininde:
pytest tests/test_example.py -v
```

`-v` (verbose) her testin adını ve sonucunu gösterir.

Çıktı şöyle olacak:
```
tests/test_example.py::test_addition PASSED
tests/test_example.py::test_string PASSED
```

---

## ASSERT KALIPLARI

```python
# Eşitlik
assert result == expected

# Yaklaşık eşitlik (float karşılaştırma)
assert abs(result - expected) < 0.001
# veya
import numpy as np
np.testing.assert_allclose(result, expected, atol=1e-3)

# Shape kontrolü
assert embeddings.shape == (500, 128)

# Değer aralığı
assert 0 <= value <= 1

# Tür kontrolü
assert isinstance(result, np.ndarray)

# Hata bekleme
import pytest
with pytest.raises(ValueError):
    my_function(invalid_input)
```

---

## FIXTURE: PAYLAŞILAN TEST VERİSİ

Birden fazla testte aynı veriyi kullanacaksan, fixture kullan:

```python
import pytest
import numpy as np

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(100, 128)

# Bu fixture'ı kullanmak için parametre olarak geçir:
def test_shape(sample_embeddings):
    assert sample_embeddings.shape == (100, 128)

def test_values(sample_embeddings):
    assert not np.isnan(sample_embeddings).any()
```

---

## ÇALIŞTIRMA KOMUTLARI

```bash
# Tüm testleri çalıştır
pytest

# Belirli dosya
pytest tests/test_mock_data.py

# Belirli test
pytest tests/test_mock_data.py::test_embedding_shape

# Verbose (detaylı çıktı)
pytest -v

# İlk hatada dur
pytest -x

# Print çıktılarını göster
pytest -s

# Birleşik (en kullanışlı)
pytest -v -s tests/test_mock_data.py
```

---

## PROJE YAPILANMASI

pytest'in testleri bulabilmesi için proje kökünde `conftest.py` veya
`pytest.ini` / `pyproject.toml` ayarı gerekebilir.

En basit yol — proje kökünde:

```ini
# pytest.ini
[pytest]
testpaths = tests
```

Veya `pyproject.toml` kullanıyorsan:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## test_mock_data.py İÇİN YAZMAN GEREKEN TESTLER

Aşağıda test edilmesi gereken durumlar var. Her birini kendi kodunla yaz:

### generate_mock_embeddings() testleri:
1. **Shape doğru mu?** — (n_samples, embedding_dim) döndürmeli
2. **L2 norm ~1 mi?** — Her satırın normu 1'e yakın olmalı (tolerance: 1e-5)
3. **Label sayısı doğru mu?** — Unique label sayısı n_disease_groups'a eşit mi
4. **Seed tekrarlanabilirliği** — Aynı seed ile iki kez çağırınca aynı sonuç mu
5. **Değerler finite mi?** — NaN veya Inf yok mu

### generate_mock_coxph_scores() testleri:
6. **Shape doğru mu?** — (n_samples, n_diseases) döndürmeli
7. **Değerler makul aralıkta mı?** — Log-hazard ratioları aşırı büyük/küçük olmamalı

### generate_mock_modality_embeddings() testleri:
8. **Tüm modaliteler var mı?** — Dict'te "eeg", "ecg", "resp", "emg", "combined" key'leri
9. **Her modalite doğru shape'te mi?** — (n_samples, embedding_dim)
10. **Combined embedding L2 normalized mı?**

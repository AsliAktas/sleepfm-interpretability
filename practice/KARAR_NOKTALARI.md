# GÜN 5 — KARAR NOKTALARI

Bu dosyada bugün kod yazarken senin vermen gereken kararlar var.
Her karar için seçenekleri, artılarını ve eksilerini sunuyorum.
Kararı sen vereceksin.

---

## KARAR 1: Mock Embedding Üretim Stratejisi

Mock data "kasıtlı olarak beklenen sonuçları üretmemeli" (program kuralı #5).
Ama tamamen rastgele de olmamalı — o zaman test edemezsin.
Soru: Ne kadar yapı (structure) koyacaksın?
### Seçenek A: Kümeli + Gürültülü (Cluster-Based with Noise)
```
Mantık: Her hastalık grubu için bir "merkez" belirle, etrafına Gaussian noise ekle.
Kümeler belirgin ama mükemmel ayrılmış değil — gerçek veriye benzer.
```
- ✅ UMAP'ta anlamlı kümeler görürsün, HDBSCAN test edebilirsin
- ✅ Hastalık-küme ilişkisi var ama mükemmel değil (gerçekçi)
- ❌ Küme merkezlerini sen belirliyorsun — bias riski var

### Seçenek B: Karışık Gauss Dağılımı (Gaussian Mixture)
```
Mantık: sklearn GaussianMixture ile n_components kadar dağılım üret.
Overlap (örtüşme) oranını kontrol et.
```
- ✅ İstatistiksel olarak daha sağlam, overlap kontrol edilebilir
- ✅ Gerçek embedding dağılımlarına daha yakın
- ❌ Hastalık etiketleriyle doğrudan bağlantı kurmak ek adım gerektirir

### Seçenek C: Hibrit (Cluster + Random Perturbation)
```
Mantık: A'daki gibi kümeli üret, sonra noktaların %15-20'sini rastgele
başka kümelere taşı veya noise olarak işaretle. "Yanlış" kümelemeleri
kasıtlı olarak koy.
```
- ✅ En gerçekçi: gerçek veride de yanlış sınıflamalar olur
- ✅ Pipeline'ın edge case'lerini test eder
- ❌ En karmaşık implementasyon

**Senin kararın:** A, B, veya C?
Cevap: 
Seçenek A senin projen için zayıf. 12 hastalık sınıfın ve 3 modaliten var. Küme merkezlerini elle belirlediğinde, aslında hastalıklar arası ilişkileri sen tanımlıyorsun. Kardiyak aritmi ile epilepsi arasındaki mesafeyi sen koyuyorsun. Bu, biyomedikal bağlamda ciddi bir bias çünkü gerçekte bu ilişkileri keşfetmek istiyorsun, varsaymak değil.
Seçenek B istatistiksel olarak daha temiz ama senin interpretability amacınla çelişiyor. Gaussian mixture dağılımları anlamsal olarak opak — "bu bileşen neden bu şekilde" sorusuna cevap üretmez. Hastalık etiketleriyle bağlantının ek adım gerektirmesi sadece pratik bir zorluk değil, kavramsal bir kopukluk.
Seçenek C senin durumuna en uygun olanı, ve daha önce vardığımız kararla tutarlı. Sebebi şu: kasıtlı olarak yerleştirilen yanlış kümelemeler, kural #5 ile doğrudan uyumlu — sistem her zaman doğru sonucu üretecek şekilde tasarlanmamış oluyor. Aynı zamanda %15-20 pertürbasyon oranı, daha önce konuştuğumuz kademeli gürültü artışı stratejisiyle birleştirildiğinde kontrollü bir test alanı yaratıyor. Karmaşık implementasyon dezavantajı ise senin projende avantaja dönüşüyor çünkü 12 hastalık × 3 modalite zaten karmaşık bir yapı — basit bir mock bu karmaşıklığı temsil edemez.
Seçenek C'yi temel al. Üzerine modalite bazlı boyut gruplarını ekle — her küme merkezinin kardiyak, nöral ve respiratuar boyutlarda ayrı ayrı tanımlanmış bileşenleri olsun. Ağırlıkları modalite bazında ayarla — örneğin kardiyak hastalıklar için EKG boyutlarının ağırlığı yüksek, EEG boyutlarının düşük. Pertürbasyonları da modalite bazında uygula ki ablasyon testlerini çalıştırabilelim. Gürültüyü sıfırdan başlatıp kademeli artır.

---

## KARAR 2: Mock CoxPH Skorları — Embedding ile İlişki Düzeyi

Mock CoxPH skorları (hastalık risk skorları) embedding'lerle tutarlı olmalı ama mükemmel korelasyon olmamalı.
### Seçenek A: Embedding Mesafesine Dayalı
```
Mantık: Bir hastalık kümesinin merkezine uzaklık → risk skoru.
Yakın = yüksek risk. Gaussian noise ekle.
Korelasyon ~0.5-0.7 civarında tutar.
```
- ✅ Embedding ve risk arasında mantıklı ilişki var
- ✅ Ablation testlerinde anlamlı farklar görebilirsin
- ❌ "Gerçek" CoxPH böyle çalışmıyor (linear combination)

### Seçenek B: Linear Projection + Noise
```
Mantık: Rastgele ağırlık vektörü ile embedding'in dot product'ını al,
sigmoid uygula, noise ekle. CoxPH'ın gerçek çalışma mantığına daha yakın.
```
- ✅ CoxPH'ın gerçek matematiğine daha yakın
- ✅ Probing testlerinde de anlamlı çıktı verir
- ❌ Hangi boyutların "önemli" olduğunu sen belirliyorsun (ağırlık vektörüyle)

**Senin kararın:** A veya B?
Cevap:
Seçenek A neden interpretability'yi bozar: Küme merkezine uzaklık, tüm boyutları eşit ağırlıkla tek bir skaler mesafeye indirger. Kardiyak boyutların mı yoksa nöral boyutların mı risk skorunu yönlendirdiğini bu skordan çıkaramazsın. Ablasyon testlerinde "bu modaliteyi susturdum, risk skoru ne kadar değişti" diye sorduğunda, cevap mesafe metriğinin geometrisine bağlı olur — modalite ağırlıklarına değil. Karar 1'de kurduğun anlamsal yapı, risk skoruna geçişte kaybolur.
Seçenek B neden uyumlu: Linear projection, her boyutun risk skoruna katkısını açıkça tanımlar. Ağırlık vektörünü modalite gruplarına göre yapılandırabilirsin — EKG boyutlarına yüksek ağırlık, EEG boyutlarına düşük ağırlık atadığında, bu kardiyak bir hastalığın risk profili olur. Farklı hastalıklar için farklı ağırlık vektörleri tanımladığında, "bu hastalığın riski hangi sinyallerden türetiliyor" sorusu doğrudan okunabilir hale gelir.
"Ağırlıkları sen belirliyorsun" dezavantajına gelince: Bu, Karar 1'deki "küme merkezlerini sen belirliyorsun" bias riskiyle aynı sorun gibi görünüyor ama değil. Karar 1'de küme merkezlerini sen belirlemek, hastalıklar arası ilişkileri varsaymak demekti — keşfetmek istediğin şeyi önceden tanımlamak. Burada ise ağırlık vektörünü sen belirlemek, "bu mock senaryo şu klinik hipotezi temsil ediyor" demek. Bu varsayım değil, test senaryosu tanımı. Ve farklı ağırlık vektörleriyle farklı senaryoları test edebilirsin.
Kademeli gürültü stratejisiyle nasıl birleşir: Sıfır gürültüde linear projection deterministik çalışır — ağırlık vektörünün beklenen çıktıyı ürettiğini doğrularsın. Sonra modalite bazında gürültü eklediğinde, "EKG boyutlarına gürültü ekledim, kardiyak hastalıkların risk skoru ne kadar bozuldu" sorusunu sorabilirsin. Bu, tam olarak daha önce planladığımız ablasyon çalışmasının risk skoru tarafındaki karşılığı.
Tek dikkat noktası: ağırlık vektörlerini tamamen rastgele üretme. Modalite gruplarına göre yapılandır ki Karar 1'deki boyut grupları yapısıyla tutarlı kalsın. Rastgele bileşen, gürültüde olsun — ağırlıklarda değil.
---

## KARAR 3: Modalite Embedding'leri — Bağımsız mı, İlişkili mi?

SleepFM'de 4 modalite var: EEG (bas), ECG (ekg), Respiratory (resp), EMG (emg).
Her biri ayrı embedding üretiyor. Birleşik embedding hepsinin birleşimi.
### Seçenek A: Bağımsız Üretim
```
Mantık: Her modalite embedding'ini bağımsız üret (ayrı Gaussian).
Birleşik = concat veya ortalama.
```
- ✅ Basit, hızlı
- ❌ Gerçekte modaliteler birbirleriyle ilişkili (kalp ritmi solunum etkiler)

### Seçenek B: İlişkili Üretim (Correlated)
```
Mantık: Önce birleşik embedding üret, sonra her modaliteye project et.
Bazı boyutları paylaşır (shared variance), bazılarını paylaşmaz (unique).
shared_ratio parametresiyle kontrol et.
```
- ✅ Daha gerçekçi — ablation testlerinde "bir modaliteyi kaldırınca diğerleri hâlâ bilgi taşıyor" durumunu yakalar
- ✅ Convergence framework testleri için daha anlamlı
- ❌ Daha karmaşık implementasyon

**Senin kararın:** A veya B?
Cevap:
Seçenek B, ve bu önceki iki kararın doğal sonucu.
Mantık zinciri şu şekilde kuruluyor:
Karar 1'de modalite bazlı boyut grupları tanımladın ve bu grupların ağırlıklarını bağımsız olarak ayarlayarak ablasyon testleri yapma kararı aldın. Karar 2'de risk skorlarını linear projection ile üretmeyi seçtin, böylece her modalite boyutunun risk skoruna katkısı okunabilir hale geldi. Bu iki karar birlikte şunu varsayıyor: modaliteler arasındaki ilişki, sistemin anlaması gereken bir bilgi — test altyapısının görmezden gelebileceği bir detay değil.
Şimdi Seçenek A'yı bu çerçeveye oturtmaya çalışırsan ne olur: bağımsız üretimde her modalite kendi Gaussian dağılımından gelir, aralarında istatistiksel bağımlılık yoktur. Bu durumda ablasyon testinde bir modaliteyi kaldırdığında, kalan modalitelerde o modaliteye dair hiçbir iz kalmaz. Ama klinik gerçeklikte kalp ritmi bozukluğu solunum düzenini etkiler, nöral aktivite kas tonusunu değiştirir. Bağımsız mock embedding'lerle bu etkileşimi test edemezsin — daha önemlisi, Karar 1'deki ağırlıklı yapın bu etkileşimi yakalamak üzere tasarlanmıştı. Mock veri bu etkileşimi içermezse, ağırlık mekanizmasını test ettiğini sanırsın ama aslında bağımsız dağılımların aritmetiğini test edersin.
Seçenek B'nin interpretability planına katkısı ise şu noktada belirginleşiyor: shared_ratio parametresi, kendi başına yorumlanabilir bir değişken. Bu parametreyi değiştirdiğinde "modaliteler arası paylaşılan varyans arttıkça risk skoru tahmininde hangi modalite daha az bilgi kaybediyor" sorusunu sorabilirsin. Bu, Karar 2'deki ağırlık vektörleriyle birlikte çalışır — ağırlık vektörü "hangi modalite önemli" sorusuna, shared_ratio ise "modaliteler birbirinin bilgisini ne kadar taşıyor" sorusuna cevap verir. İkisi birlikte, tek başlarına veremeyecekleri bir görünürlük sağlar.
Karmaşıklık konusunda şunu belirteyim: implementasyon olarak B daha zor ama mimari olarak önceki kararlarınla tutarlı. Birleşik embedding'den modalitelere projection yapısı, Karar 2'deki linear projection mantığının tersidir — orada embedding'den skora gidiyordun, burada birleşikten modaliteye gidiyorsun. Aynı matematiksel araç, ters yönde. Dolayısıyla bir kez kurduğunda iki yönlü okunabilirlik elde edersin: birleşikten modaliteye ayrıştırma ve modaliteden riske projeksiyon.
Tek uyarım şu: shared_ratio'yu başlangıçta sabit tut, kademeli gürültü stratejinde olduğu gibi önce deterministik durumu doğrula. Sonra shared_ratio'yu değiştirerek modaliteler arası bağımlılığın etkisini gözlemle. Bu, üçüncü bir ablasyon ekseni olur — birincisi modalite bazlı gürültü, ikincisi ağırlık vektörü değişimi, üçüncüsü shared_ratio değişimi. Her biri farklı bir soruya cevap verir ve birbirini tamamlar.

---

## KARAR 4: metrics.py — Bootstrap CI Implementasyonu

Bootstrap confidence interval hesaplayacaksın. İki yaklaşım var:
### Seçenek A: Manuel Bootstrap (numpy ile)
```python
# Kendim yazarım: sample with replacement, metric hesapla, tekrarla, percentile al
```
- ✅ Tam kontrol — ne olduğunu bilirsin
- ✅ Dışarıdan kütüphane bağımlılığı yok
- ✅ Öğrenme değeri yüksek
- ❌ Edge case'leri (NaN handling, küçük sample) kendin yönetirsin

### Seçenek B: scipy.stats.bootstrap kullanımı
```python
# scipy.stats.bootstrap(data, statistic, n_resamples=1000, confidence_level=0.95)
```
- ✅ Tek satır, test edilmiş, edge case'ler handle edilmiş
- ✅ BCa (bias-corrected accelerated) gibi gelişmiş yöntemler mevcut
- ❌ Kara kutu — içini görmezsin
- ❌ scipy versiyonuna bağımlılık (1.7+ gerekli)

**Senin kararın:** A veya B?
Cevap:
Seçenek A, ve bu sefer gerekçe doğrudan interpretability planından önce daha temel bir yere dayanıyor.
Önceki üç kararında tutarlı bir ilke vardı: mock verinin her katmanında ne olduğunu bilmek ve kontrol etmek istiyorsun. Modalite boyutlarını sen tanımlıyorsun, ağırlık vektörlerini sen yapılandırıyorsun, shared_ratio'yu sen belirliyorsun. Her noktada "bu çıktı neden böyle" sorusuna cevaplayabilir olmayı tercih ettin. Bootstrap CI hesaplaması da bu zincirin bir halkası — metriklerini değerlendirdiğin araç.
Scipy'nin bootstrap fonksiyonu istatistiksel olarak sağlam, bunu tartışmıyorum. Ama senin projenin bağlamında şu sorun ortaya çıkıyor: üç karar boyunca mock veride interpretability için katman katman yapı kurdun. Sonra bu yapının ürettiği sonuçları değerlendiren metrik aracının içini göremiyorsan, bir kopukluk oluşur. Bootstrap CI'da beklenmedik bir sonuç aldığında — diyelim güven aralığı anormal genişlikte çıktı — sebebin mock verideki shared_ratio ayarında mı, gürültü seviyesinde mi, yoksa bootstrap'ın iç mekanizmasında mı olduğunu ayırt edemezsin. Kara kutu metrik aracı, şeffaf mock verinin ürettiği sinyalleri bulandırır.
Manuel implementasyonun sana verdiği şey, bootstrap sürecinin her adımını gözlemlenebilir kılmak. Hangi örneklerin seçildiğini, her iterasyonda metriğin nasıl değiştiğini, dağılımın şeklini doğrudan inceleyebilirsin. Bu, ablasyon testlerinde kritik hale gelir — modalite bazlı gürültü eklediğinde CI genişliği nasıl değişiyor sorusu, bootstrap mekanizmasını kontrol edebildiğinde çok daha anlamlı bir cevap üretir.
Edge case yönetimi konusundaki dezavantaj gerçek ama yönetilebilir. Senin vesetinde 12 hastalık sınıfı var, bazı sınıflarda küçük sample durumu olabilir. Ama bu durumları kendin yönetmek aslında bir dezavantaj değil — küçük sample'da CI'nın nasıl davrandığını görmek, mock verinin o hastalık sınıfı için yeterli temsil gücüne sahip olup olmadığını anlamana yardımcı olur. Scipy bunu sessizce handle ettiğinde bu bilgiyi kaybedersin.
Pratik bir önerim var: BCa yöntemini ilk aşamada implementasyon kapsamına alma. Temel percentile bootstrap ile başla — bu kavramsal olarak basit ve şeffaf. Eğer ileride bias correction gerektiğini veriden gözlemlersen, o zaman BCa'yı eklersin. Bu, gürültü stratejindeki "sıfırdan başla, kademeli artır" mantığının metrik tarafındaki karşılığı.



---

## KARAR 5: Proje Genelinde — requirements.txt Yönetimi
### Seçenek A: Manuel Liste
```
# Sadece kullandığın paketleri elle yaz, versiyon pinle
numpy==1.24.0
pandas==2.0.0
...
```
- ✅ Temiz, sadece gerçekten kullandıkların var
- ❌ Versiyon uyumluluğunu kendin takip etmelisin

### Seçenek B: pip freeze > requirements.txt
```
# Ortamdaki her şeyi dök
```
- ✅ Tam reproducibility
- ❌ Çok fazla gereksiz paket listelenir, okunması zor

### Seçenek C: Kategorize Manuel Liste
```
# Core
numpy>=1.24
pandas>=2.0

# ML
scikit-learn>=1.3
lifelines>=0.27

# Visualization
matplotlib>=3.7
seaborn>=0.12

# Clustering
umap-learn>=0.5
hdbscan>=0.8

# Testing
pytest>=7.0
```
- ✅ Hem temiz hem kategorize — başkası okuyunca ne niye var anlar
- ❌ Minimum versiyon belirtir ama tam pin değil

**Senin kararın:** A, B, veya C?

Cevap:
"""Seçenek C, ama bir modifikasyonla.
Bu karar önceki dördünden farklı bir düzlemde duruyor — doğrudan mock veri veya metrik tasarımıyla ilgili değil. Ama interpretability planınla bağlantısı var ve o bağlantı şu: projenin okunabilirliği.
Beş karar boyunca şunu inşa ettin: modalite bazlı boyut grupları, ağırlıklı risk skorları, ilişkili modalite üretimi, şeffaf bootstrap. Her katmanda "bu neden böyle" sorusuna cevap verebilmeyi öncelik yaptın. Requirements dosyası, bu projeye ilk kez bakan birinin karşılaşacağı ilk teknik belgelerden biri. O kişi — ya da altı ay sonra sen — bağımlılık listesine baktığında, projenin mimarisini kategorilerden okuyabilmeli.
Seçenek A temiz ama sessiz — numpy ve lifelines yan yana durduğunda aralarındaki işlevsel farkı dosya göstermiyor. Seçenek B gürültülü — tam olarak Karar 1'deki tamamen rastgele mock veri sorunuyla aynı yapıda, bilgi var ama sinyal gürültüye gömülü. Seçenek C, bağımlılıkları anlamsal kategorilere ayırarak okunabilirliği sağlıyor.
Modifikasyon şu: minimum versiyon yerine tam pin kullan ama kategorizasyonu koru.
# Core
numpy==1.24.0
pandas==2.0.0

# Survival Analysis
lifelines==0.27.8

# Clustering
umap-learn==0.5.5
hdbscan==0.8.33
Sebebi basit: senin projen üç ayrı boyutta ablasyon testleri çalıştıran, modaliteler arası bağımlılığı parametrik olarak kontrol eden bir mock altyapısı. Bu altyapıda bir paketin minör versiyon farkı, sayısal sonuçları değiştirebilir — özellikle numpy'ın floating point davranışı ve scikit-learn'ün random state yönetimi versiyonlar arası farklılık gösterir. Minimum versiyon belirttiğinde reproducibility'yi garanti edemezsin. Tam pin, sıfır gürültüdeki deterministik baseline'ını korur.
Yani sonuç: C'nin kategorizasyonu + A'nın versiyon pinlemesi. Okunabilirlik ve tekrarlanabilirlik birlikte sağlanır.
"""

---

## KARAR 6: pytest — Test Organizasyonu
### Seçenek A: Tek Dosya, Fonksiyon Bazlı
```python
# tests/test_mock_data.py
def test_embedding_shape():
    ...
def test_embedding_norm():
    ...
```
- ✅ Basit, hızlı başlarsın
- ✅ pytest ile ilk deneyimin için yeterli
- ❌ Test sayısı artınca karışabilir

### Seçenek B: Sınıf Bazlı Gruplama
```python
# tests/test_mock_data.py
class TestEmbeddingGeneration:
    def test_shape(self):
        ...
    def test_norm(self):
        ...

class TestCoxPHScores:
    def test_range(self):
        ...
```
- ✅ İlişkili testler gruplanmış, okunması kolay
- ✅ pytest ile tam uyumlu
- ❌ Biraz daha fazla boilerplate

**Senin kararın:** A veya B?
Cevap:
Seçenek B, ama dosya yapısını da düşünmen gerekiyor.
Projenin test edilecek katmanlarını say: embedding üretimi (modalite bazlı boyut grupları, shared_ratio, kümeleme yapısı), CoxPH skorları (linear projection, ağırlık vektörleri), bootstrap CI hesaplaması, ablasyon testleri (modalite bazlı gürültü, ağırlık değişimi, shared_ratio değişimi). Bunların hepsi tek dosyada sınıf bazlı gruplansa bile dosya çok uzar ve sınıflar arası ilişkiler dosya içinde görünmez hale gelir.
Ama asıl mesele dosya boyutu değil. Senin projenin yapısında testler arasında hiyerarşik bir ilişki var. Embedding üretim testleri geçmeden CoxPH skor testlerinin anlamı yok — çünkü risk skoru embedding'den türetiliyor. CoxPH testleri geçmeden ablasyon testlerinin anlamı yok — çünkü ablasyon, risk skorundaki değişimi ölçüyor. Bu bağımlılık zincirini tek dosyadaki sınıflar ifade edemez.
Önerim şu: sınıf bazlı gruplama kullan ama dosyaları katmanlara göre ayır.
tests/
  test_embedding_generation.py
    class TestModalityDimensions
    class TestSharedVariance
    class TestClusterStructure

  test_coxph_scores.py
    class TestLinearProjection
    class TestModalityWeights

  test_bootstrap_ci.py
    class TestBasicBootstrap
    class TestSmallSample

  test_ablation.py
    class TestModalityNoise
    class TestWeightPerturbation
    class TestSharedRatioVariation
Bu yapı, Karar 1'den 4'e kadar kurduğun mimariyi birebir yansıtıyor. Her dosya bir karara, her sınıf o kararın içindeki bir bileşene karşılık geliyor. Bir test başarısız olduğunda dosya adından hangi katmanda sorun olduğunu, sınıf adından hangi bileşende olduğunu, fonksiyon adından ne beklentinin karşılanmadığını okursun. Bu, interpretability planının test altyapısındaki karşılığı — "bu sonuç neden böyle" sorusu sadece mock veri ve metrikler için değil, test başarısızlıkları için de cevaplanabilir olmalı.
Fonksiyon bazlı tek dosya yaklaşımı bu okunabilirliği sağlayamaz. Sınıf bazlı tek dosya yaklaşımı kısmen sağlar ama katmanlar arası ayrımı gösteremez. Sınıf bazlı çoklu dosya, projenin mimarisini test organizasyonuna taşır.



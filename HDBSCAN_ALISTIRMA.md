# HDBSCAN HANDS-ON ALIŞTIRMA

Bu dosyayı notebook'unda adım adım takip edeceksin.
Kodları direkt kopyalama — anlayarak yaz. Yorumlar sana ne yapman gerektiğini söylüyor.

---

## Adım 1: Kütüphaneler

```python
# Bu paketler lazım, yoksa kur:
# pip install hdbscan umap-learn scikit-learn matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import hdbscan
```

---

## Adım 2: Basit Kümelenmiş Veri Üret

```python
# make_blobs: Belirli merkezler etrafında Gaussian kümeler üretir
# n_samples=300: toplam 300 nokta
# centers=4: 4 küme
# cluster_std=0.8: küme içi dağılım (büyük = daha dağınık)
# random_state=42: tekrarlanabilirlik

X_blobs, y_blobs = make_blobs(
    n_samples=300, centers=4, cluster_std=0.8, random_state=42
)

# Görselleştir — renk = gerçek etiket
plt.figure(figsize=(8, 6))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap="viridis", s=15)
plt.title("Ground Truth Labels (make_blobs)")
plt.show()
```

**Gözlemle:** 4 ayrı küme görüyor musun? Ayrım net mi?

---

## Adım 3: HDBSCAN Uygula

```python
# min_cluster_size=15: bir küme en az 15 noktadan oluşmalı
clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
cluster_labels = clusterer.fit_predict(X_blobs)

# Kaç küme buldu?
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()
print(f"Found {n_clusters} clusters, {n_noise} noise points")

# Görselleştir — renk = HDBSCAN'ın bulduğu kümeler
plt.figure(figsize=(8, 6))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=cluster_labels, cmap="viridis", s=15)
plt.scatter(
    X_blobs[cluster_labels == -1, 0],
    X_blobs[cluster_labels == -1, 1],
    c="red", marker="x", s=30, label="Noise"
)
plt.legend()
plt.title(f"HDBSCAN (min_cluster_size=15): {n_clusters} clusters, {n_noise} noise")
plt.show()
```

**Gözlemle:** Gerçek etiketlerle karşılaştır. Doğru buldu mu? Noise noktaları mantıklı mı?

---

## Adım 4: min_cluster_size Etkisini Gözlemle

```python
# 3 farklı değer dene
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, mcs in zip(axes, [5, 15, 30]):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
    labels = clusterer.fit_predict(X_blobs)
    n_c = len(set(labels)) - (1 if -1 in labels else 0)
    n_n = (labels == -1).sum()

    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, cmap="viridis", s=15)
    ax.scatter(
        X_blobs[labels == -1, 0], X_blobs[labels == -1, 1],
        c="red", marker="x", s=30
    )
    ax.set_title(f"min_cluster_size={mcs}\n{n_c} clusters, {n_n} noise")

plt.tight_layout()
plt.show()
```

**Gözlemle ve not al:**
- min_cluster_size=5 ile kaç küme var? Çok mu parçalanmış?
- min_cluster_size=30 ile kaç küme var? Küçük gruplar kayboldu mu?
- Hangi değer en mantıklı sonucu veriyor?

---

## Adım 5: Düzensiz Şekiller (K-Means vs HDBSCAN)

```python
# make_moons: Hilal şekilli kümeler — K-Means bunu yapamaz
X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gerçek etiketler
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="viridis", s=15)
axes[0].set_title("Ground Truth")

# K-Means
from sklearn.cluster import KMeans
km_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_moons)
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=km_labels, cmap="viridis", s=15)
axes[1].set_title("K-Means (k=2)")

# HDBSCAN
hdb_labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(X_moons)
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=hdb_labels, cmap="viridis", s=15)
axes[2].set_title("HDBSCAN")

plt.tight_layout()
plt.show()
```

**Gözlemle:** K-Means hilal şekilli kümeleri doğru bulabiliyor mu? HDBSCAN?
Bu, projedeki embedding kümelerinin neden HDBSCAN ile analiz edileceğini gösteriyor.

---

## Adım 6: Yüksek Boyutlu Veri (Projeye Hazırlık)

```python
# Gerçek projede 128 boyutlu embedding'ler olacak
# Şimdi 128-dim simüle et, UMAP ile 2D'ye indir, HDBSCAN ile kümele

import umap

# 128 boyutlu veri: 5 küme, biraz overlap
X_high, y_high = make_blobs(
    n_samples=500, n_features=128, centers=5, cluster_std=3.0, random_state=42
)

# UMAP ile 2D'ye indir
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_2d = reducer.fit_transform(X_high)

# HDBSCAN kümele (2D üzerinde)
hdb = hdbscan.HDBSCAN(min_cluster_size=20)
hdb_labels = hdb.fit_predict(X_2d)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_high, cmap="tab10", s=10)
axes[0].set_title("UMAP — Ground Truth Labels")

axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=hdb_labels, cmap="tab10", s=10)
n_c = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
axes[1].set_title(f"UMAP + HDBSCAN — {n_c} clusters found")

plt.tight_layout()
plt.show()
```

**Gözlemle:** HDBSCAN 5 kümeyi bulabildi mi? Gerçek etiketlerle ne kadar uyuşuyor?

---

## ÖNEMLİ NOTLAR

1. HDBSCAN'ı 2D veya UMAP çıktısı üzerinde çalıştırmak yaygındır,
   ama yüksek boyutlu orijinal veri üzerinde de çalıştırılabilir.
   Projede ikisini de deneyeceksin.

2. `clusterer.probabilities_` ile her noktanın küme üyelik güvenilirliğini görebilirsin.
   Düşük güvenilirlikli noktalar sınır bölgelerinde.

3. `clusterer.labels_` = -1 olan noktalar gürültü. Bunları analiz dışı bırakma,
   neden gürültü olduklarını araştır — belki ilginç outlier'lardır.

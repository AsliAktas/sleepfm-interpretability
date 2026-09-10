# Phase 9e — Chunk-Level Cluster × Sleep Stage Analysis (v2)

> **UYARI:** Bu rapor Phase 7/8c/8e ile aynı kontamine cohort'a dayanır (17/20 pretrain,
> 2/20 train, 1/20 test). Sonuçlar bilimsel iddia değil, method demonstration.
> Temiz cohort geldiğinde aynı script tekrar çalışacak.

## Ne Yaptık

20 MESA hastasının **her 5-dakikalık embedding chunk'ını ayrı bir data noktası** olarak
ele aldık (~2464 nokta). Per-subject analizi (Phase 7/8) sadece "hasta X hasta Y'ye benziyor
mu?" sorusunu cevaplayabilir; chunk-level analiz iki yeni soruyu açar:

1. **Küme ↔ Uyku fazı ilişkisi**: HDBSCAN'ın bulduğu kümeler MESA XML'inde
   annotate edilmiş sleep stage'lere karşılık mı geliyor?
2. **Subject purity**: Bir küme çoğunlukla tek bir hastanın chunk'larından mı
   ibaret (single-patient artefakt) yoksa hastalar arası paylaşılan bir
   pattern mı temsil ediyor?

Her chunk için MESA XML annotation'larından dominant sleep stage
(`sleep_phases.label_chunks`) ile HDBSCAN cluster label'ı yan yana konup:
- **ARI(cluster, stage)** — cluster ne kadar stage'i recover ediyor
- **Cluster × Stage contingency matrix** — per-cluster stage dağılımı
- **Subject purity** — en çok temsilcisi olan hastanın küme içindeki payı

## Özet Bulgular

| Modalite | Chunks | Clusters | Noise | Stage ARI | Mean subject purity | Median cluster size |
|---|---:|---:|---:|---:|---:|---:|
| **BAS** (EEG+EOG) | 2464 | 19 | 91 | **0.0003** | **0.74** | 58 |
| RESP | 2464 | 31 | 386 | 0.034 | 0.37 | 60 |
| EKG | 2464 | 33 | 273 | 0.024 | 0.55 | 59 |
| EMG | 2464 | 21 | 319 | 0.038 | 0.48 | 80 |

## Ana Bulgu

**BAS modalitesi (EEG+EOG) uyku fazını değil, hasta kimliğini yakalıyor.**

- ARI(cluster, stage) = 0.0003 → HDBSCAN kümeleri sleep stage bilgisini sıfıra
  yakın bir şekilde recover ediyor. Rastgeleyle karşılaştırılabilir.
- Mean subject purity = 0.74 → bir küme ortalama olarak %74 tek bir hastadan
  oluşuyor. 20 hastalı bir kohortta uniform dağılım beklentisi 1/20 = 0.05.
  Yani BAS embeddings **hasta-specific**.

**RESP, EKG, EMG** için tablo farklı: subject purity 0.37-0.55 (daha global),
stage ARI 0.02-0.04 (hâlâ düşük ama BAS'tan yüksek).

## Neden Böyle?

Üç olası açıklama:

1. **Memorization (en muhtemel):** 17/20 hasta SleepFM pretrain'inde. Model bu
   hastaları görmüş; contrastive loss zaten "hastaları ayır" objektifiyle
   eğitilmiş — yani BAS embeddings pratik olarak "hasta imzası"ne yakınsamış.
   Bu hipotez temiz cohort'ta subject purity'nin 0.05'e düşmesiyle test edilebilir.

2. **BAS kanal seti dar:** MESA'da sadece 3 EEG + 2 EOG kanalı var, model
   BAS_CHANNELS=10 bekliyor. Sinyal kanalları sınırlı olduğunda embedding
   uzayı hasta-specific varyasyonlara odaklanmış olabilir.

3. **5-dk agregasyon uyku fazını dilute ediyor:** 300 sn içinde uyku fazı
   değişebiliyor (N2 ↔ N3 ↔ REM). Dominant stage yerine daha ince granülite
   (30 sn epoch) ile analiz farklı sonuç verebilir.

Clean cohort test'i #1'i doğrudan test edecek. Şu an bu bilimsel bir bulgu
değil, ama pipeline'ın bu hipotezi test edebildiğini gösteren bir method demo.

## Modalitelere Göre Confusion Matrix (Kısa)

`contingency_*.csv` dosyalarında tam matrix var. Bazı gözlemler:
- BAS'ta bazı kümeler %100 wake ya da %100 tek-hasta çıkıyor
- RESP kümelerinden bir kısmı N2/N3 baskın (uyku evrelerinden çok solunum kalıpları)
- EMG bazı kümeler pure Wake (kas hareketi yoğun uyanıklık dönemleri)

## Çıktılar

- `contingency_{MODALITY}.csv` — cluster × stage crosstab
- `purity_{MODALITY}.csv` — per-cluster subject dağılımı
- `SUMMARY.csv` — 4 modalite × 7 metrik özet

## Sıradaki Adım

1. **Temiz cohort ile tekrar çalıştır** — subject purity 0.05'e düşer mi test et
2. **HDBSCAN min_cluster_size sensitivity** — 15/30/50 dene
3. **Sleep-phase filtered analiz** — Wake chunk'larını exclude et, sadece
   uyku chunk'larında ARI hesapla
4. **Modalite-agnostik hasta similarity** — clinical-bridge entegrasyonu için
   temel bu subject purity metriği olabilir

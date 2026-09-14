"""Ablation demo — heatmap + tablo üretir."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from mock_data import generate_mock_embeddings
from config import DISEASE_NAMES, MODALITY_DIM_RANGES
from similarity_engine import build_reference_index
from pipeline import compute_ablation_scores

# 1. Mock data
embeddings, labels = generate_mock_embeddings(n_samples=500, seed=42)
print(f"Embeddings: {embeddings.shape}")

# 2. Her hastalık grubundan ablation
results = {}
for disease_id, disease_name in sorted(DISEASE_NAMES.items()):
    group_mask = labels == disease_id
    if not group_mask.any():
        continue
    query_idx = int(np.where(group_mask)[0][0])
    query_emb = embeddings[query_idx]
    ref_mask = np.ones(len(embeddings), dtype=bool)
    ref_mask[query_idx] = False
    ref_index = build_reference_index(embeddings[ref_mask], n_neighbors=10)
    scores = compute_ablation_scores(query_emb, ref_index, labels[ref_mask])
    results[disease_name] = scores
    print(f"  {disease_id:2d}. {disease_name:40s} | EEG={scores['eeg']:.4f}  ECG={scores['ecg']:.4f}  Resp={scores['resp']:.4f}  EMG={scores['emg']:.4f}")

# 3. DataFrame
df = pd.DataFrame(results).T
df.columns = [m.upper() for m in df.columns]
df["DOMINANT"] = df.idxmax(axis=1)

print("\n" + "="*70)
print("ABLATION SONUÇLARI (Hastalık x Modalite)")
print("="*70)
print(df.to_string())
print("\n--- Ortalama Skorlar ---")
print(df[["EEG","ECG","RESP","EMG"]].mean().to_string())

# 4. Heatmap
os.makedirs("outputs/figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(
    df[["EEG","ECG","RESP","EMG"]].astype(float),
    annot=True, fmt=".4f", cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Similarity Drop (ablation score)"},
    ax=ax,
)
ax.set_title("Modality Importance per Disease\n(Mask-Based Ablation — Similarity Drop)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Disease Group", fontsize=11)
ax.set_xlabel("Modality", fontsize=11)
plt.tight_layout()
fig.savefig("outputs/figures/ablation_heatmap.png", dpi=150, bbox_inches="tight")
print("\nHeatmap saved -> outputs/figures/ablation_heatmap.png")

# 5. Bar chart
mean_scores = df[["EEG","ECG","RESP","EMG"]].mean().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(6, 4))
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
mean_scores.plot(kind="bar", ax=ax2, color=colors[:len(mean_scores)], edgecolor="black")
ax2.set_title("Average Modality Importance (across all diseases)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Mean Ablation Score")
ax2.set_xlabel("")
ax2.tick_params(axis="x", rotation=0)
for i, (idx, val) in enumerate(mean_scores.items()):
    ax2.text(i, val + 0.001, f"{val:.4f}", ha="center", fontsize=9)
plt.tight_layout()
fig2.savefig("outputs/figures/ablation_bar_chart.png", dpi=150, bbox_inches="tight")
print("Bar chart saved -> outputs/figures/ablation_bar_chart.png")
print("\nDone!")

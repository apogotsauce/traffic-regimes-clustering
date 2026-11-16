"""
Прегенерира PCA и EDA графиките с английски надписи — за научната статия.
Изход: OutPut/figures_en/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE, "traffic_anonymized.csv")
OUT = os.path.join(BASE, "figures_en")
SAMPLE, K, SEED = 500_000, 4, 42

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False, nrows=SAMPLE)
df = df[df["vehicle_class"].between(0, 7)]

# ── EDA графики ───────────────────────────────────────────
hourly = df.groupby("hour").size()
plt.figure(figsize=(9, 3.6))
plt.bar(hourly.index, hourly.values, color="#4a7ba7")
plt.title("Vehicle count by hour of day")
plt.xlabel("Hour of day")
plt.ylabel("Number of vehicles")
plt.xticks(range(24))
plt.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/en_by_hour.png", dpi=160)
plt.close()

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily = df.groupby("day_of_week").size()
plt.figure(figsize=(7, 3.6))
plt.bar([days[i] for i in daily.index], daily.values, color="#4a8a5f")
plt.title("Vehicle count by day of week")
plt.xlabel("Day of week")
plt.ylabel("Number of vehicles")
plt.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/en_by_day.png", dpi=160)
plt.close()

sp = df["speed_kmh"].dropna()
plt.figure(figsize=(8, 3.6))
plt.hist(sp[(sp > 0) & (sp < 150)], bins=80, color="#e07b5a", edgecolor="black", lw=.3)
plt.title("Speed distribution")
plt.xlabel("Speed (km/h)")
plt.ylabel("Number of vehicles")
plt.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/en_speed_hist.png", dpi=160)
plt.close()

# ── Клъстериране + PCA ────────────────────────────────────
feats = ["speed_kmh", "hour", "day_of_week", "vehicle_class", "length_mm", "is_weekend"]
dm = df[feats].dropna()
X = StandardScaler().fit_transform(dm)
X2 = PCA(n_components=2, random_state=SEED).fit_transform(X)

labels = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X)
cols = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
plt.figure(figsize=(7.5, 5.5))
for i in range(K):
    m = labels == i
    plt.scatter(X2[m, 0], X2[m, 1], c=cols[i], alpha=.3, s=4,
                label=f"C{i}  (n = {m.sum():,})")
plt.legend(markerscale=4, fontsize=9)
plt.title(f"K-Means clusters (K={K}) in PCA projection")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.tight_layout()
plt.savefig(f"{OUT}/en_kmeans_pca.png", dpi=160)
plt.close()

idx = np.random.RandomState(SEED).choice(len(X), size=min(100_000, len(X)), replace=False)
dl = DBSCAN(eps=0.5, min_samples=50, n_jobs=-1).fit_predict(X[idx])
Xs = X2[idx]
n_cl = len(set(dl)) - (1 if -1 in dl else 0)
noise_pct = (dl == -1).sum() / len(dl) * 100

plt.figure(figsize=(7.5, 5.5))
uniq = sorted(set(dl))
cmap = plt.colormaps["tab10"].resampled(len(uniq))
for i, l in enumerate(uniq):
    m = dl == l
    plt.scatter(Xs[m, 0], Xs[m, 1],
                c=["lightgrey"] if l == -1 else [cmap(i)],
                alpha=.15 if l == -1 else .55,
                s=3 if l == -1 else 7,
                label="Noise" if l == -1 else f"C{l}  (n = {m.sum():,})")
plt.legend(markerscale=4, fontsize=8)
plt.title(f"DBSCAN clusters in PCA projection\n({n_cl} clusters, {noise_pct:.1f}% noise)")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.tight_layout()
plt.savefig(f"{OUT}/en_dbscan_pca.png", dpi=160)
plt.close()

print("Готово — графиките са в", OUT)
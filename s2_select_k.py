"""
СТЪПКА 2: Избор на брой клъстери K  (поправена версия)
=======================================================
Поправки спрямо първата версия:
  - inertia се изчислява ЯВНО върху целия набор, а не се чете
    от MiniBatchKMeans.inertia_ (която отразява само последния пакет)
  - добавено отчитане на време и брой обходени редове за контрол
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
ENCODING = "linear"
K_RANGE = range(2, 11)
N_SEEDS = 10
METRIC_SAMPLE = 100_000
BATCH = 10_000
CHUNK = 500_000

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def full_inertia(X, scaler, km, chunk=CHUNK):
    """Реална инерция върху ЦЕЛИЯ набор: сума от квадратите
    на разстоянията до най-близкия центроид."""
    total = 0.0
    seen = 0
    C = km.cluster_centers_
    for i in range(0, len(X), chunk):
        Xs = scaler.transform(X[i:i + chunk])
        # квадрат на разстоянието до всеки центроид
        d2 = ((Xs[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        total += float(d2.min(axis=1).sum())
        seen += len(Xs)
    return total, seen


def main():
    log("=" * 68)
    log(f"  СТЪПКА 2: Избор на K  ({ENCODING} кодиране)")
    log("=" * 68)

    X = np.load(os.path.join(OUT, f"X_{ENCODING}.npy"), mmap_mode="r")
    n, d = X.shape
    log(f"\nДанни: {n:,} записа, {d} признака")

    log("Стандартизация (partial_fit по части)...")
    scaler = StandardScaler()
    for i in range(0, n, CHUNK):
        scaler.partial_fit(X[i:i + CHUNK])
    log(f"  средни:    {np.round(scaler.mean_, 3)}")
    log(f"  ст. откл.: {np.round(scaler.scale_, 3)}")

    rng = np.random.RandomState(0)
    idx_metric = np.sort(rng.choice(n, METRIC_SAMPLE, replace=False))
    Xs_metric = scaler.transform(X[idx_metric])

    perm = rng.permutation(n)
    half = min(400_000, n // 2)
    Xs_a = scaler.transform(X[np.sort(perm[:half])])
    Xs_b = scaler.transform(X[np.sort(perm[half:2 * half])])
    log(f"  извадка за метрики:    {METRIC_SAMPLE:,}")
    log(f"  извадки за стабилност: 2 × {half:,}")

    rows = []
    for k in K_RANGE:
        t0 = time.time()
        log(f"\n--- K = {k} ---")
        iner, sil, dbi, chi, ari = [], [], [], [], []
        rows_seen = None

        for seed in range(N_SEEDS):
            km = MiniBatchKMeans(n_clusters=k, random_state=seed,
                                 batch_size=BATCH, n_init=3,
                                 max_iter=100, reassignment_ratio=0.01)
            for i in range(0, n, CHUNK):
                km.partial_fit(scaler.transform(X[i:i + CHUNK]))

            # реална инерция върху целия набор (само за първите 3 seed-а,
            # защото е скъпо изчисление)
            if seed < 3:
                fi, seen = full_inertia(X, scaler, km)
                iner.append(fi)
                rows_seen = seen

            lab = km.predict(Xs_metric)
            if len(np.unique(lab)) > 1:
                sil.append(silhouette_score(Xs_metric, lab,
                                            sample_size=20_000, random_state=seed))
                dbi.append(davies_bouldin_score(Xs_metric, lab))
                chi.append(calinski_harabasz_score(Xs_metric, lab))

            if seed < 5:
                ka = MiniBatchKMeans(n_clusters=k, random_state=seed,
                                     batch_size=BATCH, n_init=3).fit(Xs_a)
                kb = MiniBatchKMeans(n_clusters=k, random_state=seed + 100,
                                     batch_size=BATCH, n_init=3).fit(Xs_b)
                ari.append(adjusted_rand_score(ka.predict(Xs_metric),
                                               kb.predict(Xs_metric)))

        row = dict(
            k=k,
            inertia=np.mean(iner), inertia_sd=np.std(iner),
            silhouette=np.mean(sil), silhouette_sd=np.std(sil),
            davies_bouldin=np.mean(dbi), davies_bouldin_sd=np.std(dbi),
            calinski=np.mean(chi), calinski_sd=np.std(chi),
            ari_stability=np.mean(ari), ari_sd=np.std(ari),
        )
        rows.append(row)
        log(f"  обходени редове при inertia: {rows_seen:,}")
        log(f"  Inertia (целия набор): {row['inertia']:>16,.0f} ± {row['inertia_sd']:,.0f}")
        log(f"  Silhouette:            {row['silhouette']:>16.4f} ± {row['silhouette_sd']:.4f}")
        log(f"  Davies-Bouldin:        {row['davies_bouldin']:>16.4f} ± {row['davies_bouldin_sd']:.4f}")
        log(f"  Calinski-Harabasz:     {row['calinski']:>16,.0f} ± {row['calinski_sd']:,.0f}")
        log(f"  ARI стабилност:        {row['ari_stability']:>16.4f} ± {row['ari_sd']:.4f}")
        log(f"  време: {time.time() - t0:.0f} сек")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"k_selection_{ENCODING}.csv"), index=False)

    log("\n" + "=" * 68)
    log("  ОБОБЩЕНИЕ")
    log("=" * 68)
    log(f"{'K':>3}{'Inertia':>18}{'Silhouette':>13}{'DB':>10}{'CH':>12}{'ARI':>9}")
    for _, r in df.iterrows():
        log(f"{int(r['k']):>3}{r['inertia']:>18,.0f}{r['silhouette']:>13.4f}"
            f"{r['davies_bouldin']:>10.3f}{r['calinski']:>12,.0f}{r['ari_stability']:>9.3f}")

    log(f"\nПрепоръки:")
    log(f"  max Silhouette      -> K = {int(df.loc[df['silhouette'].idxmax(), 'k'])}")
    log(f"  min Davies-Bouldin  -> K = {int(df.loc[df['davies_bouldin'].idxmin(), 'k'])}")
    log(f"  max Calinski-Har.   -> K = {int(df.loc[df['calinski'].idxmax(), 'k'])}")
    log(f"  max стабилност(ARI) -> K = {int(df.loc[df['ari_stability'].idxmax(), 'k'])}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ks = df["k"].values
    panels = [
        (0, 0, "inertia", "inertia_sd", "Inertia", "(a) Elbow method", "#1f4e79"),
        (0, 1, "silhouette", "silhouette_sd", "Silhouette score", "(b) Silhouette (higher is better)", "#2e75b6"),
        (1, 0, "davies_bouldin", "davies_bouldin_sd", "Davies-Bouldin index", "(c) Davies-Bouldin (lower is better)", "#c8552d"),
        (1, 1, "ari_stability", "ari_sd", "Adjusted Rand Index", "(d) Stability across data halves", "#4daf4a"),
    ]
    for r, c, col, sd, ylab, title, color in panels:
        ax[r, c].errorbar(ks, df[col], yerr=df[sd], marker="o", color=color, capsize=3)
        ax[r, c].set_xlabel("Number of clusters (K)")
        ax[r, c].set_ylabel(ylab)
        ax[r, c].set_title(title)
        ax[r, c].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"fig_k_selection_{ENCODING}.png"), dpi=600)
    plt.close()

    with open(os.path.join(OUT, f"k_report_{ENCODING}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\nГотово. Резултатите са в {OUT}")


if __name__ == "__main__":
    main()
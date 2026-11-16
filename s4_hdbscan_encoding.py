"""
СТЪПКА 4: HDBSCAN при различни представяния на признаците
===========================================================
При линейно кодиране HDBSCAN се проваля — разцепва данните по
дискретния признак day_of_week, вместо да намери смислени режими.

Тук се проверяват три конфигурации върху една и съща извадка:
  A) linear    — 6 признака както в основния анализ
  B) cyclic    — час и ден като sin/cos двойки
  C) continuous— само непрекъснати признаци (без дискретните)

Изход: out/hdbscan_encoding.csv, out/hdbscan_encoding_report.txt,
       out/profile_hdbscan_{config}.csv
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
import hdbscan

# ─── КОНФИГУРАЦИЯ ─────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
SAMPLE = 300_000
SEED = 42
MIN_CLUSTER = 15_000
MIN_SAMPLES = 100
# ──────────────────────────────────────────────────────────

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def evaluate(X, labels):
    mask = labels != -1
    n_cl = len(set(labels[mask]))
    noise = 100 * (~mask).sum() / len(labels)
    if n_cl < 2:
        return n_cl, noise, np.nan, np.nan, np.nan
    return (n_cl, noise,
            silhouette_score(X[mask], labels[mask],
                             sample_size=min(20_000, int(mask.sum())),
                             random_state=SEED),
            davies_bouldin_score(X[mask], labels[mask]),
            calinski_harabasz_score(X[mask], labels[mask]))


def profile(meta, labels, tag):
    cols = ["hour", "day_of_week", "vehicle_class", "speed_kmh", "length_mm"]
    df = pd.DataFrame(meta[:, :5], columns=cols)
    df["cluster"] = labels
    keep = df[df["cluster"] != -1]
    if len(keep) == 0:
        return pd.DataFrame()
    p = keep.groupby("cluster")[cols].mean().round(2)
    p["n"] = keep.groupby("cluster").size()
    p["share_%"] = (100 * p["n"] / len(df)).round(2)
    p = p.sort_values("n", ascending=False)
    p.to_csv(os.path.join(OUT, f"profile_hdbscan_{tag}.csv"))
    return p


def day_purity(meta, labels):
    """Дял на клъстерите, които реално са 'един ден от седмицата'.
    Висока стойност = алгоритъмът дроби по дискретния признак."""
    dow = meta[:, 1]
    pure = 0
    total = 0
    for c in sorted(set(labels)):
        if c == -1:
            continue
        total += 1
        vals = dow[labels == c]
        # ако над 95% от точките са един и същи ден -> клъстерът е "ден"
        if len(vals) and (np.bincount(vals.astype(int)).max() / len(vals)) > 0.95:
            pure += 1
    return 100 * pure / total if total else 0.0


def main():
    log("=" * 72)
    log("  СТЪПКА 4: HDBSCAN при различни представяния на признаците")
    log("=" * 72)

    X_lin = np.load(os.path.join(OUT, "X_linear.npy"), mmap_mode="r")
    X_cyc = np.load(os.path.join(OUT, "X_cyclic.npy"), mmap_mode="r")
    M = np.load(os.path.join(OUT, "meta.npy"), mmap_mode="r")
    n = len(X_lin)

    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(n, SAMPLE, replace=False))
    meta = np.asarray(M[idx])
    log(f"\nОбща извадка: {SAMPLE:,} записа (seed={SEED})")
    log(f"Параметри: min_cluster_size={MIN_CLUSTER:,}, min_samples={MIN_SAMPLES}")

    # ── трите конфигурации ──
    # linear: speed, hour, dow, class, length, is_weekend
    A = np.asarray(X_lin[idx])
    # cyclic: speed, hour_sin, hour_cos, dow_sin, dow_cos, class, length
    B = np.asarray(X_cyc[idx])
    # continuous: speed, hour_sin, hour_cos, length  (без дискретните dow и class)
    C = B[:, [0, 1, 2, 6]]

    configs = [
        ("A_linear", A,
         ["speed", "hour", "day_of_week", "vehicle_class", "length", "is_weekend"],
         "Линейно кодиране — както в основния анализ"),
        ("B_cyclic", B,
         ["speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "vehicle_class", "length"],
         "Циклично кодиране — час и ден като sin/cos"),
        ("C_continuous", C,
         ["speed", "hour_sin", "hour_cos", "length"],
         "Само непрекъснати признаци — без дискретните"),
    ]

    rows = []
    for tag, Xraw, names, desc in configs:
        log("\n" + "─" * 72)
        log(f"{tag}:  {desc}")
        log(f"   признаци ({len(names)}): {names}")

        Xs = StandardScaler().fit_transform(Xraw).astype(np.float64)
        t0 = time.time()
        lab = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER,
                              min_samples=MIN_SAMPLES,
                              core_dist_n_jobs=-1).fit_predict(Xs)
        dt = time.time() - t0

        n_cl, noise, sil, dbi, chi = evaluate(Xs, lab)
        purity = day_purity(meta, lab)

        log(f"   време: {dt:.0f} сек")
        log(f"   клъстери: {n_cl},  шум: {noise:.1f}%")
        log(f"   Silhouette: {sil:.4f},  Davies-Bouldin: {dbi:.4f},  CH: {chi:,.0f}")
        log(f"   клъстери, отговарящи на един ден: {purity:.0f}%"
            f"   {'<- ДРОБЕНЕ ПО ДИСКРЕТЕН ПРИЗНАК' if purity > 50 else ''}")

        p = profile(meta, lab, tag)
        if len(p):
            log("\n   Профил (подредени по големина):")
            log(p.head(10).to_string())

        np.save(os.path.join(OUT, f"labels_hdbscan_{tag}.npy"), lab)
        rows.append(dict(config=tag, description=desc, n_features=len(names),
                         n_clusters=n_cl, noise_pct=round(noise, 2),
                         silhouette=round(sil, 4) if sil == sil else None,
                         davies_bouldin=round(dbi, 4) if dbi == dbi else None,
                         calinski=round(chi) if chi == chi else None,
                         day_split_pct=round(purity), time_s=round(dt)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "hdbscan_encoding.csv"), index=False)

    log("\n" + "=" * 72)
    log("  СРАВНИТЕЛНА ТАБЛИЦА")
    log("=" * 72)
    log(f"{'Конфигурация':<16}{'призн.':>8}{'клъст.':>8}{'шум %':>8}"
        f"{'Silh.':>9}{'DB':>9}{'дроб. %':>10}{'сек':>7}")
    for _, r in df.iterrows():
        sil = f"{r['silhouette']:.4f}" if r['silhouette'] is not None else "  n/a"
        dbi = f"{r['davies_bouldin']:.3f}" if r['davies_bouldin'] is not None else "  n/a"
        log(f"{r['config']:<16}{r['n_features']:>8}{r['n_clusters']:>8}"
            f"{r['noise_pct']:>8.1f}{sil:>9}{dbi:>9}"
            f"{r['day_split_pct']:>10}{r['time_s']:>7}")

    log("\nЛегенда: 'дроб. %' = дял на клъстерите, в които над 95% от точките")
    log("са от един и същи ден от седмицата. Висока стойност означава, че")
    log("алгоритъмът е разцепил данните по дискретния признак вместо да")
    log("открие смислени трафични режими.")

    with open(os.path.join(OUT, "hdbscan_encoding_report.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\nГотово. Резултатите са в {OUT}")


if __name__ == "__main__":
    main()
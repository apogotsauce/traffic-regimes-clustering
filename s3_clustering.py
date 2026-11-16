"""
СТЪПКА 3: Клъстериране с четири алгоритъма
===========================================
  1. MiniBatchKMeans   — партиционен, върху ЦЕЛИЯ набор
  2. HDBSCAN           — йерархичен, базиран на гъстота
  3. Gaussian Mixture  — вероятностен модел
  4. Agglomerative     — йерархичен, свързващ

Всеки алгоритъм се оценява с три метрики върху обща извадка,
за да са резултатите директно съпоставими.

Изход: out/clustering_results.csv, out/labels_*.npy,
       out/profiles_*.csv, out/clustering_report.txt
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# ─── КОНФИГУРАЦИЯ ─────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
ENCODING = "linear"
K = 4                      # от стъпка 2 (най-висока стабилност)
SEED = 42
CHUNK = 500_000

EVAL_SAMPLE = 100_000      # обща извадка за метриките
HDBSCAN_SAMPLE = 300_000   # HDBSCAN е скъп по памет
AGGLO_SAMPLE = 25_000      # Agglomerative е O(n²) по памет
# ──────────────────────────────────────────────────────────

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def evaluate(X_eval, labels, name):
    """Трите вътрешни метрики, изчислени само за неshумовите точки."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())

    if n_clusters < 2:
        return dict(algorithm=name, n_clusters=n_clusters,
                    noise_pct=100 * n_noise / len(labels),
                    silhouette=np.nan, davies_bouldin=np.nan, calinski=np.nan)

    return dict(
        algorithm=name,
        n_clusters=n_clusters,
        noise_pct=100 * n_noise / len(labels),
        silhouette=silhouette_score(X_eval[mask], labels[mask],
                                    sample_size=min(20_000, mask.sum()),
                                    random_state=SEED),
        davies_bouldin=davies_bouldin_score(X_eval[mask], labels[mask]),
        calinski=calinski_harabasz_score(X_eval[mask], labels[mask]),
    )


def profile(meta, labels, name):
    """Средни стойности на характеристиките по клъстер в оригинални единици."""
    cols = ["hour", "day_of_week", "vehicle_class", "speed_kmh", "length_mm"]
    df = pd.DataFrame(meta[:, :5], columns=cols)
    df["cluster"] = labels
    p = df[df["cluster"] != -1].groupby("cluster")[cols].mean().round(2)
    p["n"] = df[df["cluster"] != -1].groupby("cluster").size()
    p["share_%"] = (100 * p["n"] / len(df)).round(2)
    p.to_csv(os.path.join(OUT, f"profile_{name}.csv"))
    return p


def main():
    log("=" * 70)
    log(f"  СТЪПКА 3: Клъстериране с четири алгоритъма  ({ENCODING})")
    log("=" * 70)

    X = np.load(os.path.join(OUT, f"X_{ENCODING}.npy"), mmap_mode="r")
    M = np.load(os.path.join(OUT, "meta.npy"), mmap_mode="r")
    n = len(X)
    log(f"\nДанни: {n:,} записа, {X.shape[1]} признака")

    log("Стандартизация...")
    scaler = StandardScaler()
    for i in range(0, n, CHUNK):
        scaler.partial_fit(X[i:i + CHUNK])

    rng = np.random.RandomState(SEED)
    idx_eval = np.sort(rng.choice(n, EVAL_SAMPLE, replace=False))
    X_eval = scaler.transform(X[idx_eval])
    M_eval = np.asarray(M[idx_eval])
    log(f"Обща извадка за оценка: {EVAL_SAMPLE:,} записа")

    results = []

    # ═══════════ 1. MiniBatchKMeans (целия набор) ═══════════
    log("\n" + "─" * 70)
    log("1) MiniBatchKMeans — върху ЦЕЛИЯ набор")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, random_state=SEED,
                         batch_size=10_000, n_init=5, max_iter=100)
    for i in range(0, n, CHUNK):
        km.partial_fit(scaler.transform(X[i:i + CHUNK]))
    lab_km_eval = km.predict(X_eval)
    log(f"   обучен за {time.time() - t0:.0f} сек")

    r = evaluate(X_eval, lab_km_eval, "MiniBatchKMeans")
    r["train_n"] = n
    r["time_s"] = round(time.time() - t0)
    results.append(r)
    log(f"   клъстери: {r['n_clusters']}, Silhouette: {r['silhouette']:.4f}, "
        f"DB: {r['davies_bouldin']:.4f}, CH: {r['calinski']:,.0f}")

    # етикети за целия набор (на части, за да не претоварим паметта)
    log("   предсказване за целия набор...")
    lab_full = np.empty(n, dtype=np.int8)
    for i in range(0, n, CHUNK):
        lab_full[i:i + CHUNK] = km.predict(scaler.transform(X[i:i + CHUNK]))
    np.save(os.path.join(OUT, f"labels_kmeans_{ENCODING}.npy"), lab_full)

    log("\n   Профил (целия набор):")
    p_full = profile(np.asarray(M), lab_full, f"kmeans_full_{ENCODING}")
    log(p_full.to_string())
    del lab_full

    # ═══════════ 2. HDBSCAN ═══════════
    if HAS_HDBSCAN:
        log("\n" + "─" * 70)
        log(f"2) HDBSCAN — върху извадка от {HDBSCAN_SAMPLE:,}")
        t0 = time.time()
        idx_h = np.sort(rng.choice(n, HDBSCAN_SAMPLE, replace=False))
        Xh = scaler.transform(X[idx_h])
        hdb = hdbscan.HDBSCAN(min_cluster_size=15000, min_samples=100,
                              core_dist_n_jobs=-1)
        lab_h = hdb.fit_predict(Xh)
        log(f"   обучен за {time.time() - t0:.0f} сек")

        r = evaluate(Xh, lab_h, "HDBSCAN")
        r["train_n"] = HDBSCAN_SAMPLE
        r["time_s"] = round(time.time() - t0)
        results.append(r)
        log(f"   клъстери: {r['n_clusters']}, шум: {r['noise_pct']:.1f}%, "
            f"Silhouette: {r['silhouette']:.4f}, DB: {r['davies_bouldin']:.4f}")

        np.save(os.path.join(OUT, f"labels_hdbscan_{ENCODING}.npy"), lab_h)
        log("\n   Профил:")
        log(profile(np.asarray(M[idx_h]), lab_h, f"hdbscan_{ENCODING}").to_string())
        del Xh
    else:
        log("\n[!] hdbscan не е инсталиран — пропуснат")

    # ═══════════ 3. Gaussian Mixture ═══════════
    log("\n" + "─" * 70)
    log(f"3) Gaussian Mixture Model — K={K}")
    t0 = time.time()
    idx_g = np.sort(rng.choice(n, 500_000, replace=False))
    Xg = scaler.transform(X[idx_g])
    gmm = GaussianMixture(n_components=K, random_state=SEED,
                          covariance_type="full", n_init=3, max_iter=200)
    gmm.fit(Xg)
    lab_g_eval = gmm.predict(X_eval)
    log(f"   обучен за {time.time() - t0:.0f} сек")
    log(f"   BIC: {gmm.bic(Xg):,.0f}   AIC: {gmm.aic(Xg):,.0f}")

    r = evaluate(X_eval, lab_g_eval, "GaussianMixture")
    r["train_n"] = len(idx_g)
    r["time_s"] = round(time.time() - t0)
    r["bic"] = round(gmm.bic(Xg))
    results.append(r)
    log(f"   клъстери: {r['n_clusters']}, Silhouette: {r['silhouette']:.4f}, "
        f"DB: {r['davies_bouldin']:.4f}, CH: {r['calinski']:,.0f}")

    np.save(os.path.join(OUT, f"labels_gmm_{ENCODING}.npy"), lab_g_eval)
    log("\n   Профил:")
    log(profile(M_eval, lab_g_eval, f"gmm_{ENCODING}").to_string())
    del Xg

    # ═══════════ 4. Agglomerative ═══════════
    log("\n" + "─" * 70)
    log(f"4) Agglomerative Clustering — върху извадка от {AGGLO_SAMPLE:,}")
    t0 = time.time()
    idx_a = np.sort(rng.choice(n, AGGLO_SAMPLE, replace=False))
    Xa = scaler.transform(X[idx_a]).astype(np.float32)
    agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
    lab_a = agg.fit_predict(Xa)
    log(f"   обучен за {time.time() - t0:.0f} сек")

    r = evaluate(Xa, lab_a, "Agglomerative")
    r["train_n"] = AGGLO_SAMPLE
    r["time_s"] = round(time.time() - t0)
    results.append(r)
    log(f"   клъстери: {r['n_clusters']}, Silhouette: {r['silhouette']:.4f}, "
        f"DB: {r['davies_bouldin']:.4f}, CH: {r['calinski']:,.0f}")

    np.save(os.path.join(OUT, f"labels_agglo_{ENCODING}.npy"), lab_a)
    log("\n   Профил:")
    log(profile(np.asarray(M[idx_a]), lab_a, f"agglo_{ENCODING}").to_string())

    # ═══════════ ОБОБЩЕНИЕ ═══════════
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, f"clustering_results_{ENCODING}.csv"), index=False)

    log("\n" + "=" * 70)
    log("  СРАВНИТЕЛНА ТАБЛИЦА")
    log("=" * 70)
    log(f"{'Алгоритъм':<20}{'клъстери':>10}{'шум %':>9}{'Silh.':>9}"
        f"{'DB':>9}{'CH':>11}{'n_train':>12}{'сек':>7}")
    for _, r in df.iterrows():
        log(f"{r['algorithm']:<20}{r['n_clusters']:>10}{r['noise_pct']:>9.1f}"
            f"{r['silhouette']:>9.4f}{r['davies_bouldin']:>9.3f}"
            f"{r['calinski']:>11,.0f}{r['train_n']:>12,}{r['time_s']:>7}")

    with open(os.path.join(OUT, f"clustering_report_{ENCODING}.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\nГотово. Резултатите са в {OUT}")


if __name__ == "__main__":
    main()
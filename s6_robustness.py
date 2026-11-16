"""
СТЪПКА 6: Устойчивост на решението и важност на признаците
===========================================================
Отговаря на три въпроса, които всеки рецензент задава:

  1. Случайни ли са резултатите?        -> 10 различни seed-а, двойки ARI
  2. Зависят ли от конкретната извадка? -> 3 независими извадки, двойки ARI
  3. Кой признак носи разделянето?      -> ablation: махаме по един признак
                                           и гледаме как се променя решението

При ablation се отчитат две неща:
  - метриките в намаленото пространство (Silhouette, DB, CH)
  - ARI спрямо пълния модел върху една и съща оценъчна извадка

Второто е по-важното: метриките в пространства с различна размерност не са
пряко съпоставими, докато ARI показва доколко се е променило самото
разделяне на точките. Нисък ARI = признакът е носел разделянето.

Изход: out/robustness_seeds.csv, out/robustness_subsamples.csv,
       out/feature_ablation.csv, out/fig_ablation.png,
       out/robustness_report.txt
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             adjusted_mutual_info_score)

# ─── КОНФИГУРАЦИЯ ─────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
ENCODING = "linear"
K = 4
SEED = 42
CHUNK = 500_000

FIT_SAMPLE = 500_000     # върху колко се обучава всеки модел
EVAL_SAMPLE = 100_000    # обща извадка, върху която се мерят всички
SIL_SAMPLE = 20_000      # подизвадка за Silhouette
N_SEEDS = 10             # за проверката на устойчивост по seed
N_SUBSAMPLES = 3         # независими извадки
# ──────────────────────────────────────────────────────────

FEATURES = ["speed_kmh", "hour", "day_of_week",
            "vehicle_class", "length_mm", "is_weekend"]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def fit_kmeans(X, k=K, seed=SEED):
    return MiniBatchKMeans(n_clusters=k, random_state=seed,
                           batch_size=10_000, n_init=5,
                           max_iter=100).fit(X)


def metrics(X, labels):
    if len(set(labels)) < 2:
        return np.nan, np.nan, np.nan
    return (silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, len(X)),
                             random_state=SEED),
            davies_bouldin_score(X, labels),
            calinski_harabasz_score(X, labels))


def pairwise(labels_list):
    """Средно и ст. откл. на ARI и AMI по всички двойки."""
    aris, amis = [], []
    for a, b in itertools.combinations(range(len(labels_list)), 2):
        aris.append(adjusted_rand_score(labels_list[a], labels_list[b]))
        amis.append(adjusted_mutual_info_score(labels_list[a], labels_list[b]))
    return np.mean(aris), np.std(aris), np.mean(amis), np.std(amis)


def main():
    t_start = time.time()
    log("=" * 72)
    log(f"  СТЪПКА 6: Устойчивост и важност на признаците  ({ENCODING})")
    log("=" * 72)

    X = np.load(os.path.join(OUT, f"X_{ENCODING}.npy"), mmap_mode="r")
    n = len(X)
    log(f"\nДанни: {n:,} записа, {X.shape[1]} признака")
    log(f"Признаци: {FEATURES}")

    log("\nСтандартизация (partial_fit по части)...")
    scaler = StandardScaler()
    for i in range(0, n, CHUNK):
        scaler.partial_fit(X[i:i + CHUNK])

    rng = np.random.RandomState(SEED)
    idx_eval = np.sort(rng.choice(n, EVAL_SAMPLE, replace=False))
    X_eval = scaler.transform(X[idx_eval])
    idx_fit = np.sort(rng.choice(n, FIT_SAMPLE, replace=False))
    X_fit = scaler.transform(X[idx_fit])
    log(f"Извадка за обучение: {FIT_SAMPLE:,}")
    log(f"Обща извадка за оценка: {EVAL_SAMPLE:,}")

    # ═══════════ 1. Устойчивост по seed ═══════════
    log("\n" + "─" * 72)
    log(f"1) Устойчивост по случайна инициализация — {N_SEEDS} seed-а, K={K}")
    t0 = time.time()
    labs_seed, rows_seed = [], []
    for s in range(N_SEEDS):
        km = fit_kmeans(X_fit, seed=s)
        lab = km.predict(X_eval)
        labs_seed.append(lab)
        sil, dbi, chi = metrics(X_eval, lab)
        rows_seed.append(dict(seed=s, silhouette=round(sil, 4),
                              davies_bouldin=round(dbi, 4),
                              calinski=round(chi),
                              sizes="/".join(str(c) for c in
                                             np.bincount(lab, minlength=K))))
        log(f"   seed {s:>2}: Sil {sil:.4f}  DB {dbi:.4f}  CH {chi:,.0f}")

    ari_m, ari_s, ami_m, ami_s = pairwise(labs_seed)
    df_seed = pd.DataFrame(rows_seed)
    df_seed.to_csv(os.path.join(OUT, "robustness_seeds.csv"), index=False)
    log(f"\n   Silhouette:  {df_seed['silhouette'].mean():.4f} "
        f"± {df_seed['silhouette'].std():.4f}")
    log(f"   Davies-Bouldin: {df_seed['davies_bouldin'].mean():.4f} "
        f"± {df_seed['davies_bouldin'].std():.4f}")
    log(f"   ARI между двойки решения: {ari_m:.4f} ± {ari_s:.4f}")
    log(f"   AMI между двойки решения: {ami_m:.4f} ± {ami_s:.4f}")
    log(f"   време: {time.time() - t0:.0f} сек")

    # ═══════════ 2. Устойчивост по извадка ═══════════
    log("\n" + "─" * 72)
    log(f"2) Устойчивост по независими извадки — {N_SUBSAMPLES} × {FIT_SAMPLE:,}")
    t0 = time.time()
    perm = rng.permutation(n)
    labs_sub, rows_sub = [], []
    for j in range(N_SUBSAMPLES):
        part = np.sort(perm[j * FIT_SAMPLE:(j + 1) * FIT_SAMPLE])
        Xj = scaler.transform(X[part])
        km = fit_kmeans(Xj, seed=SEED)
        lab = km.predict(X_eval)
        labs_sub.append(lab)
        sil, dbi, chi = metrics(X_eval, lab)
        rows_sub.append(dict(subsample=j + 1, n=len(part),
                             silhouette=round(sil, 4),
                             davies_bouldin=round(dbi, 4),
                             calinski=round(chi)))
        log(f"   извадка {j + 1}: Sil {sil:.4f}  DB {dbi:.4f}  CH {chi:,.0f}")
        del Xj

    ari_m2, ari_s2, ami_m2, ami_s2 = pairwise(labs_sub)
    pd.DataFrame(rows_sub).to_csv(
        os.path.join(OUT, "robustness_subsamples.csv"), index=False)
    log(f"\n   ARI между двойки извадки: {ari_m2:.4f} ± {ari_s2:.4f}")
    log(f"   AMI между двойки извадки: {ami_m2:.4f} ± {ami_s2:.4f}")
    log(f"   време: {time.time() - t0:.0f} сек")

    # ═══════════ 3. Ablation по признаци ═══════════
    log("\n" + "─" * 72)
    log("3) Важност на признаците — ablation (махане на по един признак)")
    t0 = time.time()

    km_full = fit_kmeans(X_fit, seed=SEED)
    lab_full = km_full.predict(X_eval)
    sil_f, db_f, ch_f = metrics(X_eval, lab_full)
    log(f"\n   Базов модел (всички {len(FEATURES)} признака):")
    log(f"      Silhouette {sil_f:.4f}   DB {db_f:.4f}   CH {ch_f:,.0f}")

    rows_ab = [dict(removed="(няма — базов модел)", n_features=len(FEATURES),
                    silhouette=round(sil_f, 4), davies_bouldin=round(db_f, 4),
                    calinski=round(ch_f), ari_vs_full=1.0, ami_vs_full=1.0)]

    for j, name in enumerate(FEATURES):
        keep = [c for c in range(len(FEATURES)) if c != j]
        km_j = fit_kmeans(X_fit[:, keep], seed=SEED)
        lab_j = km_j.predict(X_eval[:, keep])
        sil, dbi, chi = metrics(X_eval[:, keep], lab_j)
        ari = adjusted_rand_score(lab_full, lab_j)
        ami = adjusted_mutual_info_score(lab_full, lab_j)
        rows_ab.append(dict(removed=name, n_features=len(keep),
                            silhouette=round(sil, 4),
                            davies_bouldin=round(dbi, 4),
                            calinski=round(chi),
                            ari_vs_full=round(ari, 4),
                            ami_vs_full=round(ami, 4)))
        log(f"   без {name:<14} Sil {sil:>7.4f}  DB {dbi:>6.3f}  "
            f"CH {chi:>10,.0f}  ARI спрямо пълния {ari:.4f}")

    df_ab = pd.DataFrame(rows_ab)
    df_ab.to_csv(os.path.join(OUT, "feature_ablation.csv"), index=False)
    log(f"   време: {time.time() - t0:.0f} сек")

    order = df_ab[df_ab["removed"] != "(няма — базов модел)"] \
        .sort_values("ari_vs_full")
    log("\n   Подредба по важност (нисък ARI = признакът носи разделянето):")
    for rank, (_, r) in enumerate(order.iterrows(), 1):
        log(f"      {rank}. {r['removed']:<14} ARI {r['ari_vs_full']:.4f}")

    # ═══════════ Фигура ═══════════
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    names = order["removed"].tolist()
    ax[0].barh(names, order["ari_vs_full"], color="#2e75b6")
    ax[0].axvline(1.0, color="#c8552d", ls="--", lw=1.2,
                  label="identical to full model")
    ax[0].set_xlabel("Adjusted Rand Index vs. full model")
    ax[0].set_title("(a) Feature ablation — lower means more important")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="x")

    ax[1].errorbar(df_seed["seed"], df_seed["silhouette"],
                   marker="o", color="#4daf4a", lw=1.5)
    ax[1].axhline(df_seed["silhouette"].mean(), color="#1f4e79", ls="--", lw=1.2,
                  label=f"mean = {df_seed['silhouette'].mean():.4f}")
    ax[1].set_xlabel("Random seed")
    ax[1].set_ylabel("Silhouette score")
    ax[1].set_title(f"(b) Stability across {N_SEEDS} random initialisations")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_ablation.png"), dpi=600)
    plt.close()

    # ═══════════ ОБОБЩЕНИЕ ═══════════
    log("\n" + "=" * 72)
    log("  ОБОБЩЕНИЕ")
    log("=" * 72)
    log(f"Устойчивост по seed:    ARI {ari_m:.4f} ± {ari_s:.4f}")
    log(f"Устойчивост по извадка: ARI {ari_m2:.4f} ± {ari_s2:.4f}")
    log(f"Най-важен признак:      {order.iloc[0]['removed']} "
        f"(ARI пада до {order.iloc[0]['ari_vs_full']:.4f})")
    log(f"Най-малко важен:        {order.iloc[-1]['removed']} "
        f"(ARI {order.iloc[-1]['ari_vs_full']:.4f})")
    log(f"\nОбщо време: {time.time() - t_start:.0f} сек")

    with open(os.path.join(OUT, "robustness_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Готово. Резултатите са в {OUT}")


if __name__ == "__main__":
    main()

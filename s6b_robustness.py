"""
СТЪПКА 6b: Устойчивост и важност на признаците — поправена методика
====================================================================
Първата версия (s6) показа ARI 0.69 ± 0.21 между решения с различен
seed. Причината не е в данните, а в алгоритъма: MiniBatchKMeans с малък
n_init попада в различни локални минимуми. При това ниво на собствен шум
ablation резултатите не могат да се тълкуват — разликата между признак,
който носи структурата, и такъв, който не носи, потъва в шума.

Поправки:
  1. Пълен KMeans (k-means++, n_init=N_INIT) вместо MiniBatchKMeans.
  2. Явно измерен ПРАГ НА ШУМА: ARI между два пълни модела, които се
     различават само по seed. Всяка ablation стойност се сравнява с него.
  3. Ablation, усреднен по няколко seed-а, а не по един.
  4. Директно сравнение MiniBatch срещу пълен KMeans — документира
     защо методиката е сменена.

Изход: out/s6b_minibatch_vs_full.csv, out/s6b_seeds.csv,
       out/s6b_subsamples.csv, out/s6b_ablation.csv,
       out/fig_ablation_v2.png, out/s6b_report.txt
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
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

FIT_SAMPLE = 300_000     # върху колко се обучава един модел
EVAL_SAMPLE = 100_000    # обща извадка, върху която се мери всичко
SIL_SAMPLE = 20_000
N_INIT = 10              # рестарти вътре в KMeans
N_SEEDS = 10             # seed-ове за частите 1 и 2
N_SEEDS_ABL = 3          # seed-ове за ablation (усредняване)
N_SUBSAMPLES = 3
# ──────────────────────────────────────────────────────────

FEATURES = ["speed_kmh", "hour", "day_of_week",
            "vehicle_class", "length_mm", "is_weekend"]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def fit_full(X, seed):
    return KMeans(n_clusters=K, random_state=seed, n_init=N_INIT,
                  max_iter=300).fit(X)


def fit_mb(X, seed):
    return MiniBatchKMeans(n_clusters=K, random_state=seed,
                           batch_size=10_000, n_init=5, max_iter=100).fit(X)


def metrics(X, labels):
    if len(set(labels)) < 2:
        return np.nan, np.nan, np.nan
    return (silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, len(X)),
                             random_state=SEED),
            davies_bouldin_score(X, labels),
            calinski_harabasz_score(X, labels))


def pairwise(labels_list):
    aris, amis = [], []
    for a, b in itertools.combinations(range(len(labels_list)), 2):
        aris.append(adjusted_rand_score(labels_list[a], labels_list[b]))
        amis.append(adjusted_mutual_info_score(labels_list[a], labels_list[b]))
    return float(np.mean(aris)), float(np.std(aris)), \
        float(np.mean(amis)), float(np.std(amis))


def main():
    t_start = time.time()
    log("=" * 74)
    log(f"  СТЪПКА 6b: Устойчивост и важност на признаците  ({ENCODING})")
    log("=" * 74)

    X = np.load(os.path.join(OUT, f"X_{ENCODING}.npy"), mmap_mode="r")
    n = len(X)
    log(f"\nДанни: {n:,} записа, {X.shape[1]} признака")
    log(f"Признаци: {FEATURES}")
    log(f"Модел: KMeans (k-means++), K={K}, n_init={N_INIT}")

    log("\nСтандартизация (partial_fit по части)...")
    scaler = StandardScaler()
    for i in range(0, n, CHUNK):
        scaler.partial_fit(X[i:i + CHUNK])

    rng = np.random.RandomState(SEED)
    idx_eval = np.sort(rng.choice(n, EVAL_SAMPLE, replace=False))
    X_eval = scaler.transform(X[idx_eval]).astype(np.float64)
    idx_fit = np.sort(rng.choice(n, FIT_SAMPLE, replace=False))
    X_fit = scaler.transform(X[idx_fit]).astype(np.float64)
    log(f"Извадка за обучение: {FIT_SAMPLE:,}")
    log(f"Обща извадка за оценка: {EVAL_SAMPLE:,}")

    # ═══════ 0. MiniBatch срещу пълен KMeans ═══════
    log("\n" + "─" * 74)
    log("0) Защо е сменен алгоритъмът: MiniBatchKMeans срещу пълен KMeans")
    t0 = time.time()
    labs_mb, labs_fu, rows_cmp = [], [], []
    for s in range(N_SEEDS):
        lab_mb = fit_mb(X_fit, s).predict(X_eval)
        lab_fu = fit_full(X_fit, s).predict(X_eval)
        labs_mb.append(lab_mb)
        labs_fu.append(lab_fu)
        s_mb = metrics(X_eval, lab_mb)
        s_fu = metrics(X_eval, lab_fu)
        rows_cmp.append(dict(seed=s,
                             sil_minibatch=round(s_mb[0], 4),
                             sil_full=round(s_fu[0], 4),
                             db_minibatch=round(s_mb[1], 4),
                             db_full=round(s_fu[1], 4)))
        log(f"   seed {s:>2}:  MiniBatch Sil {s_mb[0]:.4f} DB {s_mb[1]:.4f}"
            f"   |  KMeans Sil {s_fu[0]:.4f} DB {s_fu[1]:.4f}")

    mb_ari, mb_sd, _, _ = pairwise(labs_mb)
    fu_ari, fu_sd, fu_ami, fu_ami_sd = pairwise(labs_fu)
    df_cmp = pd.DataFrame(rows_cmp)
    df_cmp.to_csv(os.path.join(OUT, "s6b_minibatch_vs_full.csv"), index=False)
    log(f"\n   MiniBatchKMeans: Sil {df_cmp['sil_minibatch'].mean():.4f} "
        f"± {df_cmp['sil_minibatch'].std():.4f},  ARI между seed-ове "
        f"{mb_ari:.4f} ± {mb_sd:.4f}")
    log(f"   Пълен KMeans:    Sil {df_cmp['sil_full'].mean():.4f} "
        f"± {df_cmp['sil_full'].std():.4f},  ARI между seed-ове "
        f"{fu_ari:.4f} ± {fu_sd:.4f}")
    log(f"   време: {time.time() - t0:.0f} сек")

    # ═══════ 1. Устойчивост по seed (пълен KMeans) ═══════
    log("\n" + "─" * 74)
    log(f"1) Устойчивост по случайна инициализация — {N_SEEDS} seed-а")
    rows_seed = []
    for s, lab in enumerate(labs_fu):
        sil, dbi, chi = metrics(X_eval, lab)
        rows_seed.append(dict(seed=s, silhouette=round(sil, 4),
                              davies_bouldin=round(dbi, 4),
                              calinski=round(chi),
                              sizes="/".join(str(c) for c in
                                             np.bincount(lab, minlength=K))))
    df_seed = pd.DataFrame(rows_seed)
    df_seed.to_csv(os.path.join(OUT, "s6b_seeds.csv"), index=False)
    log(f"   Silhouette:     {df_seed['silhouette'].mean():.4f} "
        f"± {df_seed['silhouette'].std():.4f}")
    log(f"   Davies-Bouldin: {df_seed['davies_bouldin'].mean():.4f} "
        f"± {df_seed['davies_bouldin'].std():.4f}")
    log(f"   ARI между двойки решения: {fu_ari:.4f} ± {fu_sd:.4f}")
    log(f"   AMI между двойки решения: {fu_ami:.4f} ± {fu_ami_sd:.4f}")

    NOISE_FLOOR = fu_ari
    log(f"\n   >>> ПРАГ НА ШУМА (ARI при непроменени признаци): {NOISE_FLOOR:.4f}")
    log("   Ablation стойност над този праг означава, че махнатият признак")
    log("   не променя решението повече, отколкото самата инициализация.")

    # ═══════ 2. Устойчивост по независими извадки ═══════
    log("\n" + "─" * 74)
    log(f"2) Устойчивост по независими извадки — {N_SUBSAMPLES} × {FIT_SAMPLE:,}")
    t0 = time.time()
    perm = rng.permutation(n)
    labs_sub, rows_sub = [], []
    for j in range(N_SUBSAMPLES):
        part = np.sort(perm[j * FIT_SAMPLE:(j + 1) * FIT_SAMPLE])
        Xj = scaler.transform(X[part]).astype(np.float64)
        lab = fit_full(Xj, SEED).predict(X_eval)
        labs_sub.append(lab)
        sil, dbi, chi = metrics(X_eval, lab)
        rows_sub.append(dict(subsample=j + 1, n=len(part),
                             silhouette=round(sil, 4),
                             davies_bouldin=round(dbi, 4),
                             calinski=round(chi)))
        log(f"   извадка {j + 1}: Sil {sil:.4f}  DB {dbi:.4f}  CH {chi:,.0f}")
        del Xj

    sub_ari, sub_sd, sub_ami, _ = pairwise(labs_sub)
    pd.DataFrame(rows_sub).to_csv(
        os.path.join(OUT, "s6b_subsamples.csv"), index=False)
    log(f"\n   ARI между двойки извадки: {sub_ari:.4f} ± {sub_sd:.4f}")
    log(f"   AMI между двойки извадки: {sub_ami:.4f}")
    log(f"   (сравни с прага на шума {NOISE_FLOOR:.4f})")
    log(f"   време: {time.time() - t0:.0f} сек")

    # ═══════ 3. Ablation, усреднен по seed-ове ═══════
    log("\n" + "─" * 74)
    log(f"3) Важност на признаците — ablation, усреднен по {N_SEEDS_ABL} seed-а")
    t0 = time.time()

    base_labs = labs_fu[:N_SEEDS_ABL]
    sil_f = df_seed["silhouette"].head(N_SEEDS_ABL).mean()
    db_f = df_seed["davies_bouldin"].head(N_SEEDS_ABL).mean()
    log(f"\n   Базов модел (всички {len(FEATURES)} признака): "
        f"Sil {sil_f:.4f}  DB {db_f:.4f}")

    rows_ab = []
    for j, name in enumerate(FEATURES):
        keep = [c for c in range(len(FEATURES)) if c != j]
        aris, amis, sils, dbis, chis = [], [], [], [], []
        for si in range(N_SEEDS_ABL):
            lab_j = fit_full(X_fit[:, keep], si).predict(X_eval[:, keep])
            aris.append(adjusted_rand_score(base_labs[si], lab_j))
            amis.append(adjusted_mutual_info_score(base_labs[si], lab_j))
            s_, d_, c_ = metrics(X_eval[:, keep], lab_j)
            sils.append(s_); dbis.append(d_); chis.append(c_)
        ari_m = float(np.mean(aris))
        verdict = ("ЗНАЧИМ" if ari_m < NOISE_FLOOR - 0.05
                   else "в рамките на шума")
        rows_ab.append(dict(removed=name, n_features=len(keep),
                            silhouette=round(float(np.mean(sils)), 4),
                            davies_bouldin=round(float(np.mean(dbis)), 4),
                            calinski=round(float(np.mean(chis))),
                            ari_vs_full=round(ari_m, 4),
                            ari_sd=round(float(np.std(aris)), 4),
                            ami_vs_full=round(float(np.mean(amis)), 4),
                            verdict=verdict))
        log(f"   без {name:<14} Sil {np.mean(sils):>7.4f}  DB {np.mean(dbis):>6.3f}"
            f"  ARI {ari_m:>6.4f} ± {np.std(aris):.4f}   {verdict}")

    df_ab = pd.DataFrame(rows_ab)
    df_ab.to_csv(os.path.join(OUT, "s6b_ablation.csv"), index=False)
    log(f"   време: {time.time() - t0:.0f} сек")

    order = df_ab.sort_values("ari_vs_full")
    log("\n   Подредба по важност (нисък ARI = признакът носи разделянето):")
    for rank, (_, r) in enumerate(order.iterrows(), 1):
        log(f"      {rank}. {r['removed']:<14} ARI {r['ari_vs_full']:.4f}"
            f"   {r['verdict']}")

    # ═══════ Фигура ═══════
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    colors = ["#c8552d" if v == "ЗНАЧИМ" else "#9bb7d4"
              for v in order["verdict"]]
    ax[0].barh(order["removed"], order["ari_vs_full"],
               xerr=order["ari_sd"], color=colors, capsize=3)
    ax[0].axvline(NOISE_FLOOR, color="#1f4e79", ls="--", lw=1.4,
                  label=f"noise floor = {NOISE_FLOOR:.3f}")
    ax[0].set_xlabel("Adjusted Rand Index vs. full model")
    ax[0].set_title("(a) Feature ablation — lower means more important")
    ax[0].set_xlim(0, 1.05)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=.3, axis="x")

    w = 0.38
    xs = np.arange(len(df_cmp))
    ax[1].bar(xs - w / 2, df_cmp["sil_minibatch"], w,
              label="MiniBatchKMeans", color="#9bb7d4")
    ax[1].bar(xs + w / 2, df_cmp["sil_full"], w,
              label=f"KMeans (n_init={N_INIT})", color="#2e75b6")
    ax[1].set_xlabel("Random seed")
    ax[1].set_ylabel("Silhouette score")
    ax[1].set_xticks(xs)
    ax[1].set_title("(b) Solution stability across random initialisations")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_ablation_v2.png"), dpi=600)
    plt.close()

    # ═══════ ОБОБЩЕНИЕ ═══════
    log("\n" + "=" * 74)
    log("  ОБОБЩЕНИЕ")
    log("=" * 74)
    log(f"MiniBatchKMeans, ARI между seed-ове: {mb_ari:.4f} ± {mb_sd:.4f}")
    log(f"Пълен KMeans,    ARI между seed-ове: {fu_ari:.4f} ± {fu_sd:.4f}")
    log(f"Устойчивост по независими извадки:   {sub_ari:.4f} ± {sub_sd:.4f}")
    log(f"Праг на шума за ablation:            {NOISE_FLOOR:.4f}")
    sig = order[order["verdict"] == "ЗНАЧИМ"]["removed"].tolist()
    log(f"Признаци със значим ефект: {sig if sig else 'няма'}")
    log(f"\nОбщо време: {time.time() - t_start:.0f} сек")

    with open(os.path.join(OUT, "s6b_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Готово. Резултатите са в {OUT}")


if __name__ == "__main__":
    main()

"""
СТЪПКА 6c: Групов ablation — проверка за колинеарност между признаците
=======================================================================
s6b показа, че при махане на ЕДИН признак vehicle_class, length_mm и
day_of_week почти не променят разделянето. Това обаче може да е ефект
от колинеарност: дължината на практика определя класа на превозното
средство, а is_weekend е функция на day_of_week. Когато единият е
махнат, другият покрива информацията и ефектът изглежда нулев.

Тук се махат цели групи признаци, за да се види какво носи групата
като цяло:

  G1  без vehicle_class + length_mm      (характеристика на возилото)
  G2  без day_of_week + is_weekend       (седмичен режим)
  G3  без speed_kmh + length_mm          (двата непрекъснати признака)
  G4  само времеви: hour, day_of_week, is_weekend
  G5  само нетемпорални: speed_kmh, vehicle_class, length_mm

Прагът на шума се измерва отново, за да е самостоятелен отчетът.

Изход: out/s6c_group_ablation.csv, out/fig_group_ablation.png,
       out/s6c_report.txt
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

FIT_SAMPLE = 300_000
EVAL_SAMPLE = 100_000
SIL_SAMPLE = 20_000
N_INIT = 10
N_SEEDS = 3
# ──────────────────────────────────────────────────────────

FEATURES = ["speed_kmh", "hour", "day_of_week",
            "vehicle_class", "length_mm", "is_weekend"]
IDX = {f: i for i, f in enumerate(FEATURES)}

GROUPS = [
    ("G0_full", FEATURES, "Всички признаци — базов модел"),
    ("G1_no_vehicle", ["speed_kmh", "hour", "day_of_week", "is_weekend"],
     "Без vehicle_class + length_mm (характеристика на возилото)"),
    ("G2_no_week", ["speed_kmh", "hour", "vehicle_class", "length_mm"],
     "Без day_of_week + is_weekend (седмичен режим)"),
    ("G3_no_continuous", ["hour", "day_of_week", "vehicle_class", "is_weekend"],
     "Без speed_kmh + length_mm (непрекъснатите признаци)"),
    ("G4_temporal_only", ["hour", "day_of_week", "is_weekend"],
     "Само времеви признаци"),
    ("G5_nontemporal_only", ["speed_kmh", "vehicle_class", "length_mm"],
     "Само нетемпорални признаци"),
]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def fit_full(X, seed):
    return KMeans(n_clusters=K, random_state=seed, n_init=N_INIT,
                  max_iter=300).fit(X)


def metrics(X, labels):
    if len(set(labels)) < 2:
        return np.nan, np.nan, np.nan
    return (silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, len(X)),
                             random_state=SEED),
            davies_bouldin_score(X, labels),
            calinski_harabasz_score(X, labels))


def profile(meta, labels):
    cols = ["hour", "day_of_week", "vehicle_class", "speed_kmh", "length_mm"]
    df = pd.DataFrame(meta[:, :5], columns=cols)
    df["cluster"] = labels
    p = df.groupby("cluster")[cols].mean().round(2)
    p["n"] = df.groupby("cluster").size()
    p["share_%"] = (100 * p["n"] / len(df)).round(2)
    return p.sort_values("n", ascending=False)


def main():
    t_start = time.time()
    log("=" * 74)
    log(f"  СТЪПКА 6c: Групов ablation — колинеарност  ({ENCODING})")
    log("=" * 74)

    X = np.load(os.path.join(OUT, f"X_{ENCODING}.npy"), mmap_mode="r")
    M = np.load(os.path.join(OUT, "meta.npy"), mmap_mode="r")
    n = len(X)
    log(f"\nДанни: {n:,} записа, {X.shape[1]} признака")
    log(f"Модел: KMeans (k-means++), K={K}, n_init={N_INIT}, {N_SEEDS} seed-а")

    log("\nСтандартизация (partial_fit по части)...")
    scaler = StandardScaler()
    for i in range(0, n, CHUNK):
        scaler.partial_fit(X[i:i + CHUNK])

    rng = np.random.RandomState(SEED)
    idx_eval = np.sort(rng.choice(n, EVAL_SAMPLE, replace=False))
    X_eval = scaler.transform(X[idx_eval]).astype(np.float64)
    M_eval = np.asarray(M[idx_eval])
    idx_fit = np.sort(rng.choice(n, FIT_SAMPLE, replace=False))
    X_fit = scaler.transform(X[idx_fit]).astype(np.float64)
    log(f"Извадка за обучение: {FIT_SAMPLE:,}, за оценка: {EVAL_SAMPLE:,}")

    # ── корелационна матрица (мотивира целия тест) ──
    log("\n" + "─" * 74)
    log("Корелации на Пирсън между признаците (върху оценъчната извадка)")
    C = np.corrcoef(X_eval, rowvar=False)
    log("".ljust(15) + "".join(f"{f[:9]:>11}" for f in FEATURES))
    for i, f in enumerate(FEATURES):
        log(f"{f:<15}" + "".join(f"{C[i, j]:>11.3f}" for j in range(len(FEATURES))))
    strong = [(FEATURES[i], FEATURES[j], C[i, j])
              for i, j in itertools.combinations(range(len(FEATURES)), 2)
              if abs(C[i, j]) > 0.5]
    log("\nСилни корелации (|r| > 0.5):")
    for a, b, r in sorted(strong, key=lambda t: -abs(t[2])):
        log(f"   {a} ↔ {b}: r = {r:.3f}")
    if not strong:
        log("   няма")

    # ── базов модел и праг на шума ──
    log("\n" + "─" * 74)
    log("Базов модел и праг на шума")
    base_labs = [fit_full(X_fit, s).predict(X_eval) for s in range(N_SEEDS)]
    noise = [adjusted_rand_score(base_labs[a], base_labs[b])
             for a, b in itertools.combinations(range(N_SEEDS), 2)]
    NOISE_FLOOR = float(np.mean(noise))
    sil_b, db_b, ch_b = metrics(X_eval, base_labs[0])
    log(f"   Silhouette {sil_b:.4f}  DB {db_b:.4f}  CH {ch_b:,.0f}")
    log(f"   ПРАГ НА ШУМА (ARI между seed-ове): {NOISE_FLOOR:.4f}")

    # ── групите ──
    rows = []
    for tag, feats, desc in GROUPS:
        log("\n" + "─" * 74)
        log(f"{tag}:  {desc}")
        log(f"   признаци ({len(feats)}): {feats}")
        keep = [IDX[f] for f in feats]
        t0 = time.time()

        aris, amis, sils, dbis, chis = [], [], [], [], []
        lab_first = None
        for si in range(N_SEEDS):
            lab = fit_full(X_fit[:, keep], si).predict(X_eval[:, keep])
            if si == 0:
                lab_first = lab
            aris.append(adjusted_rand_score(base_labs[si], lab))
            amis.append(adjusted_mutual_info_score(base_labs[si], lab))
            s_, d_, c_ = metrics(X_eval[:, keep], lab)
            sils.append(s_); dbis.append(d_); chis.append(c_)

        ari_m, ari_s = float(np.mean(aris)), float(np.std(aris))
        verdict = ("базов" if tag == "G0_full" else
                   "ЗНАЧИМ" if ari_m < NOISE_FLOOR - 0.05 else
                   "в рамките на шума")
        dt = time.time() - t0

        log(f"   Silhouette {np.mean(sils):.4f}   DB {np.mean(dbis):.4f}"
            f"   CH {np.mean(chis):,.0f}")
        log(f"   ARI спрямо пълния модел: {ari_m:.4f} ± {ari_s:.4f}   {verdict}")
        log(f"   време: {dt:.0f} сек")

        p = profile(M_eval, lab_first)
        p.to_csv(os.path.join(OUT, f"profile_{tag}.csv"))
        log("\n   Профил на клъстерите (оригинални единици):")
        log(p.to_string())

        rows.append(dict(config=tag, description=desc, n_features=len(feats),
                         features=" + ".join(feats),
                         silhouette=round(float(np.mean(sils)), 4),
                         davies_bouldin=round(float(np.mean(dbis)), 4),
                         calinski=round(float(np.mean(chis))),
                         ari_vs_full=round(ari_m, 4),
                         ari_sd=round(ari_s, 4),
                         ami_vs_full=round(float(np.mean(amis)), 4),
                         verdict=verdict, time_s=round(dt)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "s6c_group_ablation.csv"), index=False)

    # ── фигура ──
    sub = df[df["config"] != "G0_full"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))

    colors = ["#c8552d" if v == "ЗНАЧИМ" else "#9bb7d4" for v in sub["verdict"]]
    ax[0].barh(sub["config"], sub["ari_vs_full"], xerr=sub["ari_sd"],
               color=colors, capsize=3)
    ax[0].axvline(NOISE_FLOOR, color="#1f4e79", ls="--", lw=1.4,
                  label=f"noise floor = {NOISE_FLOOR:.3f}")
    ax[0].set_xlabel("Adjusted Rand Index vs. full model")
    ax[0].set_title("(a) Group ablation — lower means the group carries the split")
    ax[0].set_xlim(0, 1.05)
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=.3, axis="x")

    ax[1].bar(df["config"], df["silhouette"], color="#2e75b6")
    ax[1].axhline(sil_b, color="#c8552d", ls="--", lw=1.3,
                  label=f"full model = {sil_b:.3f}")
    ax[1].set_ylabel("Silhouette score")
    ax[1].set_title("(b) Internal validity of each feature subset")
    ax[1].tick_params(axis="x", rotation=30)
    for lbl in ax[1].get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_group_ablation.png"), dpi=600)
    plt.close()

    # ── обобщение ──
    log("\n" + "=" * 74)
    log("  СРАВНИТЕЛНА ТАБЛИЦА")
    log("=" * 74)
    log(f"{'Конфигурация':<22}{'призн.':>8}{'Silh.':>9}{'DB':>9}"
        f"{'ARI':>9}{'заключение':>22}")
    for _, r in df.iterrows():
        log(f"{r['config']:<22}{r['n_features']:>8}{r['silhouette']:>9.4f}"
            f"{r['davies_bouldin']:>9.3f}{r['ari_vs_full']:>9.4f}"
            f"{r['verdict']:>22}")

    log(f"\nПраг на шума: {NOISE_FLOOR:.4f}")
    log(f"Общо време: {time.time() - t_start:.0f} сек")

    with open(os.path.join(OUT, "s6c_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Готово. Резултатите са в {OUT}")


if __name__ == "__main__":
    main()

"""
СТЪПКА 1: Подготовка на матрицата от признаци
==============================================
Чете traffic_anonymized.csv на части, изгражда признаците в компактни
типове (float32) и записва резултата като .npy файлове.

Това позволява следващите скриптове да четат данните чрез mmap,
без да зареждат целия набор в паметта — критично при 8 GB RAM.

Изход:
  out/X_linear.npy      — 6 признака, линейно кодиране
  out/X_cyclic.npy      — 7 признака, sin/cos кодиране
  out/meta.npy          — час, ден, клас, посока в оригинални единици
  out/prep_report.txt
"""

import os
import numpy as np
import pandas as pd

# ─── КОНФИГУРАЦИЯ ─────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE, "traffic_anonymized.csv")
OUT = os.path.join(BASE, "out")
CHUNK = 500_000
# ──────────────────────────────────────────────────────────

os.makedirs(OUT, exist_ok=True)
log_lines = []


def log(s=""):
    print(s)
    log_lines.append(str(s))


USE_COLS = ["datetime", "direction", "speed_kmh", "hour",
            "day_of_week", "vehicle_class", "length_mm",
            "is_weekend", "year"]

DTYPES = {
    "speed_kmh": "float32",
    "hour": "int8",
    "day_of_week": "int8",
    "vehicle_class": "int8",
    "length_mm": "float32",
    "is_weekend": "int8",
}


def main():
    log("=" * 64)
    log("  СТЪПКА 1: Подготовка на признаците")
    log("=" * 64)

    # какви колони реално има във файла
    head = pd.read_csv(INPUT_FILE, nrows=5)
    available = [c for c in USE_COLS if c in head.columns]
    log(f"\nНалични колони: {available}")

    has_year = "year" in available
    has_dir = "direction" in available

    feat_lin, feat_cyc, meta_list = [], [], []
    total = 0

    log("\nЧетене на части...")
    for i, ch in enumerate(pd.read_csv(INPUT_FILE, usecols=available,
                                       chunksize=CHUNK, low_memory=False)):
        # валидни редове
        ch = ch.dropna(subset=["speed_kmh", "hour", "day_of_week",
                               "vehicle_class", "length_mm"])
        ch = ch[ch["vehicle_class"].between(0, 7)]

        # компактни типове
        for c, t in DTYPES.items():
            if c in ch.columns:
                ch[c] = ch[c].astype(t)

        # ── линейни признаци ──
        lin = np.column_stack([
            ch["speed_kmh"].to_numpy("float32"),
            ch["hour"].to_numpy("float32"),
            ch["day_of_week"].to_numpy("float32"),
            ch["vehicle_class"].to_numpy("float32"),
            ch["length_mm"].to_numpy("float32"),
            ch["is_weekend"].to_numpy("float32"),
        ]).astype("float32")

        # ── циклични признаци ──
        h_ang = 2 * np.pi * ch["hour"].to_numpy("float32") / 24.0
        d_ang = 2 * np.pi * ch["day_of_week"].to_numpy("float32") / 7.0
        cyc = np.column_stack([
            ch["speed_kmh"].to_numpy("float32"),
            np.sin(h_ang), np.cos(h_ang),
            np.sin(d_ang), np.cos(d_ang),
            ch["vehicle_class"].to_numpy("float32"),
            ch["length_mm"].to_numpy("float32"),
        ]).astype("float32")

        # ── метаданни за интерпретация и валидация ──
        year = ch["year"].to_numpy("int16") if has_year else np.zeros(len(ch), "int16")
        direction = (ch["direction"].astype(str).str.startswith("Ц").to_numpy("int8")
                     if has_dir else np.zeros(len(ch), "int8"))
        meta = np.column_stack([
            ch["hour"].to_numpy("float32"),
            ch["day_of_week"].to_numpy("float32"),
            ch["vehicle_class"].to_numpy("float32"),
            ch["speed_kmh"].to_numpy("float32"),
            ch["length_mm"].to_numpy("float32"),
            year.astype("float32"),
            direction.astype("float32"),
        ]).astype("float32")

        feat_lin.append(lin)
        feat_cyc.append(cyc)
        meta_list.append(meta)
        total += len(ch)

        if (i + 1) % 5 == 0:
            log(f"  обработени {total:,} реда...")

    X_lin = np.vstack(feat_lin); del feat_lin
    X_cyc = np.vstack(feat_cyc); del feat_cyc
    M = np.vstack(meta_list);    del meta_list

    log(f"\nОбщо валидни записи: {len(X_lin):,}")
    log(f"  X_linear: {X_lin.shape}, {X_lin.nbytes / 1e6:.0f} MB")
    log(f"  X_cyclic: {X_cyc.shape}, {X_cyc.nbytes / 1e6:.0f} MB")

    np.save(os.path.join(OUT, "X_linear.npy"), X_lin)
    np.save(os.path.join(OUT, "X_cyclic.npy"), X_cyc)
    np.save(os.path.join(OUT, "meta.npy"), M)

    # ── описателна статистика ──
    log("\n--- ОПИСАТЕЛНА СТАТИСТИКА ---")
    names = ["speed_kmh", "hour", "day_of_week", "vehicle_class",
             "length_mm", "is_weekend"]
    log(f"{'признак':<16}{'средно':>12}{'ст.откл.':>12}{'min':>10}{'max':>10}")
    for j, n in enumerate(names):
        col = X_lin[:, j]
        log(f"{n:<16}{col.mean():>12.3f}{col.std():>12.3f}"
            f"{col.min():>10.1f}{col.max():>10.1f}")

    if has_year:
        log("\n--- ЗАПИСИ ПО ГОДИНИ ---")
        yrs, cnts = np.unique(M[:, 5].astype(int), return_counts=True)
        for y, c in zip(yrs, cnts):
            if y > 0:
                log(f"  {y}: {c:,}")

    with open(os.path.join(OUT, "prep_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    log(f"\nГотово. Файловете са в {OUT}")


if __name__ == "__main__":
    main()
"""
Rebuild traffic_anonymized.csv from the yearly files in data/.
Run this once after cloning the repository: python merge_data.py
"""
import glob, gzip, os

BASE = os.path.dirname(os.path.abspath(__file__))
parts = sorted(glob.glob(os.path.join(BASE, "data", "traffic_*.csv.gz")))
dst = os.path.join(BASE, "traffic_anonymized.csv")
rows = 0
with open(dst, "w", encoding="utf-8", newline="") as out:
    for i, p in enumerate(parts):
        with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
            header = f.readline()
            if i == 0:
                out.write(header)
            for line in f:
                out.write(line)
                rows += 1
print(f"{rows:,} rows written to {dst}")

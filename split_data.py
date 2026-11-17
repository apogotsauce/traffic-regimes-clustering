"""
Split traffic_anonymized.csv into one compressed file per year (data/traffic_YYYY.csv.gz),
so that each file is below GitHub's 100 MB limit. Run once: python split_data.py
"""
import csv, gzip, os

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "traffic_anonymized.csv")
OUT = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)

files, writers = {}, {}
with open(SRC, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    yi = header.index("year")
    for row in reader:
        y = row[yi]
        if y not in writers:
            files[y] = gzip.open(os.path.join(OUT, f"traffic_{y}.csv.gz"), "wt", encoding="utf-8", newline="")
            writers[y] = csv.writer(files[y], lineterminator="\n")
            writers[y].writerow(header)
        writers[y].writerow(row)
for fh in files.values():
    fh.close()
print("Written:", sorted(os.listdir(OUT)))

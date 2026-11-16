"""
Remove licence-plate numbers from the traffic data before publishing.

Reads  cleaned_traffic.csv          (original, contains plates - NOT for GitHub)
Writes traffic_anonymized.csv       (same rows, without 'plate' and 'country')

The two removed columns are not used anywhere in the analysis
(see USE_COLS in s1_prepare.py), so all results stay the same.
Run:  python anonymize.py
"""
import csv
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE, "cleaned_traffic.csv")
OUTPUT_FILE = os.path.join(BASE, "traffic_anonymized.csv")
DROP = {"plate", "country"}

# Bulgarian-style plate, e.g. PB1234AB; used only for the final check
PLATE_RE = re.compile(r"\b[A-Z]{1,2}\d{4}[A-Z]{1,2}\b")


def main():
    rows = 0
    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        keep = [i for i, name in enumerate(header) if name not in DROP]
        missing = DROP - set(header)
        if missing:
            print("Warning: columns not found:", missing)
        writer.writerow([header[i] for i in keep])
        for row in reader:
            writer.writerow([row[i] for i in keep])
            rows += 1
            if rows % 1_000_000 == 0:
                print(f"{rows:,} rows done")
    print(f"Finished: {rows:,} rows written to {OUTPUT_FILE}")

    # check: no plate-like text left in the new file
    found = 0
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if PLATE_RE.search(line):
                found += 1
    print("Plate-like values left:", found)


if __name__ == "__main__":
    main()

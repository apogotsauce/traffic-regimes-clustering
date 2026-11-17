# Traffic regimes from multi-year vehicle archives

Code and results for the article

> Beytula, A.; Mecheva, T. *Identifying Reproducible Traffic Regimes from Multi-Year Vehicle Archives: A Comparison of Clustering Algorithms.* Technologies (submitted, 2026).

The study clusters 11,266,900 individual vehicle records from one automatic counting
station in Plovdiv, Bulgaria (2022–2024) with K-Means, mini-batch K-Means,
a Gaussian mixture model, Ward agglomerative clustering and HDBSCAN, and tests how
reproducible the resulting partitions are.

## Data

The anonymized vehicle records are in `data/`, one compressed file per year:

| File | Records |
|---|---|
| `data/traffic_2022.csv.gz` | 3,234,878 |
| `data/traffic_2023.csv.gz` | 6,004,518 |
| `data/traffic_2024.csv.gz` | 2,032,419 |

The original export contained licence-plate numbers. `anonymize.py` removed the
`plate` and `country` columns; these columns are not used anywhere in the analysis.
`split_data.py` produced the yearly files. The 4,915 records with an invalid vehicle
class code are removed later by `s1_prepare.py`, giving the 11,266,900 records used in the article.

The data were provided by the Center of Competence "Smart Mechatronic, Eco- and Energy-Saving
Systems and Technologies" (Gabrovo, Bulgaria).

Columns:
`datetime, direction, speed_kmh, vehicle_class, lane, length_mm, hour, day_of_week, month, year, is_weekend, vehicle_type`

## How to run

Python 3.14. Install the packages:

```
pip install -r requirements.txt
```

First rebuild the single data file:

```
python merge_data.py
```

This creates `traffic_anonymized.csv` (about 1 GB). Then run the scripts in this order.
All outputs go to `out/`.

| Script | What it does |
|---|---|
| `merge_data.py` | Joins the yearly files from `data/` into `traffic_anonymized.csv` |
| `s1_prepare.py` | Builds and standardizes the six features, saves `X_linear.npy`, `X_cyclic.npy`, `meta.npy` |
| `s2_select_k.py` | Selects the number of clusters K (elbow, silhouette, Davies–Bouldin, Calinski–Harabasz) |
| `s3_clustering.py` | Runs the clustering algorithms and computes the validity indices |
| `s4_hdbscan_encoding.py` | HDBSCAN with different feature encodings |
| `s6_robustness.py` | Stability across random seeds and subsamples, single-feature ablation (first version) |
| `s6b_robustness.py` | Corrected stability analysis; mini-batch vs. standard K-Means |
| `s6c_group_ablation.py` | Ablation of feature groups |
| `step4_english_figures.py` | Exploratory and PCA figures with English labels (`figures_en/`) |

The `.npy` files are not in the repository (too large); `s1_prepare.py` recreates them.
Comments inside the scripts are in Bulgarian.

## Results

`out/` contains the reports (`*_report.txt`), result tables (`*.csv`) and figures (`*.png`)
used in the article.

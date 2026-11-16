# Traffic regimes from multi-year vehicle archives

Code and results for the article

> Beytula, A.; Mecheva, T. *Identifying Reproducible Traffic Regimes from Multi-Year Vehicle Archives: A Comparison of Clustering Algorithms.* Technologies (submitted, 2026).

The study clusters 11,266,900 individual vehicle records from one automatic counting
station in Plovdiv, Bulgaria (2022–2024) with K-Means, mini-batch K-Means,
a Gaussian mixture model, Ward agglomerative clustering and HDBSCAN, and tests how
reproducible the resulting partitions are.

## Data

The raw data are **not included** in this repository.

- The original export contains licence-plate numbers, which are personal data.
  `anonymize.py` removes the `plate` and `country` columns and writes
  `traffic_anonymized.csv`; these columns are not used anywhere in the analysis.
- Even after anonymisation the file is about 1 GB, above GitHub's file-size limit.

The data were provided by the Center of Competence "Smart Mechatronic, Eco- and Energy-Saving Systems and Technologies" (Gabrovo, Bulgaria) and are available from the authors on reasonable request.

Expected columns of `traffic_anonymized.csv`:
`datetime, direction, speed_kmh, vehicle_class, lane, length_mm, hour, day_of_week, month, year, is_weekend, vehicle_type`

## How to run

Python 3.14. Install the packages:

```
pip install -r requirements.txt
```

Put `traffic_anonymized.csv` in the project folder and run the scripts in this order.
All outputs go to `out/`.

| Script | What it does |
|---|---|
| `anonymize.py` | Removes licence plates from the original export (only needed once) |
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

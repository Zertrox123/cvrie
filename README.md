## CVRIE – Unsupervised part (patient testimonies clustering)

This repository currently contains the **unsupervised** deliverables for the CVRIE project:
- **2 notebooks** (cleaning/EDA + modeling)
- **1 CLI script** that outputs the clustering (exit code **84** on error)
- **exported figures** for the defense

### Requirements (subject constraints)
- **Allowed tools**: scikit-learn, pandas, numpy, matplotlib, Jupyter
- **No labels** are added to the dataset
- The hidden `color` column (e.g. `0x000000`) is used **only for visualization**, never for training

### Unsupervised folder structure
- `unsupervised/01_data_cleaning_eda.ipynb`: parsing, cleaning, EDA + CSV export
- `unsupervised/02_vectorization_clustering.ipynb`: TF‑IDF → LSA → clustering comparison, metrics, plots, final choice
- `unsupervised/cluster_testimonies.py`: CLI script that produces `unsupervised/unsupervised_clusters.csv`
- `unsupervised/DECISIONS.md`: rationale for all key choices (defense-ready)
- `unsupervised/figures/`: exported PNG plots used in the defense

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (notebooks)
- Run `unsupervised/01_data_cleaning_eda.ipynb` first
  - Produces `unsupervised/cleaned_unsupervised_dataset.csv`
  - Produces EDA plots in `unsupervised/figures/`
- Then run `unsupervised/02_vectorization_clustering.ipynb`
  - Produces `unsupervised/unsupervised_clusters.csv`
  - Produces modeling plots in `unsupervised/figures/`

### Run (CLI clustering)

From the project root:

```bash
.venv/bin/python unsupervised/cluster_testimonies.py \
  --input Student_Dataset.csv \
  --output unsupervised/unsupervised_clusters.csv
```

### Notes
- The final frozen model choice is documented in `unsupervised/DECISIONS.md`.


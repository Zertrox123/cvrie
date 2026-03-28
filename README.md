## CVRIE – Computer vision & text (Epitech)

This repository contains **supervised** (PneumoniaMNIST) and **unsupervised** (patient testimonies clustering) deliverables for the CVRIE project.

### Supervised part (PneumoniaMNIST)

- **2 notebooks**: `supervised/01_eda_preprocessing.ipynb` (EDA + pre-processing rationale), `supervised/02_models_evaluation.ipynb` (model comparison, metrics, conclusions)
- **Dataset**: place MedMNIST Pneumonia **NumPy** exports at **`pneumoniamnist/`** next to `export_pneumonia_png.py` — `train_images.npy`, `train_labels.npy`, and the same for **`val`** and **`test`**
- **1 CLI** [`supervised/train_classify_pneumonia.py`](supervised/train_classify_pneumonia.py): trains a classifier and writes `supervised/supervised_test_predictions.csv` (exit code **84** on error)
- **Rationale**: [`supervised/DECISIONS.md`](supervised/DECISIONS.md); figures under `supervised/figures/` after running the notebooks

```bash
.venv/bin/python supervised/train_classify_pneumonia.py \
  --data-dir pneumoniamnist \
  --model logistic_regression \
  --fit-on train_val \
  --output supervised/supervised_test_predictions.csv
```

Shared helpers: [`supervised/pneumonia_preprocessing.py`](supervised/pneumonia_preprocessing.py) (pickle-safe features), [`supervised/pneumonia_evaluation.py`](supervised/pneumonia_evaluation.py) (threshold tuning + rates). The CLI **tunes** the pneumonia probability cut-off on `val` by default (see `--no-tune-threshold` / `--threshold`).

### Unsupervised part (patient testimonies)

- **2 notebooks** (cleaning/EDA + modeling)
- **1 CLI script** that outputs the clustering (exit code **84** on error)
- **exported figures** for the defense

### Modeling notes (summary)
- **Text cleaning** maps punctuation to spaces so tokens align with sklearn’s stop-word list (see `unsupervised/DECISIONS.md`).
- **Stop words**: `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` plus a small fixed set of narrative fillers—same list in notebooks **01**, **02**, and in `cluster_testimonies.py`.
- **TF‑IDF + LSA** are **fitted on 80%** of rows (`train_test_split`, seed **42**), then **all** rows are transformed for clustering, so hold-out texts do not influence vocabulary, IDF, or SVD.
- **`k` for KMeans**: notebook **02** picks the best silhouette among KMeans-on-LSA runs in the sweep; the **CLI** currently uses a **fixed `n_clusters=20`** export path—align the script with the notebook’s `best_k` if you need identical labels in `unsupervised_clusters.csv`.

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
- **Supervised**: run `supervised/01_eda_preprocessing.ipynb`, then `supervised/02_models_evaluation.ipynb` (kernel working directory can be repo root or `supervised/`; paths resolve automatically). Plots go to `supervised/figures/`.
- **Unsupervised**: run `unsupervised/01_data_cleaning_eda.ipynb` first
  - Produces `unsupervised/cleaned_unsupervised_dataset.csv`
  - Produces EDA plots in `unsupervised/figures/` (including length boxplot, id vs. word count, top tokens)
- Then run `unsupervised/02_vectorization_clustering.ipynb`
  - Produces `unsupervised/unsupervised_clusters.csv` at the end (save clustering output), using the notebook’s selected `best_k` from the sweep
  - Produces modeling plots in `unsupervised/figures/` (LSA variance curve, silhouette vs. k, leaderboard, 2D projections by cluster and by hidden `color`, …)

### Run (CLI clustering)

From the project root:

```bash
.venv/bin/python unsupervised/cluster_testimonies.py \
  --input Student_Dataset.csv \
  --output unsupervised/unsupervised_clusters.csv
```

Running the CLI **overwrites** `unsupervised/unsupervised_clusters.csv` with **k = 20** clusters. Use the notebook export if you need labels consistent with **`best_k`** (e.g. **24** in the last documented run).

### Notes
- Rationale, metrics, and the notebook-vs-CLI `k` distinction are in `unsupervised/DECISIONS.md`.


## Unsupervised part – data cleaning

This folder contains the code used to perform **cleaning and preprocessing** for the
unsupervised learning task of the CVRIE project (patient testimonies clustering).

- **Language**: `Python`
- **Main libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- **Dataset used**: `Student_Dataset.csv` (provided with the subject)

The cleaning code:

- Parses the raw CSV file and **extracts three logical columns**:
  - `id`: integer identifier
  - `color`: optional hidden label / color code (values such as `0x000000`)
  - `testimony_raw`: original free‑text testimony
- Builds a cleaned text column:
  - lower‑casing
  - normalising quotes and whitespace
  - removing non textual noise characters
  - removing empty / invalid entries
  - removing duplicate testimonies

The **hidden color column is kept** so that you can later color your clusters for
visualisation during the defense, but **it must not be used during training**.

### Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the cleaning script

From the project root:

```bash
python unsupervised_preprocessing.py \
  --input Student_Dataset.csv \
  --output cleaned_unsupervised_dataset.csv
```

This will create `cleaned_unsupervised_dataset.csv` containing:

- `id`
- `color` (may be empty when no color code is present)
- `testimony_raw`
- `testimony_clean` (text ready to be vectorised for clustering)


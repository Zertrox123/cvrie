# Unsupervised (CVRIE) – Decisions & Justifications

This document explains **why** we made each major choice for the unsupervised part, in a way that is easy to defend orally.

## Non‑negotiable constraints (from the subject)

- **No labels**: the testimonies dataset is unlabeled; we do not create labels and we do not evaluate with hidden ground-truth.
- **No dataset modification**: we do not add columns that would leak semantics into training.
- **Hidden `color` column**: it is **kept only for visualization** (cluster plots colored by `color`) and is **never** used as a model input.
- **Allowed tools**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, Jupyter.

## Decision 1 — Parsing the raw CSV line-by-line

- **Decision**: Parse `Student_Dataset.csv` with Python’s `csv` reader and detect an optional `0xRRGGBB` field.
- **Options considered**:
  - `pandas.read_csv()` directly (simple, but fails/produces shifted columns on heterogeneous rows).
  - Line-by-line parsing (robust to heterogeneous format).
- **Why**: The file has mixed row schemas (`id,text…` vs `id,color,text…`). Line parsing keeps it correct and auditable.
- **Trade-offs**: More code than `read_csv`, but far less risk of silent parsing corruption.
- **Where**: `unsupervised/01_data_cleaning_eda.ipynb`.

## Decision 2 — Conservative text normalization (transparent cleaning)

- **Decision**: Basic normalization only: lowercase, normalize common unicode punctuation, remove unusual characters, collapse whitespace.
- **Options considered**:
  - Heavy NLP (lemmatization, external models) — not allowed / not “by the books” under the tool constraints.
  - Minimal normalization — chosen.
- **Why**: Keeps preprocessing **retraceable** and avoids injecting strong assumptions. Also stays inside allowed libraries.
- **Trade-offs**: Some linguistic variants remain (e.g., tense/plural), but TF‑IDF n-grams + dimensionality reduction partially compensate.
- **Where**: `normalise_text()` in `unsupervised/01_data_cleaning_eda.ipynb`.

## Decision 3 — Removing exact duplicates + filtering extremely short entries

- **Decision**: Drop exact duplicates on the cleaned text and remove entries shorter than a small threshold (`MIN_LEN=5` chars).
- **Options considered**:
  - Keep everything (duplicates can overweight topics and distort clustering).
  - Remove duplicates (chosen).
  - Aggressive filtering (risk of deleting meaningful short symptoms).
- **Why**:
  - Duplicates create artificial density and can make clusters look “better” numerically while being less representative.
  - Extremely short strings often become noise in TF‑IDF.
- **Trade-offs**: If a short testimony is meaningful, it may be dropped; we keep the threshold small to reduce that risk.
- **Where**: `unsupervised/01_data_cleaning_eda.ipynb`.

## Decision 4 — Primary representation: TF‑IDF with uni+bi-grams

- **Decision**: Use `TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.90)`.
- **Options considered**:
  - CountVectorizer (simpler, but weights common words too heavily).
  - TF‑IDF (chosen).
  - Embeddings (SentenceTransformers) — not allowed under the strict tool constraint.
- **Why**:
  - TF‑IDF is a strong baseline for short clinical-like narratives.
  - Bi-grams capture key phrases (e.g., “shortness breath”, “chest pain”) that single tokens miss.
  - `min_df` removes ultra-rare tokens that act like identifiers; `max_df` removes near-global words that don’t separate topics.
- **Trade-offs**:
  - Bag-of-words ignores word order beyond n-grams and ignores deeper semantics.
  - Vocabulary can be large; this is addressed by the next decision (LSA).
- **Where**: `unsupervised/02_vectorization_clustering.ipynb`.

## Decision 5 — Dimensionality reduction: TF‑IDF → TruncatedSVD (LSA) + normalization

- **Decision**: Reduce TF‑IDF into a dense space with `TruncatedSVD(n_components=100)` then `Normalizer`.
- **Options considered**:
  - Cluster directly in sparse TF‑IDF only.
  - Reduce with SVD (chosen).
  - UMAP/t-SNE for modeling — not in allowed libs; also unstable for clustering training.
- **Why**:
  - Clustering algorithms and distance computations behave better in a dense, lower-dimensional space.
  - LSA denoises synonyms/related terms by capturing latent co-occurrence structure.
- **Trade-offs**:
  - Some interpretability is lost in reduced space; we compensate by extracting top TF‑IDF terms per cluster.
- **Where**: `unsupervised/02_vectorization_clustering.ipynb`.

## Decision 6 — Compare multiple clustering families (not just one)

- **Decision**: Try KMeans, Agglomerative (average linkage), and DBSCAN (on LSA).
- **Why** (aligns with `notes.md` and subject expectations):
  - **KMeans**: fast, stable baseline for text features; easy to sweep k.
  - **Agglomerative**: can model non-spherical clusters; useful comparison.
  - **DBSCAN**: density-based; can mark outliers as noise and is a good “adaptive clustering” candidate.
- **Trade-offs**:
  - DBSCAN is sensitive to `eps` and scaling; we keep a small grid on LSA features for defendability.
- **Where**: sweeps in `unsupervised/02_vectorization_clustering.ipynb`.

## Decision 7 — Evaluation without labels: multiple internal metrics + size balance + qualitative checks

- **Decision**: Use a combination of:
  - **Silhouette** (sampled for speed)
  - **Calinski–Harabasz**
  - **Davies–Bouldin**
  - **Cluster size balance** (avoid 1 huge cluster)
  - **Interpretability**: top terms and representative testimonies
- **Why**:
  - No ground-truth labels are available; a single internal metric can be misleading.
  - Cluster size balance is explicitly mentioned in `notes.md` as important for the defense.
- **Trade-offs**:
  - Internal metrics do not guarantee “human-meaningful” groupings; qualitative inspection is necessary.
- **Where**: `unsupervised/02_vectorization_clustering.ipynb`.

## Decision 7b — Loss / objective (what the model optimizes)

- **Decision**: Make the clustering objective explicit for the defense.
- **KMeans objective (“loss”)**: minimize **inertia** (sum of squared distances to centroids) on the **LSA vectors**.\n  This is the quantity KMeans iteratively reduces while updating assignments/centroids.
- **Why it matters**: it explains what “training” means for KMeans and why we compare multiple runs/parameters.

## Decision 8 — Defense visualization: 2D projection + two colorings

- **Decision**: Project TF‑IDF to 2D with `TruncatedSVD(n_components=2)` and plot:
  - points colored by **cluster assignment**
  - points colored by hidden `**color`**
- **Why**: The subject asks to use the hidden `color` to color the plot so an examiner can visually verify coherence at a glance.
- **Trade-offs**:
  - 2D projections can distort distances; these plots are for **visual inspection**, not training decisions.
- **Where**: `unsupervised/02_vectorization_clustering.ipynb`.

## Final frozen choice (for defense)

- **Chosen pipeline**: `TF-IDF (uni+bi-grams) → TruncatedSVD(100) + Normalizer → KMeans(k=20)`
- **Why this one**:
  - It was the **best silhouette** among the KMeans/LSA runs in our sweep.
  - It produces **20 clusters** (no noise) with a **reasonably balanced** size distribution (no single cluster dominates).
  - It is stable and easy to explain/defend compared to DBSCAN sensitivity to `eps`.
- **Evidence (from executed notebook run)**:
  - **Run name**: `kmeans_lsa20`
  - **silhouette**: `0.070455`
  - **Calinski–Harabasz**: `14.080045`
  - **Davies–Bouldin**: `3.254502` (lower is better)
  - **Largest cluster sizes (top 10)**: `[101, 77, 68, 65, 61, 61, 61, 58, 50, 47]`
- **Where**:
  - Model selection and plots: `unsupervised/02_vectorization_clustering.ipynb`
  - Reproducible CLI output: `unsupervised/cluster_testimonies.py`

## Outputs produced

- `unsupervised/cleaned_unsupervised_dataset.csv`: cleaned dataset used for modeling.
- `unsupervised/unsupervised_clusters.csv`: cluster assignments (plus `color` for visualization only).


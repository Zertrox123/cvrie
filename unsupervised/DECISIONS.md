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

- **Decision**: Basic normalization only: lowercase, replace common Unicode punctuation with ASCII, map **anything that is not** `a–z`, digits, spaces, or apostrophes **to a space** (not only “strip unusual chars”), then collapse whitespace.
- **Options considered**:
  - Heavy NLP (lemmatization, external models) — not allowed / not “by the books” under the tool constraints.
  - Minimal normalization — chosen.
- **Why**: Keeps preprocessing **retraceable** and avoids injecting strong assumptions. Splitting on punctuation matters because **sklearn stop words are whole tokens**: if commas stay glued to words (`pain,`), those tokens no longer match `pain` and stop-word filtering misbehaves.
- **Trade-offs**: Some linguistic variants remain (e.g., tense/plural), but TF‑IDF n-grams + dimensionality reduction partially compensate.
- **Where**: `normalise_text()` in `unsupervised/01_data_cleaning_eda.ipynb` (same logic in `unsupervised/cluster_testimonies.py`).

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

## Decision 3b — Stop words: sklearn list + a small custom extension

- **Decision**: Use `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` **union** a fixed set of short narrative fillers (`just`, `really`, `quite`, `also`, `though`, `maybe`, `somewhat`, `bit`, `even`, `still`). Pass the sorted list as `stop_words=` to `CountVectorizer` / `TfidfVectorizer` (do **not** rely on the string `"english"`, which is a different list in some versions).
- **Why**: Same behaviour in notebook 01 (EDA top tokens) and notebook 02 (modeling); aligns with the normalisation choice above.
- **Where**: `unsupervised/01_data_cleaning_eda.ipynb`, `unsupervised/02_vectorization_clustering.ipynb`, `unsupervised/cluster_testimonies.py`.

## Decision 3c — Train split for fitting TF‑IDF and LSA (no leakage from hold-out texts)

- **Decision**: Draw **20%** of row indices as a hold-out split (`train_test_split(..., test_size=0.2, random_state=42)`). **Fit** `TfidfVectorizer` and `TruncatedSVD` (LSA) **only on cleaned testimonies in the training split**. **Transform** the **full** dataset to build `X_tfidf` / `X_lsa` used for clustering and plots, so every testimony still receives a cluster assignment.
- **Why**: IDF and SVD directions should not be estimated using texts we conceptually reserved as hold-out; this mirrors a disciplined ML setup even without supervised labels.
- **Trade-offs**: Slightly fewer documents shape the vocabulary than fitting on all data; clustering remains on all points.
- **Where**: `unsupervised/02_vectorization_clustering.ipynb`; mirrored in `build_features()` in `unsupervised/cluster_testimonies.py`.

## Decision 4 — Primary representation: TF‑IDF with uni+bi-grams

- **Decision**: Use `TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1,2), min_df=2, max_df=0.90)` with `STOP_WORDS` as in Decision 3b, **fitted on the training split** (Decision 3c) before transforming all documents.
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

- **Decision**: Reduce TF‑IDF into a dense space with `TruncatedSVD(n_components=100)` then `Normalizer`, **fitted on TF‑IDF vectors of the training split only**, then applied to all documents (same leakage rationale as Decision 3c).
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
- **KMeans objective (“loss”)**: minimize **inertia** (sum of squared distances to centroids) on the **LSA vectors**. This is the quantity KMeans iteratively reduces while updating assignments/centroids.
- **Why it matters**: it explains what “training” means for KMeans and why we compare multiple runs/parameters.

## Decision 8 — Defense visualization: EDA, diagnostics, 2D projection + two colorings

- **Decision** (notebook 01): Beyond length histograms, export **length boxplots** and **id vs. word count** scatter when useful to show distribution and outliers.
- **Decision** (notebook 02): Plot **cumulative explained variance** of the LSA components (on the training fit) to justify using 100 components. Plot **silhouette vs. k** for KMeans on LSA to show how the chosen `k` relates to neighbouring values. Project features to 2D with `TruncatedSVD(n_components=2)` and plot:
  - points colored by **cluster assignment**
  - points colored by hidden **`color`**
- **Why**: The subject asks to use the hidden `color` to color the plot so an examiner can visually verify coherence at a glance. Extra diagnostics support the vectorization and `k` choices.
- **Trade-offs**:
  - 2D projections can distort distances; these plots are for **visual inspection**, not training decisions.
- **Where**: `unsupervised/01_data_cleaning_eda.ipynb`, `unsupervised/02_vectorization_clustering.ipynb`, PNGs under `unsupervised/figures/`.

## Final frozen choice (for defense)

- **Chosen pipeline (notebook)**: `TF-IDF (uni+bi-grams) → TruncatedSVD(100) + Normalizer → KMeans`, with **`k` selected as the KMeans-on-LSA run with highest silhouette** among `{8, 12, 16, 20, 24}` in the comparison table.
- **Why this family**:
  - Among our runs, **KMeans on LSA** dominates silhouette vs. KMeans on raw sparse TF‑IDF and vs. the agglomerative/DBSCAN grids we tried.
  - It is stable and easy to explain/defend compared to DBSCAN sensitivity to `eps`.
- **Evidence (from last executed notebook run — re-run cells if data or code change)**:
  - **Selected run name**: `kmeans_lsa24`
  - **silhouette** (euclidean on LSA): `0.082900`
  - **Calinski–Harabasz**: `14.141722`
  - **Davies–Bouldin**: `3.063446` (lower is better)
  - **Largest cluster sizes (top 10)**: `[76, 65, 64, 57, 56, 54, 52, 49, 45, 42]`
- **CLI vs. notebook**:
  - `unsupervised/cluster_testimonies.py` uses a **fixed `n_clusters=20`** for a simple one-command export and **does not** auto-read `best_k` from the notebook. The committed **`unsupervised_clusters.csv`** reflects that choice until you change the script to `n_clusters=24` (or your current `best_k`) and re-run.
- **Where**:
  - Model selection and plots: `unsupervised/02_vectorization_clustering.ipynb`
  - Batch CSV export: `unsupervised/cluster_testimonies.py`

## Outputs produced

- `unsupervised/cleaned_unsupervised_dataset.csv`: cleaned dataset used for modeling.
- `unsupervised/unsupervised_clusters.csv`: cluster assignments (plus `color` for visualization only).
- `unsupervised/figures/`: EDA plots (`eda_lengths_boxplot.png`, `eda_id_vs_nwords.png`, `eda_top_tokens.png`, …), LSA variance (`lsa_cumulative_variance.png`), silhouette vs. k (`kmeans_lsa_silhouette_vs_k.png`), rankings (`model_top_silhouette.png`, `model_cluster_sizes.png`), and 2D projections (by cluster and by hidden `color`).


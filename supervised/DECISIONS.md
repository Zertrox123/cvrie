# Supervised (CVRIE) – PneumoniaMNIST: decisions & justifications

Defense-oriented notes for the **medical image classification** track (official tools: scikit-learn, pandas, numpy, matplotlib, Jupyter).

## Dataset choice

- **PneumoniaMNIST** (MedMNIST family): public **binary** benchmark of **28×28 grayscale** chest X-ray–like patches with labels **normal vs. pneumonia**.
- **Subject fit**: images + labels only; clear **classification** setting; medical theme is easy to justify ethically and pedagogically.
- **Splits**: we keep the provided **`train` / `val` / `test`** NPZ-style exports (`{split}_images.npy`, `{split}_labels.npy`) so evaluation is **comparable** and **leak-resistant**.

## Pre-processing

- **Flatten** each image to a **784-dimensional** vector and cast to `float32`.
- **Scale** pixel values by **255** so intensities lie in **[0, 1]** (transparent normalisation, as required when inputs are raw `uint8`).
- **No `StandardScaler`**: features are already bounded; adding variance scaling offers little benefit for normalized pixels and would complicate the story without empirical gain in quick trials.

## Class imbalance

- Pneumonia benchmarks are often **skewed** toward the positive class (verify counts in `01_eda_preprocessing.ipynb`).
- We use **`class_weight='balanced'`** (where supported) so optimisation **up-weights** under-represented classes and **accuracy** is not the sole focus.
- **Defence**: always cite **precision / recall / F1** per class, **specificity / NPV for “normal”**, **confusion matrix**, and (if relevant) **balanced accuracy**, not only plain accuracy.

## Probability threshold (not always 0.5)

- With **`class_weight='balanced'`**, the default **`predict`** cut-off at 0.5 can **over-predict pneumonia** on some splits (many false alarms for “normal”).
- We **grid-search a single threshold** \(t\) on **validation** so we predict pneumonia iff **`P(pneumonia) ≥ t`**, choosing \(t\) to maximise **balanced accuracy** (mean of per-class recall). This uses a model **fit on `train` only** when evaluating `val`, so **no leakage** from `val` into the threshold choice.
- **CLI**: when using `--fit-on train_val`, the script still learns \(t\) from a **train-only** model on `val`, then refits on `train ∪ val` and applies the **same** \(t\) on `test` (same logic as the notebook).

## Models compared (notebook 02)

1. **Logistic regression** (`lbfgs`): minimises **logistic loss** (cross-entropy); **calibrated probabilities** by construction; strong baseline on flattened MNIST-like inputs.
2. **Random forest**: optimises **Gini impurity** per split; captures **non-linear** interactions; `predict_proba` enables **log loss** on validation/test for fair comparison.
3. **Linear SVM + calibration**: base learner minimises the **hinge** (margin) objective; **`CalibratedClassifierCV`** adds **probability estimates** so **log loss** is meaningful alongside the other candidates.

**Selection protocol**: fit each pipeline on **`train` only** for model comparison; on `val`, tune \(t\) and rank primarily by **validation balanced accuracy after tuning**; **log loss** is the first **tie-break** (better-calibrated probabilities among equally balanced models). **All** architectures are refit on **`train ∪ val`** and evaluated on **`test`** with their **own** validation-chosen \(t\); the artifact saves the **single winning** row from this table.

## Metrics & loss reporting

- **Log loss** and **Brier score** assess **probabilistic** quality; **reliability curves** (calibration plots) show whether predicted probabilities match empirical frequencies.
- **Precision / recall / F1**, **specificity**, **NPV** summarise errors for **each class** on imbalanced data.
- **MCC** (Matthews correlation) is a single balanced summary when comparing models on `test`.
- **Confusion matrix**: shows **false negatives** on pneumonia and **false positives** on normal at a glance.

## Deliverable script (`train_classify_pneumonia.py`)

- **Trains** the selected sklearn pipeline and **classifies** the official **`test`** split.
- Writes **`supervised_test_predictions.csv`** (`index`, `y_true`, `y_pred`, `p_normal`, `p_pneumonia`) and persists **`artifacts/trained_pipeline.joblib`** (pipeline, `pneumonia_probability_threshold`, metadata). Use **`--no-tune-threshold`** or **`--threshold`** to override the default val-tuned cut-off.
- **Errors** print to **stderr**; process exits with code **84** on failure (subject convention).

## Limitations (honest scope)

- **28×28** inputs discard fine anatomical detail; results benchmark **the dataset**, not clinical deployment.
- Tooling constraint forbids deep CNNs in PyTorch/TensorFlow; we stay within **sklearn** for reproducibility and compliance.

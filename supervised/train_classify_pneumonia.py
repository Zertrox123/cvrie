"""Train a sklearn classifier on PneumoniaMNIST and write test-set predictions (CVRIE supervised deliverable)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from pneumonia_evaluation import predict_from_threshold, tune_threshold_balanced_accuracy
from pneumonia_preprocessing import build_preprocess

ROOT_DIR = Path(__file__).resolve().parent.parent
SUPERVISED_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42
LABEL_NAMES = ("normal", "pneumonia")


def build_pipeline(model_name: str) -> Pipeline:
    pre = build_preprocess()
    if model_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "linear_svc_calibrated":
        clf = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", dual=False, random_state=RANDOM_STATE),
            cv=3,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    return Pipeline([("prep", pre), ("clf", clf)])


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    img_path = data_dir / f"{split}_images.npy"
    lab_path = data_dir / f"{split}_labels.npy"
    if not img_path.is_file() or not lab_path.is_file():
        raise FileNotFoundError(f"Missing {split} arrays under {data_dir}")
    X = np.load(img_path)
    y = np.load(lab_path).astype(np.int64).ravel()
    return X, y


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train PneumoniaMNIST classifier and classify the test split (supervised CVRIE)."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT_DIR / "pneumoniamnist"),
        help="Directory with train/val/test_{images,labels}.npy (default: <repo>/pneumoniamnist).",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=("logistic_regression", "random_forest", "linear_svc_calibrated"),
        default="logistic_regression",
        help="Classifier pipeline (default: logistic_regression — strong baseline, probabilities native).",
    )
    p.add_argument(
        "--fit-on",
        type=str,
        choices=("train", "train_val"),
        default="train_val",
        help="train = train only; train_val = concatenate official train+val before test (default: train_val).",
    )
    p.add_argument(
        "--model-out",
        type=str,
        default=str(SUPERVISED_DIR / "artifacts" / "trained_pipeline.joblib"),
        help="Path to write joblib payload (pipeline + metadata).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(SUPERVISED_DIR / "supervised_test_predictions.csv"),
        help="CSV path for test predictions (default: supervised/supervised_test_predictions.csv).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed P(pneumonia) threshold (0–1). Overrides --no-tune-threshold.",
    )
    p.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Use 0.5 cut-off instead of tuning on val for max balanced accuracy.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    model_out = Path(args.model_out).expanduser().resolve()
    output_csv = Path(args.output).expanduser().resolve()

    try:
        X_test, y_test = load_split(data_dir, "test")
        X_tr, y_tr = load_split(data_dir, "train")

        if args.fit_on == "train":
            X_fit, y_fit = X_tr, y_tr
        else:
            X_va, y_va = load_split(data_dir, "val")
            X_fit = np.concatenate([X_tr, X_va], axis=0)
            y_fit = np.concatenate([y_tr, y_va], axis=0)

        if args.threshold is not None:
            threshold = float(args.threshold)
        elif args.no_tune_threshold:
            threshold = 0.5
        else:
            X_va, y_va = load_split(data_dir, "val")
            tune_pipe = build_pipeline(args.model)
            tune_pipe.fit(X_tr, y_tr)
            proba_val = tune_pipe.predict_proba(X_va)
            threshold, _ = tune_threshold_balanced_accuracy(y_va, proba_val[:, 1])

        pipe = build_pipeline(args.model)
        pipe.fit(X_fit, y_fit)

        proba = pipe.predict_proba(X_test)
        pred = predict_from_threshold(proba, threshold)
        ll = log_loss(y_test, proba)
        report = classification_report(
            y_test, pred, target_names=list(LABEL_NAMES), digits=4, zero_division=0
        )

        model_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": args.model,
            "fit_on": args.fit_on,
            "labels": list(LABEL_NAMES),
            "pipeline": pipe,
            "pneumonia_probability_threshold": threshold,
            "test_log_loss": float(ll),
        }
        joblib.dump(payload, model_out)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(
                [
                    "index",
                    "y_true",
                    "y_pred",
                    "p_normal",
                    "p_pneumonia",
                ]
            )
            for i in range(len(y_test)):
                w.writerow(
                    [
                        i,
                        int(y_test[i]),
                        int(pred[i]),
                        float(proba[i, 0]),
                        float(proba[i, 1]),
                    ]
                )

        print(f"Wrote model: {model_out}")
        print(f"Wrote predictions: {output_csv}")
        print(f"P(pneumonia) threshold: {threshold:.5f}")
        print(f"Test log loss: {ll:.5f}")
        print(report)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 84

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

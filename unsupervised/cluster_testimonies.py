import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


HEX_COLOR_RE = re.compile(r"^0x[0-9A-Fa-f]{6}$")

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class TestimonyRow:
    id: int
    color: Optional[str]
    testimony_raw: str


def parse_raw_dataset(path: Path) -> pd.DataFrame:
    ids: list[int] = []
    colors: list[str | None] = []
    texts: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader: Iterable[list[str]] = csv.reader(f)
        for row in reader:
            if not row:
                continue

            try:
                current_id = int(str(row[0]).strip())
            except ValueError:
                continue

            fields = [str(x) for x in row[1:]]
            if not fields:
                continue

            if HEX_COLOR_RE.fullmatch((fields[0] or "").strip()):
                current_color = fields[0].strip()
                text_parts = fields[1:]
            else:
                current_color = None
                text_parts = fields

            text = ",".join(text_parts).strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
                text = text[1:-1].strip()

            if not text:
                continue

            ids.append(current_id)
            colors.append(current_color)
            texts.append(text)

    if not ids:
        raise ValueError(f"No valid rows found in {path}")

    return pd.DataFrame({"id": ids, "color": colors, "testimony_raw": texts})


def normalise_text(text: str) -> str:
    cleaned = str(text).lower()
    cleaned = (
        cleaned.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_EXTRA_STOPS = frozenset(
    {"just", "really", "quite", "also", "though", "maybe", "somewhat", "bit", "even", "still"}
)
STOP_WORDS = sorted(frozenset(ENGLISH_STOP_WORDS) | _EXTRA_STOPS)

TEST_SIZE = 0.2
RNG = 42


def build_features(texts: list[str]) -> tuple[object, np.ndarray]:
    # Frozen, defense-friendly parameters (see unsupervised/DECISIONS.md).
    # Fit TF-IDF vocabulary + LSA on a train split only; transform all texts (matches notebook 02).
    idx = np.arange(len(texts))
    train_idx, _ = train_test_split(idx, test_size=TEST_SIZE, random_state=RNG)
    texts_train = [texts[i] for i in train_idx]

    tfidf = TfidfVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
    )
    tfidf.fit(texts_train)
    X_tfidf = tfidf.transform(texts)

    lsa = make_pipeline(
        TruncatedSVD(n_components=100, random_state=42),
        Normalizer(copy=False),
    )
    X_tfidf_train = tfidf.transform(texts_train)
    lsa.fit(X_tfidf_train)
    X_lsa = lsa.transform(X_tfidf)
    return (tfidf, lsa), X_lsa


def cluster_lsa(X_lsa: np.ndarray) -> np.ndarray:
    model = KMeans(n_clusters=20, n_init=20, random_state=42)
    return model.fit_predict(X_lsa)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster CVRIE testimonies (unsupervised).")
    p.add_argument(
        "--input",
        type=str,
        default=str(ROOT_DIR / "Student_Dataset.csv"),
        help="Path to raw dataset CSV (default: <repo>/Student_Dataset.csv).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "unsupervised" / "unsupervised_clusters.csv"),
        help="Path to write clustering output CSV (default: <repo>/unsupervised/unsupervised_clusters.csv).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_raw = parse_raw_dataset(input_path)
        df = df_raw.copy()
        df["testimony_clean"] = df["testimony_raw"].map(normalise_text)
        df = df[df["testimony_clean"].str.len() > 0].drop_duplicates(subset=["testimony_clean"]).reset_index(drop=True)

        texts = df["testimony_clean"].astype(str).tolist()
        _, X_lsa = build_features(texts)
        labels = cluster_lsa(X_lsa)

        # Subject compliance: keep `color` only for visualization, never as training input.
        out = df[["id", "color", "testimony_raw"]].copy()
        out["cluster"] = labels
        out.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")
        print(f"Wrote: {output_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 84

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


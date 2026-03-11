import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


HEX_COLOR_RE = re.compile(r"^0x[0-9A-Fa-f]{6}$")


@dataclass
class TestimonyRow:
    """Structured representation of one line from the raw CSV."""

    id: int
    color: Optional[str]
    testimony_raw: str


def _parse_csv_line(fields: List[str]) -> Optional[TestimonyRow]:
    """
    Parse a single CSV row from the raw dataset.

    Format (heterogeneous in the provided file):
    - Most lines:    id, text...
    - Many lines:    id, 0xXXXXXX, text...

    We detect an optional hex color code as the second field when it matches
    the pattern 0xRRGGBB, and treat everything after it as the free text.
    """
    if not fields:
        return None

    try:
        identifier = int(fields[0])
    except ValueError:
        # This could be a header or malformed line – we skip it.
        return None

    # Nothing but the id – invalid for our use case
    if len(fields) == 1:
        return None

    remaining = fields[1:]

    color: Optional[str] = None
    text_parts: List[str]

    # Case: id, 0xXXXXXX, text...
    if remaining and HEX_COLOR_RE.fullmatch(remaining[0] or ""):
        color = remaining[0]
        text_parts = remaining[1:]
    else:
        # Case: id, text...
        text_parts = remaining

    text = ",".join(text_parts).strip()
    # Strip surrounding quotes if present
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()

    if not text:
        return None

    return TestimonyRow(id=identifier, color=color, testimony_raw=text)


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """
    Load the provided unsupervised dataset and return a DataFrame with
    columns: id, color, testimony_raw.
    """
    rows: List[TestimonyRow] = []

    with path.open(newline="", encoding="utf-8") as csvfile:
        reader: Iterable[List[str]] = csv.reader(csvfile)
        for raw_fields in reader:
            parsed = _parse_csv_line(list(raw_fields))
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise ValueError("No valid rows were found in the dataset.")

    return pd.DataFrame(
        {
            "id": [r.id for r in rows],
            "color": [r.color for r in rows],
            "testimony_raw": [r.testimony_raw for r in rows],
        }
    )


def _normalise_text(text: str) -> str:
    """
    Basic text normalisation for testimonies.

    This step prepares the text for later vectorisation (e.g. TF‑IDF) without
    making modelling assumptions. You can extend it in your notebook if needed.
    """
    # Lowercase
    cleaned = text.lower()

    # Normalise common unicode punctuation
    cleaned = (
        cleaned.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    # Keep letters, digits, basic punctuation; replace everything else with space
    cleaned = re.sub(r"[^a-z0-9\s'.,;:?!-]", " ", cleaned)

    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_dataset(df: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
    """
    Apply cleaning operations:
    - build a `testimony_clean` column
    - drop empty or too-short entries
    - drop exact duplicates on the cleaned text
    """
    if "testimony_raw" not in df.columns:
        raise ValueError("Expected a 'testimony_raw' column in the input DataFrame.")

    clean_df = df.copy()
    clean_df["testimony_clean"] = clean_df["testimony_raw"].astype(str).map(_normalise_text)

    # Remove entries that became empty after cleaning
    clean_df = clean_df[clean_df["testimony_clean"].str.len() > 0]

    # Optionally remove very short testimonies that are unlikely to be informative
    if min_length > 0:
        clean_df = clean_df[clean_df["testimony_clean"].str.len() >= min_length]

    # Remove duplicates based on cleaned text
    clean_df = clean_df.drop_duplicates(subset=["testimony_clean"])

    clean_df = clean_df.reset_index(drop=True)

    return clean_df


def preprocess_file(input_path: Path, output_path: Path, min_length: int = 5) -> None:
    """
    High‑level helper: load, clean and save the dataset.
    """
    df_raw = load_raw_dataset(input_path)
    df_clean = clean_dataset(df_raw, min_length=min_length)

    # We keep `color` in the saved file so that it can be used for plotting
    # during the defense, but it must not be fed to the clustering algorithm.
    df_clean.to_csv(output_path, index=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the unsupervised CVRIE dataset "
            "(patient testimonies) for clustering."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Student_Dataset.csv",
        help="Path to the raw dataset CSV (default: Student_Dataset.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cleaned_unsupervised_dataset.csv",
        help="Path where the cleaned CSV will be written "
        "(default: cleaned_unsupervised_dataset.csv).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum length (number of characters) for a cleaned testimony "
        "to be kept (default: 5).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        preprocess_file(input_path, output_path, min_length=args.min_length)
    except Exception as exc:  # noqa: BLE001 – top-level CLI handler
        print(f"Error during preprocessing: {exc}", file=sys.stderr)
        return 84

    return 0


if __name__ == "__main__":
    sys.exit(main())


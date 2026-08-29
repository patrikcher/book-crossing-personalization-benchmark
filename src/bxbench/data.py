"""Loaders for the raw Book-Crossing CSVs.

Schema notes:
- BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv are ';'-delimited with quoted
  fields, encoded latin-1 (the original Book-Crossing dump).
- Books_Descriptions.csv is a separate, comma-delimited, utf-8 file joined by ISBN.
  It only covers 7,021 / 271,379 books (2.6%).
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_users(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "BX-Users.csv",
        sep=";",
        encoding="latin-1",
        dtype={"User-ID": "int64"},
    )
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df


def load_books(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "BX-Books.csv",
        sep=";",
        encoding="latin-1",
        dtype=str,
        on_bad_lines="skip",
    )
    df["Year-Of-Publication"] = pd.to_numeric(df["Year-Of-Publication"], errors="coerce")
    return df


def load_ratings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "BX-Book-Ratings.csv",
        sep=";",
        encoding="latin-1",
        dtype={"User-ID": "int64", "ISBN": str, "Book-Rating": "int64"},
    )
    return df


def load_descriptions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "Books_Descriptions.csv", encoding="utf-8")
    cols = [c for c in df.columns if c.lower() != "unnamed: 0"]
    return df[cols]


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {
        "users": load_users(raw_dir),
        "books": load_books(raw_dir),
        "ratings": load_ratings(raw_dir),
        "descriptions": load_descriptions(raw_dir),
    }

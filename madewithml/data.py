"""
data.py
Data loading, cleaning, and preprocessing functions.
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from madewithml.config import logger


# ── Constants ──────────────────────────────────────────────────────────────────
ACCEPTED_TAGS = [
    "natural-language-processing",
    "computer-vision",
    "mlops",
    "graph-learning",
]


def load_data(dataset_loc: str) -> pd.DataFrame:
    """Load CSV dataset from disk."""
    logger.info(f"Loading data from {dataset_loc}")
    df = pd.read_csv(dataset_loc)
    logger.info(f"Loaded {len(df)} rows")
    return df


def clean_text(text: str) -> str:
    """Lowercase and strip special characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame, accepted_tags: list = ACCEPTED_TAGS) -> pd.DataFrame:
    """Filter to accepted tags, clean text, drop nulls."""
    logger.info("Preprocessing data ...")
    df = df[df["tag"].isin(accepted_tags)].copy()
    df.dropna(subset=["title", "description", "tag"], inplace=True)
    df["text"] = df["title"] + " " + df["description"]
    df["text"] = df["text"].apply(clean_text)
    logger.info(f"After preprocessing: {len(df)} rows")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test sets."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["tag"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        stratify=train_df["tag"],
        random_state=seed,
    )
    logger.info(
        f"Split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df

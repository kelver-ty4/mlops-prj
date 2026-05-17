"""
tests/test_data.py
Unit tests for data loading and preprocessing functions.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "madewithml"))

from data import clean_text, preprocess, split_data, ACCEPTED_TAGS


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Minimal dataframe that mimics the real CSV structure."""
    rows = []
    for i, tag in enumerate(ACCEPTED_TAGS * 10):   # 40 rows, balanced
        rows.append({
            "title":       f"Project {i}",
            "description": f"This project is about {tag} and deep learning",
            "tag":         tag,
        })
    return pd.DataFrame(rows)


# ── clean_text ────────────────────────────────────────────────────────────────
def test_clean_text_lowercases():
    assert clean_text("Hello World") == "hello world"


def test_clean_text_removes_special_chars():
    result = clean_text("NLP! Is @great#")
    assert "!" not in result
    assert "@" not in result
    assert "#" not in result


def test_clean_text_strips_extra_spaces():
    assert clean_text("  too   many   spaces  ") == "too many spaces"


def test_clean_text_empty_string():
    assert clean_text("") == ""


# ── preprocess ────────────────────────────────────────────────────────────────
def test_preprocess_filters_unknown_tags(sample_df):
    sample_df.loc[0, "tag"] = "unknown-tag"
    result = preprocess(sample_df.copy())
    assert "unknown-tag" not in result["tag"].values


def test_preprocess_creates_text_column(sample_df):
    result = preprocess(sample_df.copy())
    assert "text" in result.columns


def test_preprocess_drops_nulls():
    df = pd.DataFrame([
        {"title": None, "description": "desc", "tag": ACCEPTED_TAGS[0]},
        {"title": "ok",  "description": "desc", "tag": ACCEPTED_TAGS[0]},
    ])
    result = preprocess(df)
    assert len(result) == 1


def test_preprocess_returns_only_accepted_tags(sample_df):
    result = preprocess(sample_df.copy())
    assert set(result["tag"].unique()).issubset(set(ACCEPTED_TAGS))


# ── split_data ────────────────────────────────────────────────────────────────
def test_split_data_sizes(sample_df):
    df              = preprocess(sample_df.copy())
    train, val, test = split_data(df)
    total = len(train) + len(val) + len(test)
    assert total == len(df)


def test_split_data_no_overlap(sample_df):
    df              = preprocess(sample_df.copy())
    train, val, test = split_data(df)
    train_idx = set(train.index)
    val_idx   = set(val.index)
    test_idx  = set(test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_data_reproducible(sample_df):
    df = preprocess(sample_df.copy())
    train1, _, _ = split_data(df, seed=42)
    train2, _, _ = split_data(df, seed=42)
    assert list(train1.index) == list(train2.index)

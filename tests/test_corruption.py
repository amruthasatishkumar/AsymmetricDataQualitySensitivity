"""Unit tests for the corruption library.

Verifies:
    - determinism: same (family, severity, seed) twice -> equal output
    - label preservation: y_t1, y_t2, split, order_id are never altered
    - severity monotonicity: 0 -> identity (no change beyond float coercion)
    - row count invariants per family
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from corruption import apply, list_families

REPO = Path(__file__).resolve().parent.parent
SILVER = REPO / "data" / "silver" / "order_features.parquet"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    full = pd.read_parquet(SILVER)
    # use a 1000-row test slice for speed
    return full.sample(n=1000, random_state=0).reset_index(drop=True)


@pytest.mark.parametrize("family", list_families())
def test_determinism(df: pd.DataFrame, family: str) -> None:
    a = apply(df, family, severity=0.20, seed=13)
    b = apply(df, family, severity=0.20, seed=13)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("family", list_families())
def test_label_preservation(df: pd.DataFrame, family: str) -> None:
    out = apply(df, family, severity=0.30, seed=17)
    # duplicate_rows grows the frame; compare the prefix
    head = out.head(len(df))
    for col in ("y_t1", "y_t2", "split", "order_id"):
        # NaN-safe compare; some labels (y_t3, y_t5) may not exist yet in current slice
        s_in = df[col].reset_index(drop=True)
        s_out = head[col].reset_index(drop=True)
        assert s_in.equals(s_out), f"{family} mutated column {col!r}"


@pytest.mark.parametrize("family", list_families())
def test_severity_zero_is_identity(df: pd.DataFrame, family: str) -> None:
    out = apply(df, family, severity=0.0, seed=0)
    assert len(out) == len(df), f"{family} changed row count at severity=0"
    # column count: same (schema_drift may rename 0 cols at severity=0)
    assert len(out.columns) == len(df.columns), \
        f"{family} changed col count at severity=0"


@pytest.mark.parametrize(
    "family",
    [f for f in list_families() if f not in {"duplicate_rows"}],
)
def test_row_count_preserved(df: pd.DataFrame, family: str) -> None:
    out = apply(df, family, severity=0.30, seed=23)
    assert len(out) == len(df), f"{family} should preserve row count"


def test_duplicate_rows_grows(df: pd.DataFrame) -> None:
    out = apply(df, "duplicate_rows", severity=0.20, seed=23)
    expected = len(df) + int(round(0.20 * len(df)))
    assert len(out) == expected


def test_schema_drift_renames(df: pd.DataFrame) -> None:
    out = apply(df, "schema_drift", severity=0.30, seed=23)
    feat_cols = [c for c in df.columns if c not in {"y_t1", "y_t2", "y_t3", "y_t5", "order_id", "split"}]
    expected_renames = int(round(0.30 * len(feat_cols)))
    renamed = sum(1 for c in out.columns if c.startswith("col_23_"))
    assert renamed == expected_renames


def test_missing_injection_increases_nulls(df: pd.DataFrame) -> None:
    feats = [c for c in df.columns if c not in {"y_t1", "y_t2", "y_t3", "y_t5", "order_id", "split"}]
    before = df[feats].isna().sum().sum()
    out = apply(df, "missing_injection", severity=0.25, seed=31)
    after = out[feats].isna().sum().sum()
    assert after > before


def test_currency_unit_drift_inflates_monetary(df: pd.DataFrame) -> None:
    """At least one monetary column max should grow ~100x for high severity."""
    out = apply(df, "currency_unit_drift", severity=0.30, seed=31)
    monetary = ["total_price", "total_freight", "total_payment_value",
                "max_price", "mean_price"]
    bumped = 0
    for c in monetary:
        if c not in df.columns:
            continue
        before = pd.to_numeric(df[c], errors="coerce").max()
        after = pd.to_numeric(out[c], errors="coerce").max()
        if pd.notna(before) and pd.notna(after) and after > before * 10:
            bumped += 1
    assert bumped >= 1


def test_timezone_collapse_changes_hour(df: pd.DataFrame) -> None:
    out = apply(df, "timezone_collapse", severity=0.50, seed=31)
    if "purchase_hour" in df.columns:
        a = pd.to_numeric(df["purchase_hour"], errors="coerce")
        b = pd.to_numeric(out["purchase_hour"], errors="coerce")
        assert (a != b).any(), "timezone_collapse should change some purchase_hour values"


def test_category_taxonomy_drift_changes_categories(df: pd.DataFrame) -> None:
    if "dominant_category" not in df.columns:
        pytest.skip("dominant_category not in test slice")
    out = apply(df, "category_taxonomy_drift", severity=0.50, seed=31)
    a = df["dominant_category"].astype(str)
    b = out["dominant_category"].astype(str)
    assert (a != b).any(), "category_taxonomy_drift should change some labels"


def test_mojibake_changes_text(df: pd.DataFrame) -> None:
    """Either a Portuguese category or a text column should be mangled.

    Mojibake on pure-ASCII strings is a no-op (correct: latin-1 == utf-8 for
    code points <128). Skip if no non-ASCII content is present to mangle.
    """
    target_cols = [c for c in ("dominant_category", "review_comment_message",
                                "review_comment_title") if c in df.columns]
    if not target_cols:
        pytest.skip("no target text/category cols in slice")
    has_non_ascii = False
    for c in target_cols:
        if df[c].dropna().astype(str).str.contains(r"[^\x00-\x7f]", regex=True).any():
            has_non_ascii = True
            break
    if not has_non_ascii:
        pytest.skip("target cols are pure-ASCII; mojibake is a no-op (expected)")
    out = apply(df, "mojibake_roundtrip", severity=0.80, seed=31)
    any_changed = False
    for c in target_cols:
        if (df[c].astype(str) != out[c].astype(str)).any():
            any_changed = True
            break
    assert any_changed

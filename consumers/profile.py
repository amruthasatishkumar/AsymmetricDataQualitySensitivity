"""Compact training-set profile generator and sample-row utility.

Produces a small text profile used by the agent prompt. We deliberately avoid
sending a full ydata-profiling HTML report — it is too large for context windows
and noisy. Instead we produce a short markdown-ish block with per-column stats.

The profile is built ONCE per task on the training split and cached to
configs/profile_<task>.txt for reproducibility.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SILVER = REPO / "data" / "silver"
CONFIGS = REPO / "configs"

NUMERIC_TEMPLATE = (
    "{col}: numeric  n={n}  miss={miss:.1%}  "
    "mean={mean:.3g}  std={std:.3g}  "
    "p10={p10:.3g}  p50={p50:.3g}  p90={p90:.3g}  "
    "min={mn:.3g}  max={mx:.3g}  "
    "corr_y={corr:+.3f}"
)

CATEGORICAL_TEMPLATE = (
    "{col}: categorical  n={n}  miss={miss:.1%}  "
    "unique={uniq}  top={top}  prev_y_top={py:.3f}"
)


def _profile_numeric(s: pd.Series, y: pd.Series) -> str:
    s_num = pd.to_numeric(s, errors="coerce")
    n = int(s_num.notna().sum())
    miss = float(s_num.isna().mean())
    if n == 0:
        return f"{s.name}: numeric  n=0  miss=100%  (empty)"
    valid = s_num.dropna()
    aligned_y = y[valid.index]
    try:
        corr = float(np.corrcoef(valid, aligned_y)[0, 1])
    except Exception:
        corr = float("nan")
    return NUMERIC_TEMPLATE.format(
        col=s.name, n=n, miss=miss,
        mean=float(valid.mean()), std=float(valid.std()),
        p10=float(valid.quantile(0.10)), p50=float(valid.quantile(0.50)), p90=float(valid.quantile(0.90)),
        mn=float(valid.min()), mx=float(valid.max()),
        corr=0.0 if np.isnan(corr) else corr,
    )


def _profile_categorical(s: pd.Series, y: pd.Series, max_levels: int = 3) -> str:
    s = s.astype("object")
    n = int(s.notna().sum())
    miss = float(s.isna().mean())
    vc = s.value_counts(dropna=True)
    uniq = int(vc.shape[0])
    if uniq == 0:
        return f"{s.name}: categorical  n=0  miss=100%  (empty)"
    top_levels = vc.head(max_levels).index.tolist()
    top_str = ", ".join(f"{lvl}={vc[lvl]}" for lvl in top_levels)
    top_mask = s == top_levels[0]
    py = float(y[top_mask].mean()) if top_mask.any() else float("nan")
    return CATEGORICAL_TEMPLATE.format(
        col=s.name, n=n, miss=miss, uniq=uniq, top=top_str,
        py=0.0 if np.isnan(py) else py,
    )


def make_profile(task: str, feature_cols: list[str], categorical_cols: set[str]) -> str:
    df = pd.read_parquet(SILVER / "order_features.parquet")
    target = "y_t1" if task == "t1" else "y_t2"
    sub = df[(df["split"] == "train") & df[target].notna()].copy()
    y = sub[target].astype(int)

    header = (
        f"Training-set profile for task={task}  "
        f"n_train={len(sub):,}  "
        f"prevalence(y=1)={float(y.mean()):.4f}\n"
    )
    lines = [header]
    for c in feature_cols:
        if c not in sub.columns:
            continue
        if c in categorical_cols:
            lines.append(_profile_categorical(sub[c], y))
        else:
            lines.append(_profile_numeric(sub[c], y))
    return "\n".join(lines)


def get_or_make_profile(
    task: str, feature_cols: list[str], categorical_cols: set[str], rebuild: bool = False
) -> str:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    cache = CONFIGS / f"profile_{task}.txt"
    if cache.exists() and not rebuild:
        return cache.read_text(encoding="utf-8")
    txt = make_profile(task, feature_cols, categorical_cols)
    cache.write_text(txt, encoding="utf-8")
    return txt


if __name__ == "__main__":
    from consumers.features import T1_EXTRA_FEATURE_COLS, T2_FEATURE_COLS
    cats = {"customer_state", "payment_type_modal", "dominant_category", "order_status"}
    for task, feats in (("t2", T2_FEATURE_COLS),
                        ("t1", T2_FEATURE_COLS + T1_EXTRA_FEATURE_COLS)):
        p = get_or_make_profile(task, feats, cats, rebuild=True)
        print(f"--- profile_{task} ({len(p)} chars) ---")
        print(p[:500] + ("..." if len(p) > 500 else ""))
        print()

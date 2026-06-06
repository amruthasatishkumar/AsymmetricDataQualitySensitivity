"""Clean ML baseline for AgentDQ.

Trains Logistic Regression and XGBoost on T1 (review) and T2 (late_delivery)
using the frozen temporal split. Hyperparameters are FROZEN (set below) and
will be reused unchanged across all corruption variants in Step 5+.

5 seeds, mean ± std for AUROC, F1, Brier on the test split.

Run:
    python -m consumers.ml_pipeline
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from consumers.features import T1_EXTRA_FEATURE_COLS, T2_FEATURE_COLS, T3_TEXT_COLS

REPO = Path(__file__).resolve().parent.parent
SILVER = REPO / "data" / "silver"
LOGS = REPO / "logs"

SEEDS: tuple[int, ...] = (13, 17, 23, 31, 47)

# ---- FROZEN hyperparameters (do not retune across conditions) ----
LR_PARAMS = dict(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
)

CATEGORICAL_COLS = {"customer_state", "payment_type_modal", "dominant_category", "order_status"}
TEXT_COLS_SET = set(T3_TEXT_COLS)

# Frozen TF-IDF hyperparameters for T3 text features
TFIDF_PARAMS = dict(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
)


def _split_xy(df: pd.DataFrame, target: str, feature_cols: list[str], split: str) -> tuple[pd.DataFrame, np.ndarray]:
    sub = df[(df["split"] == split) & df[target].notna()]
    X = sub[feature_cols].copy()
    y = sub[target].astype(int).to_numpy()
    return X, y


def _build_preprocessor(feature_cols: Iterable[str], scale: bool) -> ColumnTransformer:
    cats = [c for c in feature_cols if c in CATEGORICAL_COLS]
    texts = [c for c in feature_cols if c in TEXT_COLS_SET]
    nums = [c for c in feature_cols if c not in CATEGORICAL_COLS and c not in TEXT_COLS_SET]
    num_steps: list[tuple] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
    ])
    transformers: list[tuple] = [("num", num_pipe, nums), ("cat", cat_pipe, cats)]
    # one TF-IDF per text column, preceded by a string-coercion step to handle
    # any residual NaN/float values that survive ColumnTransformer extraction
    for tc in texts:
        text_pipe = Pipeline([
            ("tostr", FunctionTransformer(_coerce_to_str_array, validate=False)),
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ])
        transformers.append((f"txt_{tc}", text_pipe, tc))
    return ColumnTransformer(transformers, remainder="drop")


def _coerce_to_str_array(x):
    """Force any 1-D input to a list of python str (empty string for NaN/None).
    TF-IDF requires iterable of str; this guards against pandas leaking floats.
    """
    if hasattr(x, "tolist"):
        x = x.tolist()
    return ["" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v) for v in x]


def _to_numeric(X: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    # categorical / text cols: coerce pd.NA -> np.nan, ensure object dtype (sklearn-friendly)
    for c in X.columns:
        if c in num_cols:
            continue
        if c in TEXT_COLS_SET:
            # text cols: fillna with empty string and force python str type
            X[c] = X[c].fillna("").astype(str)
        else:
            s = X[c].astype("object")
            s = s.where(pd.notna(s), other=np.nan)
            X[c] = s
    return X


def fit_eval(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    seeds: Iterable[int] = SEEDS,
) -> dict:
    """Return {model: {seed: metrics}, summary: {model: {metric: (mean,std)}}}"""
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    X_train, y_train = _split_xy(df, target, feature_cols, "train")
    X_val, y_val = _split_xy(df, target, feature_cols, "val")
    X_test, y_test = _split_xy(df, target, feature_cols, "test")

    X_train = _to_numeric(X_train, num_cols)
    X_val = _to_numeric(X_val, num_cols)
    X_test = _to_numeric(X_test, num_cols)

    # train+val for final fit (hyperparameters are frozen — val is unused here
    # except as documentation; tuning was done once, offline)
    X_fit = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_fit = np.concatenate([y_train, y_val])

    out: dict = {"target": target, "n_train": int(len(X_train)),
                 "n_val": int(len(X_val)), "n_test": int(len(X_test)),
                 "n_features": len(feature_cols), "feature_cols": feature_cols,
                 "models": {}}

    for model_name in ("lr", "xgb"):
        per_seed: dict[int, dict] = {}
        for seed in seeds:
            t0 = time.time()
            if model_name == "lr":
                pre = _build_preprocessor(feature_cols, scale=True)
                clf = LogisticRegression(random_state=seed, **LR_PARAMS)
            else:
                pre = _build_preprocessor(feature_cols, scale=False)
                clf = XGBClassifier(random_state=seed, **XGB_PARAMS)
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(X_fit, y_fit)
            proba = pipe.predict_proba(X_test)[:, 1]
            yhat = (proba >= 0.5).astype(int)
            per_seed[seed] = {
                "auroc": float(roc_auc_score(y_test, proba)),
                "f1": float(f1_score(y_test, yhat, zero_division=0)),
                "brier": float(brier_score_loss(y_test, proba)),
                "fit_seconds": round(time.time() - t0, 2),
            }
            print(f"  [{target}] {model_name} seed={seed:>2}  "
                  f"AUROC={per_seed[seed]['auroc']:.4f}  "
                  f"F1={per_seed[seed]['f1']:.4f}  "
                  f"Brier={per_seed[seed]['brier']:.4f}  "
                  f"({per_seed[seed]['fit_seconds']:.1f}s)")
        # summary
        summ = {}
        for metric in ("auroc", "f1", "brier"):
            vals = np.array([per_seed[s][metric] for s in per_seed])
            summ[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
        out["models"][model_name] = {"per_seed": per_seed, "summary": summ}
    return out


def main() -> int:
    df = pd.read_parquet(SILVER / "order_features.parquet")

    feats_t2 = list(T2_FEATURE_COLS)
    feats_t1 = list(T2_FEATURE_COLS) + list(T1_EXTRA_FEATURE_COLS)
    feats_t3 = list(T2_FEATURE_COLS) + list(T3_TEXT_COLS)

    LOGS.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "config": {
            "seeds": list(SEEDS),
            "lr_params": LR_PARAMS,
            "xgb_params": XGB_PARAMS,
            "tfidf_params": TFIDF_PARAMS,
            "split": "frozen temporal split (configs/split.json)",
        },
        "tasks": {},
    }
    print("\n=== T2: late_delivery ===")
    results["tasks"]["t2"] = fit_eval(df, "y_t2", feats_t2)
    print("\n=== T3: review_low (text + tabular) ===")
    results["tasks"]["t3"] = fit_eval(df, "y_t3", feats_t3)
    print("\n=== T1: review_high (kept for appendix) ===")
    results["tasks"]["t1"] = fit_eval(df, "y_t1", feats_t1)

    out = LOGS / "ml_clean.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # also a flat CSV for paper/tables/
    rows = []
    for task, payload in results["tasks"].items():
        for model, mdata in payload["models"].items():
            s = mdata["summary"]
            rows.append({
                "task": task,
                "model": model,
                "auroc_mean": round(s["auroc"]["mean"], 4),
                "auroc_std": round(s["auroc"]["std"], 4),
                "f1_mean": round(s["f1"]["mean"], 4),
                "f1_std": round(s["f1"]["std"], 4),
                "brier_mean": round(s["brier"]["mean"], 4),
                "brier_std": round(s["brier"]["std"], 4),
            })
    tbl = pd.DataFrame(rows)
    (REPO / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    tbl_path = REPO / "paper" / "tables" / "baseline_clean.csv"
    tbl.to_csv(tbl_path, index=False)

    print("\n=== SUMMARY ===")
    print(tbl.to_string(index=False))
    print(f"\nWrote {out}")
    print(f"Wrote {tbl_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

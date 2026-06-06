"""Pre-experiment GATE for AgentDQ (Step 5 in plan).

Runs ML and the LLM agent on the SAME 200 stratified test rows under:
    - clean
    - corruption variants from a small grid

Computes per-condition AUROC and the delta vs clean. Gate passes if for at
least one corruption (family, seed), the absolute ML-vs-agent loss-delta ratio
exceeds 2 in either direction:

    |delta_agent / delta_ml| > 2     OR     |delta_ml / delta_agent| > 2

If the gate fails, the full sweep would not produce a publishable signal and
the scope must be revisited before burning the LLM budget.

Run:
    python -m eval.gate
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from consumers.agent import Agent, AgentConfig, _row_dict
from consumers.features import T1_EXTRA_FEATURE_COLS, T2_FEATURE_COLS, T3_TEXT_COLS
from consumers.ml_pipeline import CATEGORICAL_COLS, fit_eval as _fit_eval_unused, _to_numeric  # noqa: F401
from consumers.ml_pipeline import (LR_PARAMS, XGB_PARAMS, _build_preprocessor)
from corruption import apply
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
SILVER = REPO / "data" / "silver" / "order_features.parquet"
LOGS = REPO / "logs"

# Gate config
TASK = "t3"
N_TEST_ROWS = 60
GATE_SEED_FOR_SAMPLE = 13
GATE_CORRUPTIONS = [
    # Mix of canonical (kept) and new realistic corruptions.
    # New families are most relevant for T3 since they hit text & categories.
    ("missing_injection", 0.20, 13),
    ("missing_injection", 0.20, 17),
    ("mojibake_roundtrip", 0.30, 13),
    ("mojibake_roundtrip", 0.30, 17),
    ("category_taxonomy_drift", 0.30, 13),
    ("currency_unit_drift", 0.20, 13),
]


def _train_xgb(df: pd.DataFrame, target: str, feature_cols: list[str], seed: int = 13) -> Pipeline:
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    train = df[(df["split"].isin(["train", "val"])) & df[target].notna()]
    X = _to_numeric(train[feature_cols].copy(), num_cols)
    y = train[target].astype(int).to_numpy()
    pre = _build_preprocessor(feature_cols, scale=False)
    clf = XGBClassifier(random_state=seed, **XGB_PARAMS)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


def _stratified_test_indices(df: pd.DataFrame, target: str, n: int, seed: int) -> np.ndarray:
    pool = df[(df["split"] == "test") & df[target].notna()]
    half = n // 2
    pos = pool[pool[target] == 1].sample(n=half, random_state=seed)
    neg = pool[pool[target] == 0].sample(n=n - half, random_state=seed)
    chosen = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
    return chosen.index.to_numpy()


def _ml_auc_on(pipe: Pipeline, df_corrupted: pd.DataFrame, idxs: np.ndarray,
               target: str, feature_cols: list[str]) -> float:
    sub = df_corrupted.loc[idxs]
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    # if schema_drift renamed columns, missing ones become NaN
    X = sub.reindex(columns=feature_cols)
    X = _to_numeric(X, num_cols)
    y = sub[target].astype(int).to_numpy()
    proba = pipe.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def _agent_auc_on(agent: Agent, df_corrupted: pd.DataFrame, idxs: np.ndarray,
                  target: str) -> tuple[float, dict]:
    """Run agent on each row; return AUROC of confidence-based score + stats."""
    sub = df_corrupted.loc[idxs]
    y_true = sub[target].astype(int).to_numpy()
    scores: list[float] = []
    preds: list[int] = []
    parse_ok = 0
    for _, row in sub.iterrows():
        d = _row_dict(row, agent.feature_cols)
        r = agent.predict_one(d)
        p = r["prediction"]
        c = r["confidence"]
        if p in (0, 1):
            parse_ok += 1
            score = c if p == 1 else 1.0 - c
        else:
            # parse fail -> neutral score
            score = 0.5
        scores.append(score)
        preds.append(p if p in (0, 1) else 0)
    auc = float(roc_auc_score(y_true, np.array(scores)))
    return auc, {
        "parse_ok": parse_ok,
        "n": int(len(y_true)),
        "tpr": float(((np.array(preds) == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())),
        "tnr": float(((np.array(preds) == 0) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())),
    }


def main() -> int:
    t0 = time.time()
    if TASK == "t1":
        feature_cols = list(T2_FEATURE_COLS) + list(T1_EXTRA_FEATURE_COLS)
    elif TASK == "t3":
        feature_cols = list(T2_FEATURE_COLS) + list(T3_TEXT_COLS)
    else:
        feature_cols = list(T2_FEATURE_COLS)
    target = {"t1": "y_t1", "t2": "y_t2", "t3": "y_t3"}[TASK]

    print(f"=== AgentDQ GATE (task={TASK}, mode=a1, n_test={N_TEST_ROWS}) ===\n")

    df = pd.read_parquet(SILVER)
    idxs = _stratified_test_indices(df, target, N_TEST_ROWS, seed=GATE_SEED_FOR_SAMPLE)
    print(f"Selected {len(idxs)} test rows (stratified 50/50)")

    print("\n[ML] training XGBoost on clean train+val...")
    pipe = _train_xgb(df, target, feature_cols)
    print("    done.")

    print("\n[Agent] booting A1 agent...")
    agent = Agent(AgentConfig(task=TASK, log_tag="gate"))
    print(f"    log -> {agent.log_path.name}")

    results: dict = {
        "config": {
            "task": TASK, "mode": "a1",
            "n_test": N_TEST_ROWS,
            "row_sample_seed": GATE_SEED_FOR_SAMPLE,
            "feature_cols": feature_cols,
            "corruptions": [{"family": f, "severity": s, "seed": k}
                             for f, s, k in GATE_CORRUPTIONS],
            "ml_model": "xgboost",
        },
        "conditions": [],
    }

    # --- clean baseline ---
    print("\n[clean]")
    auc_ml_clean = _ml_auc_on(pipe, df, idxs, target, feature_cols)
    auc_agent_clean, stats_clean = _agent_auc_on(agent, df, idxs, target)
    print(f"    ML    AUROC = {auc_ml_clean:.4f}")
    print(f"    Agent AUROC = {auc_agent_clean:.4f}   "
          f"(parse_ok={stats_clean['parse_ok']}/{stats_clean['n']}, "
          f"tpr={stats_clean['tpr']:.2f}, tnr={stats_clean['tnr']:.2f})")
    results["clean"] = {
        "ml_auc": auc_ml_clean,
        "agent_auc": auc_agent_clean,
        "agent_stats": stats_clean,
    }

    # --- corruptions ---
    for fam, sev, seed in GATE_CORRUPTIONS:
        print(f"\n[{fam} sev={sev} seed={seed}]")
        df_corr = apply(df, fam, sev, seed)
        # if schema_drift, idxs still valid (same rows, possibly renamed cols)
        auc_ml = _ml_auc_on(pipe, df_corr, idxs, target, feature_cols)
        auc_agent, stats = _agent_auc_on(agent, df_corr, idxs, target)
        d_ml = auc_ml - auc_ml_clean
        d_ag = auc_agent - auc_agent_clean
        ratio = (abs(d_ag) / abs(d_ml)) if abs(d_ml) > 1e-6 else float("inf")
        inv_ratio = (abs(d_ml) / abs(d_ag)) if abs(d_ag) > 1e-6 else float("inf")
        max_ratio = max(ratio, inv_ratio)
        print(f"    ML    AUROC = {auc_ml:.4f}   delta = {d_ml:+.4f}")
        print(f"    Agent AUROC = {auc_agent:.4f}   delta = {d_ag:+.4f}   "
              f"(parse_ok={stats['parse_ok']}/{stats['n']}, "
              f"tpr={stats['tpr']:.2f}, tnr={stats['tnr']:.2f})")
        print(f"    |ratio|     = {max_ratio:.2f}x   (gate threshold = 2.0)")
        results["conditions"].append({
            "family": fam, "severity": sev, "seed": seed,
            "ml_auc": auc_ml, "agent_auc": auc_agent,
            "delta_ml": d_ml, "delta_agent": d_ag,
            "abs_ratio_max": max_ratio,
            "agent_stats": stats,
        })

    # --- gate decision ---
    max_observed = max((c["abs_ratio_max"] for c in results["conditions"]),
                       default=0.0)
    passed = max_observed > 2.0
    results["gate"] = {
        "threshold": 2.0,
        "max_observed_ratio": max_observed,
        "passed": passed,
    }

    out = LOGS / f"gate_{TASK}_a1.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"GATE: max |delta_agent/delta_ml| or |delta_ml/delta_agent| = {max_observed:.2f}x")
    print(f"GATE: {'PASSED ✓' if passed else 'FAILED ✗'} (threshold > 2.0x)")
    print(f"Wrote {out}")
    print(f"Elapsed {time.time() - t0:.1f}s")
    return 0 if passed else 3


if __name__ == "__main__":
    import sys
    sys.exit(main())

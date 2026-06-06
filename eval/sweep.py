"""Full corruption sweep for AgentDQ.

Cartesian product over: task × family × severity × seed, n=1000 stratified
test rows per cell. Trains ML once per task (cached). For each cell, applies
the corruption to the held-out test slice and scores BOTH ML and agent.

Output: logs/sweep.parquet (one row per cell). Resumable — skips cells already
present in the parquet by (task, family, severity, seed).

Run:
    python -m eval.sweep                 # default: T3 + canonical-5 + 5 seeds
    python -m eval.sweep --tasks t3 t2   # both tasks
    python -m eval.sweep --dry-run       # print plan only
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from consumers.agent import Agent, AgentConfig, _row_dict
from consumers.features import T1_EXTRA_FEATURE_COLS, T2_FEATURE_COLS, T3_TEXT_COLS
from consumers.ml_pipeline import (CATEGORICAL_COLS, XGB_PARAMS,
                                    _build_preprocessor, _to_numeric)
from corruption import apply

REPO = Path(__file__).resolve().parent.parent
SILVER = REPO / "data" / "silver" / "order_features.parquet"
LOGS = REPO / "logs"
SWEEP_PARQUET = LOGS / "sweep.parquet"

CANONICAL_FAMILIES = [
    "missing_injection",
    "mojibake_roundtrip",
    "category_taxonomy_drift",
    "currency_unit_drift",
    "schema_drift",
    "type_flip",
]
SEVERITIES = [("low", 0.10), ("med", 0.20), ("high", 0.30)]
SEEDS_FULL = (13, 17, 23, 31, 47)
SEEDS_REDUCED = (13, 17, 23)
N_PER_CELL = 1000


def _feature_cols(task: str) -> list[str]:
    if task == "t1":
        return list(T2_FEATURE_COLS) + list(T1_EXTRA_FEATURE_COLS)
    if task == "t3":
        return list(T2_FEATURE_COLS) + list(T3_TEXT_COLS)
    return list(T2_FEATURE_COLS)


def _train_xgb(df: pd.DataFrame, target: str, feature_cols: list[str]) -> Pipeline:
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    train = df[(df["split"].isin(["train", "val"])) & df[target].notna()]
    X = _to_numeric(train[feature_cols].copy(), num_cols)
    y = train[target].astype(int).to_numpy()
    pre = _build_preprocessor(feature_cols, scale=False)
    clf = XGBClassifier(random_state=13, **XGB_PARAMS)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


def _stratified(df: pd.DataFrame, target: str, n: int, seed: int) -> np.ndarray:
    pool = df[(df["split"] == "test") & df[target].notna()]
    half = n // 2
    pos = pool[pool[target] == 1]
    neg = pool[pool[target] == 0]
    pos_n = min(half, len(pos))
    neg_n = min(n - pos_n, len(neg))
    chosen = pd.concat([
        pos.sample(n=pos_n, random_state=seed),
        neg.sample(n=neg_n, random_state=seed),
    ]).sample(frac=1.0, random_state=seed)
    return chosen.index.to_numpy()


def _ml_auc(pipe: Pipeline, df: pd.DataFrame, idxs: np.ndarray,
            target: str, feature_cols: list[str]) -> float:
    sub = df.loc[idxs]
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    X = sub.reindex(columns=feature_cols)
    X = _to_numeric(X, num_cols)
    y = sub[target].astype(int).to_numpy()
    proba = pipe.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def _agent_auc(agent: Agent, df: pd.DataFrame, idxs: np.ndarray,
               target: str, workers: int) -> tuple[float, int]:
    sub = df.loc[idxs]
    series = [sub.iloc[i] for i in range(len(sub))]
    y_true = sub[target].astype(int).to_numpy()
    scores = np.zeros(len(sub))
    parse_ok = 0

    def worker(i: int) -> tuple[int, float, int]:
        d = _row_dict(series[i], agent.feature_cols)
        r = agent.predict_one(d)
        p, c = r["prediction"], r["confidence"]
        if p in (0, 1):
            return i, (c if p == 1 else 1.0 - c), 1
        return i, 0.5, 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, i) for i in range(len(series))]
        for fut in as_completed(futs):
            i, sc, ok = fut.result()
            scores[i] = sc
            parse_ok += ok

    return float(roc_auc_score(y_true, scores)), parse_ok


def _existing_keys() -> set[tuple]:
    if not SWEEP_PARQUET.exists():
        return set()
    df = pd.read_parquet(SWEEP_PARQUET)
    return {(r.task, r.family, r.severity_label, int(r.seed))
            for r in df.itertuples()}


def _append_row(row: dict) -> None:
    df_new = pd.DataFrame([row])
    if SWEEP_PARQUET.exists():
        old = pd.read_parquet(SWEEP_PARQUET)
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new.to_parquet(SWEEP_PARQUET, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["t3"], choices=["t1", "t2", "t3"])
    ap.add_argument("--families", nargs="+", default=CANONICAL_FAMILIES)
    ap.add_argument("--n", type=int, default=N_PER_CELL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(SILVER)
    existing = _existing_keys()

    plan: list[tuple] = []
    for task in args.tasks:
        target = {"t1": "y_t1", "t2": "y_t2", "t3": "y_t3"}[task]
        seeds = SEEDS_FULL if task in ("t3", "t2") else SEEDS_REDUCED
        for family in args.families:
            for sev_label, sev_val in SEVERITIES:
                for seed in seeds:
                    key = (task, family, sev_label, seed)
                    if key in existing:
                        continue
                    plan.append((task, target, family, sev_label, sev_val, seed))

    print(f"=== AgentDQ SWEEP ===")
    print(f"  tasks      : {args.tasks}")
    print(f"  families   : {args.families}")
    print(f"  n per cell : {args.n}")
    print(f"  workers    : {args.workers}")
    print(f"  cells      : {len(plan)} pending  ({len(existing)} already done)")
    print(f"  parquet    : {SWEEP_PARQUET}")
    if args.dry_run or not plan:
        return 0

    # Train ML once per task
    pipes: dict[str, tuple[Pipeline, list[str]]] = {}
    for task in args.tasks:
        target = {"t1": "y_t1", "t2": "y_t2", "t3": "y_t3"}[task]
        feats = _feature_cols(task)
        print(f"\n[ML] training XGBoost for {task} ...")
        pipe = _train_xgb(df, target, feats)
        pipes[task] = (pipe, feats)
        print(f"     done.")

    # Cache one agent per task
    agents: dict[str, Agent] = {}
    for task in args.tasks:
        agents[task] = Agent(AgentConfig(task=task, log_tag=f"sweep"))
        print(f"[Agent] {task} log -> {agents[task].log_path.name}")

    # Cache clean baselines per (task, seed) so deltas are correct
    clean_cache: dict[tuple, tuple[float, float]] = {}

    t0 = time.time()
    for i, (task, target, family, sev_label, sev_val, seed) in enumerate(plan, 1):
        pipe, feats = pipes[task]
        agent = agents[task]
        idxs = _stratified(df, target, args.n, seed=seed)

        if (task, seed) not in clean_cache:
            ml_clean = _ml_auc(pipe, df, idxs, target, feats)
            ag_clean, ok_clean = _agent_auc(agent, df, idxs, target, args.workers)
            clean_cache[(task, seed)] = (ml_clean, ag_clean)
            print(f"[clean] task={task} seed={seed}  "
                  f"ml={ml_clean:.4f}  agent={ag_clean:.4f}  ok={ok_clean}/{args.n}")
        ml_clean, ag_clean = clean_cache[(task, seed)]

        df_corr = apply(df, family, sev_val, seed)
        ml_auc = _ml_auc(pipe, df_corr, idxs, target, feats)
        ag_auc, parse_ok = _agent_auc(agent, df_corr, idxs, target, args.workers)
        d_ml = ml_auc - ml_clean
        d_ag = ag_auc - ag_clean
        ratio = (abs(d_ag) / abs(d_ml)) if abs(d_ml) > 1e-6 else float("inf")
        inv = (abs(d_ml) / abs(d_ag)) if abs(d_ag) > 1e-6 else float("inf")
        max_ratio = max(ratio, inv)

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "family": family,
            "severity_label": sev_label,
            "severity": float(sev_val),
            "seed": int(seed),
            "n": int(args.n),
            "ml_auc_clean": ml_clean,
            "ml_auc": ml_auc,
            "ml_delta": d_ml,
            "agent_auc_clean": ag_clean,
            "agent_auc": ag_auc,
            "agent_delta": d_ag,
            "abs_ratio_max": max_ratio,
            "parse_ok": int(parse_ok),
        }
        _append_row(row)
        elapsed = time.time() - t0
        eta = elapsed / i * (len(plan) - i)
        print(f"[{i:3d}/{len(plan)}] {task}/{family}/{sev_label}/seed={seed}  "
              f"ml={ml_auc:.4f}({d_ml:+.4f})  ag={ag_auc:.4f}({d_ag:+.4f})  "
              f"r={max_ratio:.2f}x  ok={parse_ok}/{args.n}  "
              f"ETA {eta/60:.1f} min")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Wrote {SWEEP_PARQUET}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

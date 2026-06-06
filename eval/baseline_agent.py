"""Clean-test agent baseline for AgentDQ.

Runs the profiled-prompt agent on the FULL clean test split (or a stratified
subsample) for one task. Writes a single JSON with AUROC, F1, parse-ok, TPR,
TNR, n, runtime, and total tokens. This is the single big-N number cited in
the abstract / Section 2.6.

Run:
    python -m eval.baseline_agent --task t3 --n all --tag anchor
    python -m eval.baseline_agent --task t3 --n 200 --tag smoke
    python -m eval.baseline_agent --task t2 --n 2000 --tag anchor

Output:
    logs/agent_baseline_<task>_<tag>.json
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
from sklearn.metrics import f1_score, roc_auc_score

from consumers.agent import Agent, AgentConfig, _row_dict

REPO = Path(__file__).resolve().parent.parent
SILVER = REPO / "data" / "silver" / "order_features.parquet"
LOGS = REPO / "logs"


def _stratified(df: pd.DataFrame, target: str, n: int | None, seed: int) -> pd.DataFrame:
    pool = df[(df["split"] == "test") & df[target].notna()]
    if n is None or n >= len(pool):
        return pool
    half = n // 2
    pos = pool[pool[target] == 1]
    neg = pool[pool[target] == 0]
    pos_n = min(half, len(pos))
    neg_n = min(n - pos_n, len(neg))
    sub = pd.concat([
        pos.sample(n=pos_n, random_state=seed),
        neg.sample(n=neg_n, random_state=seed),
    ])
    return sub.sample(frac=1.0, random_state=seed)


def _score_row(agent: Agent, row: pd.Series, target: str) -> tuple[int, float, int, int]:
    d = _row_dict(row, agent.feature_cols)
    r = agent.predict_one(d)
    p = r["prediction"]
    c = r["confidence"]
    y = int(row[target])
    if p in (0, 1):
        score = c if p == 1 else 1.0 - c
        return y, score, p, 1
    return y, 0.5, 0, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2", "t3"], default="t3")
    ap.add_argument("--n", default="all", help="'all' or integer count")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="anchor")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent agent calls (default 8)")
    args = ap.parse_args()

    n_arg: int | None = None if args.n == "all" else int(args.n)

    target = {"t1": "y_t1", "t2": "y_t2", "t3": "y_t3"}[args.task]
    df = pd.read_parquet(SILVER)
    sub = _stratified(df, target, n_arg, seed=args.seed)
    print(f"[{args.task}] clean baseline on n={len(sub)} test rows "
          f"(class balance: pos={int((sub[target]==1).sum())}, "
          f"neg={int((sub[target]==0).sum())})")

    agent = Agent(AgentConfig(task=args.task, log_tag=f"baseline_{args.tag}",
                              seed=args.seed))
    print(f"  log -> {agent.log_path.name}")

    t0 = time.time()
    results: list[tuple[int, float, int, int]] = [None] * len(sub)  # type: ignore
    rows = list(sub.itertuples(index=False))
    series_list = [sub.iloc[i] for i in range(len(sub))]

    def task_fn(i: int):
        return i, _score_row(agent, series_list[i], target)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(task_fn, i) for i in range(len(series_list))]
        done = 0
        for fut in as_completed(futs):
            i, res = fut.result()
            results[i] = res
            done += 1
            if done % 100 == 0 or done == len(series_list):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(series_list) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(series_list)}  "
                      f"({rate:.1f} rows/s, ETA {eta/60:.1f} min)")

    y_true = np.array([r[0] for r in results])
    scores = np.array([r[1] for r in results])
    preds = np.array([r[2] for r in results])
    parse_ok = int(sum(r[3] for r in results))

    auc = float(roc_auc_score(y_true, scores))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    tpr = float(((preds == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum()))
    tnr = float(((preds == 0) & (y_true == 0)).sum() / max(1, (y_true == 0).sum()))

    # token totals from the JSONL log
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    for line in agent.log_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        total_tokens += rec.get("total_tokens", 0)
        prompt_tokens += rec.get("prompt_tokens", 0)
        completion_tokens += rec.get("completion_tokens", 0)

    elapsed = time.time() - t0
    out = {
        "task": args.task,
        "tag": args.tag,
        "ts": datetime.now(timezone.utc).isoformat(),
        "n": int(len(sub)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "seed": args.seed,
        "workers": args.workers,
        "metrics": {
            "auroc": auc,
            "f1": f1,
            "tpr": tpr,
            "tnr": tnr,
            "parse_ok": parse_ok,
            "parse_ok_rate": parse_ok / max(1, len(sub)),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "runtime_s": round(elapsed, 1),
        "log_file": agent.log_path.name,
    }

    out_path = LOGS / f"agent_baseline_{args.task}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print(f"  AUROC      : {auc:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  TPR / TNR  : {tpr:.4f} / {tnr:.4f}")
    print(f"  parse_ok   : {parse_ok}/{len(sub)} ({parse_ok/max(1,len(sub)):.2%})")
    print(f"  tokens     : {total_tokens:,} (prompt {prompt_tokens:,} + "
          f"completion {completion_tokens:,})")
    # gpt-4o-mini: $0.15/1M prompt, $0.60/1M completion
    cost = prompt_tokens * 0.15e-6 + completion_tokens * 0.60e-6
    print(f"  est. cost  : ${cost:.4f}")
    print(f"  runtime    : {elapsed/60:.1f} min")
    print(f"  wrote      : {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

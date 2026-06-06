"""Freeze the temporal split for AgentDQ.

Splits orders by `order_purchase_timestamp`:
    train: < 2018-04-01
    val:   [2018-04-01, 2018-06-01)
    test:  >= 2018-06-01

Outputs:
    configs/split.json          - boundary timestamps + counts (audited)
    data/bronze/split_orders.parquet - per-order_id split label

Run:
    python scripts/freeze_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BRONZE = REPO / "data" / "bronze"
CONFIGS = REPO / "configs"

TRAIN_END = pd.Timestamp("2018-04-01 00:00:00")
VAL_END = pd.Timestamp("2018-06-01 00:00:00")


def main() -> int:
    orders = pd.read_parquet(BRONZE / "orders.parquet")
    orders = orders[["order_id", "order_purchase_timestamp"]].copy()

    n_total = len(orders)
    n_missing_ts = orders["order_purchase_timestamp"].isna().sum()
    if n_missing_ts:
        print(f"[WARN] {n_missing_ts} orders have no purchase timestamp; assigning to 'drop'.")

    def label(ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return "drop"
        if ts < TRAIN_END:
            return "train"
        if ts < VAL_END:
            return "val"
        return "test"

    orders["split"] = orders["order_purchase_timestamp"].apply(label)
    counts = orders["split"].value_counts().to_dict()

    out_parquet = BRONZE / "split_orders.parquet"
    orders[["order_id", "split"]].to_parquet(out_parquet, index=False)

    summary = {
        "policy": "temporal split on order_purchase_timestamp",
        "boundaries": {
            "train_end_exclusive": TRAIN_END.isoformat(),
            "val_end_exclusive": VAL_END.isoformat(),
        },
        "counts": {
            "train": int(counts.get("train", 0)),
            "val": int(counts.get("val", 0)),
            "test": int(counts.get("test", 0)),
            "drop": int(counts.get("drop", 0)),
            "total": int(n_total),
        },
        "fractions": {
            "train": round(counts.get("train", 0) / n_total, 4),
            "val": round(counts.get("val", 0) / n_total, 4),
            "test": round(counts.get("test", 0) / n_total, 4),
        },
        "artifacts": {
            "split_parquet": str(out_parquet.relative_to(REPO).as_posix()),
        },
    }

    CONFIGS.mkdir(parents=True, exist_ok=True)
    out_json = CONFIGS / "split.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

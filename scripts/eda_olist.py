"""EDA on bronze tables: row counts, join cardinalities, T1/T2 target prevalences.

Run:
    python scripts/eda_olist.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BRONZE = REPO / "data" / "bronze"


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(BRONZE / f"{name}.parquet")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    orders = load("orders")
    reviews = load("order_reviews")
    items = load("order_items")
    customers = load("customers")
    products = load("products")
    sellers = load("sellers")

    section("Date range")
    purchase = orders["order_purchase_timestamp"].dropna()
    print(f"order_purchase_timestamp:  {purchase.min()}  →  {purchase.max()}")
    print(f"orders by year:")
    print(purchase.dt.year.value_counts().sort_index().to_string())
    print(f"orders by year-quarter:")
    yq = purchase.dt.to_period("Q").value_counts().sort_index()
    print(yq.to_string())

    section("Order status mix")
    print(orders["order_status"].value_counts().to_string())

    # T1: review_score binarized {1-2 = low/0, 4-5 = high/1, drop 3}
    section("T1 — predict review_score")
    rs = reviews["review_score"].dropna()
    print("review_score distribution:")
    print(rs.value_counts().sort_index().to_string())
    rs_used = rs[rs.isin([1, 2, 4, 5])]
    high = rs_used.isin([4, 5]).mean()
    print(f"\nAfter dropping score=3: kept {len(rs_used):,} of {len(rs):,} reviews")
    print(f"Positive (high) prevalence: {high:.4f}")

    # T2: late_delivery — delivered after estimated date
    section("T2 — predict late_delivery")
    od = orders.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"]).copy()
    od["late"] = (od["order_delivered_customer_date"] > od["order_estimated_delivery_date"]).astype(int)
    print(f"orders with both delivery dates present: {len(od):,} of {len(orders):,}")
    print(f"late_delivery prevalence: {od['late'].mean():.4f}")
    print(f"late_delivery counts:")
    print(od["late"].value_counts().to_string())

    section("Join cardinalities (sanity)")
    print(f"orders ↔ customers (1:1 on customer_id):    "
          f"{orders['customer_id'].nunique():,} unique vs {len(customers):,} customers")
    print(f"orders ↔ reviews   (1:1 on order_id):       "
          f"reviews per order avg = {len(reviews) / orders['order_id'].nunique():.3f}")
    print(f"orders ↔ items     (1:N on order_id):       "
          f"avg items per order = {len(items) / orders['order_id'].nunique():.2f}")
    print(f"items  ↔ products:                          "
          f"{items['product_id'].nunique():,} of {len(products):,} products used")
    print(f"items  ↔ sellers:                           "
          f"{items['seller_id'].nunique():,} of {len(sellers):,} sellers active")

    section("Summary stored")
    summary = {
        "n_orders": int(len(orders)),
        "n_reviews_used_T1": int(len(rs_used)),
        "T1_positive_prev": float(round(high, 6)),
        "n_orders_used_T2": int(len(od)),
        "T2_positive_prev": float(round(od["late"].mean(), 6)),
        "purchase_min": str(purchase.min()),
        "purchase_max": str(purchase.max()),
    }
    out = REPO / "configs" / "eda_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  → {out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

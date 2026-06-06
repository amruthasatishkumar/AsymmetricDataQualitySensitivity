"""Build per-order features for AgentDQ.

Joins the 9 bronze tables into one row per `order_id` with features that
predate the prediction target. Two label columns are attached:

    y_t1  -- review_score binarized (1 if score in {4,5}, 0 if {1,2}, NaN if 3)
    y_t2  -- late_delivery (1 if delivered > estimated, 0 otherwise, NaN if missing)

For T2, only purchase-time features are leak-free. We expose this via
`T2_FEATURE_COLS`. T1 may use any feature here (review happens post-delivery).

The split column is joined from data/bronze/split_orders.parquet.

Output:
    data/silver/order_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BRONZE = REPO / "data" / "bronze"
SILVER = REPO / "data" / "silver"

# Features safe for T2 (known at order placement time)
T2_FEATURE_COLS: list[str] = [
    "purchase_hour",
    "purchase_dow",
    "purchase_month",
    "days_to_estimated",
    "n_items",
    "total_price",
    "total_freight",
    "freight_ratio",
    "max_price",
    "mean_price",
    "n_unique_products",
    "n_unique_sellers",
    "n_payments",
    "total_payment_value",
    "max_installments",
    "payment_type_modal",
    "customer_state",
    "n_seller_states",
    "mean_product_weight_g",
    "mean_product_volume_cm3",
    "dominant_category",
]

# T1 may additionally use delivery latency features
T1_EXTRA_FEATURE_COLS: list[str] = [
    "days_to_approved",
    "days_carrier_to_customer",
    "days_purchase_to_delivery",
    "delivery_vs_estimate_days",
    "order_status",
]

# T3 features = T2 features + review text. T3 = "is this a low review (1-2★)?"
# Text columns are Portuguese free-form review_comment_title and message.
T3_TEXT_COLS: list[str] = ["review_comment_title", "review_comment_message"]
T3_FEATURE_COLS: list[str] = list(T2_FEATURE_COLS) + list(T3_TEXT_COLS)


def _load() -> dict[str, pd.DataFrame]:
    names = [
        "orders",
        "order_items",
        "order_payments",
        "order_reviews",
        "customers",
        "products",
        "sellers",
        "category_translation",
        "split_orders",
    ]
    return {n: pd.read_parquet(BRONZE / f"{n}.parquet") for n in names}


def _agg_items(items: pd.DataFrame, products: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    p = products.copy()
    p["product_volume_cm3"] = (
        p["product_length_cm"].astype("Float64")
        * p["product_height_cm"].astype("Float64")
        * p["product_width_cm"].astype("Float64")
    )
    items_j = items.merge(
        p[["product_id", "product_category_name", "product_weight_g", "product_volume_cm3"]],
        on="product_id",
        how="left",
    ).merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

    def _modal(s: pd.Series) -> object:
        s = s.dropna()
        if len(s) == 0:
            return pd.NA
        return s.mode().iloc[0]

    grp = items_j.groupby("order_id", sort=False)
    agg = grp.agg(
        n_items=("order_item_id", "size"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_price=("price", "max"),
        mean_price=("price", "mean"),
        n_unique_products=("product_id", "nunique"),
        n_unique_sellers=("seller_id", "nunique"),
        n_seller_states=("seller_state", "nunique"),
        mean_product_weight_g=("product_weight_g", "mean"),
        mean_product_volume_cm3=("product_volume_cm3", "mean"),
        dominant_category=("product_category_name", _modal),
    ).reset_index()
    agg["freight_ratio"] = agg["total_freight"] / (agg["total_price"] + 1e-6)
    return agg


def _agg_payments(payments: pd.DataFrame) -> pd.DataFrame:
    def _modal(s: pd.Series) -> object:
        s = s.dropna()
        if len(s) == 0:
            return pd.NA
        return s.mode().iloc[0]

    grp = payments.groupby("order_id", sort=False)
    return grp.agg(
        n_payments=("payment_sequential", "size"),
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_type_modal=("payment_type", _modal),
    ).reset_index()


def build() -> pd.DataFrame:
    d = _load()
    orders = d["orders"].copy()

    # Time features
    pt = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["purchase_hour"] = pt.dt.hour.astype("Int64")
    orders["purchase_dow"] = pt.dt.dayofweek.astype("Int64")
    orders["purchase_month"] = pt.dt.month.astype("Int64")

    def _days(a: pd.Series, b: pd.Series) -> pd.Series:
        return ((pd.to_datetime(a) - pd.to_datetime(b)).dt.total_seconds() / 86400.0).astype("Float64")

    orders["days_to_estimated"] = _days(orders["order_estimated_delivery_date"], orders["order_purchase_timestamp"])
    orders["days_to_approved"] = _days(orders["order_approved_at"], orders["order_purchase_timestamp"])
    orders["days_carrier_to_customer"] = _days(
        orders["order_delivered_customer_date"], orders["order_delivered_carrier_date"]
    )
    orders["days_purchase_to_delivery"] = _days(
        orders["order_delivered_customer_date"], orders["order_purchase_timestamp"]
    )
    orders["delivery_vs_estimate_days"] = _days(
        orders["order_delivered_customer_date"], orders["order_estimated_delivery_date"]
    )

    # Customer state
    orders = orders.merge(
        d["customers"][["customer_id", "customer_state"]], on="customer_id", how="left"
    )

    # Item / product / seller aggregates
    orders = orders.merge(_agg_items(d["order_items"], d["products"], d["sellers"]), on="order_id", how="left")
    orders = orders.merge(_agg_payments(d["order_payments"]), on="order_id", how="left")

    # Labels
    rev = d["order_reviews"].drop_duplicates(subset=["order_id"], keep="first")[
        ["order_id", "review_score", "review_comment_title", "review_comment_message"]
    ]
    orders = orders.merge(rev, on="order_id", how="left")
    orders["y_t1"] = np.where(
        orders["review_score"].isin([4, 5]), 1,
        np.where(orders["review_score"].isin([1, 2]), 0, np.nan),
    )

    # T3: low review (1 or 2 stars) vs high review (4 or 5 stars). Score 3 => NaN.
    orders["y_t3"] = np.where(
        orders["review_score"].isin([1, 2]), 1,
        np.where(orders["review_score"].isin([4, 5]), 0, np.nan),
    )

    delivered = pd.to_datetime(orders["order_delivered_customer_date"])
    estimated = pd.to_datetime(orders["order_estimated_delivery_date"])
    orders["y_t2"] = np.where(
        delivered.notna() & estimated.notna(),
        (delivered > estimated).astype(float),
        np.nan,
    )

    # T5: cross-source payment-vs-items consistency.
    # NOTE: Olist payments are essentially perfectly reconciled (median diff = 0,
    # only ~0.25% of orders have any meaningful mismatch). T5 was designed but
    # the dataset doesn't support it as a learnable task. Retained as a stub
    # for completeness; not used in the headline experiments.

    # Split label
    orders = orders.merge(d["split_orders"], on="order_id", how="left")

    return orders


def main() -> int:
    SILVER.mkdir(parents=True, exist_ok=True)
    df = build()

    keep_cols = (
        ["order_id", "split", "y_t1", "y_t2", "y_t3"]
        + T2_FEATURE_COLS
        + T1_EXTRA_FEATURE_COLS
        + T3_TEXT_COLS
    )
    # de-duplicate while preserving order
    seen: set[str] = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    df_out = df[keep_cols]
    out = SILVER / "order_features.parquet"
    df_out.to_parquet(out, index=False)

    print(f"Rows: {len(df_out):,}    Cols: {len(df_out.columns)}")
    for tk in ("y_t1", "y_t2", "y_t3"):
        s = df_out[tk]
        print(f"{tk} non-null: {s.notna().sum():,}    prev={s.mean():.4f}")
    print(f"Split:")
    print(df_out["split"].value_counts().to_string())
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

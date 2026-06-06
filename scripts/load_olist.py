"""Load Olist raw CSVs into Bronze parquet with declared dtypes.

Reads from data/raw/, writes to data/bronze/. Dtypes are explicit
(no inference) so corruption variants in Step 4 can be reasoned
about predictably.

Run:
    python scripts/load_olist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
BRONZE = REPO / "data" / "bronze"

# (filename, dtype_spec, parse_dates, output_name)
TABLES = [
    (
        "olist_customers_dataset.csv",
        {
            "customer_id": "string",
            "customer_unique_id": "string",
            "customer_zip_code_prefix": "Int64",
            "customer_city": "string",
            "customer_state": "string",
        },
        [],
        "customers",
    ),
    (
        "olist_geolocation_dataset.csv",
        {
            "geolocation_zip_code_prefix": "Int64",
            "geolocation_lat": "float64",
            "geolocation_lng": "float64",
            "geolocation_city": "string",
            "geolocation_state": "string",
        },
        [],
        "geolocation",
    ),
    (
        "olist_order_items_dataset.csv",
        {
            "order_id": "string",
            "order_item_id": "Int64",
            "product_id": "string",
            "seller_id": "string",
            "price": "float64",
            "freight_value": "float64",
        },
        ["shipping_limit_date"],
        "order_items",
    ),
    (
        "olist_order_payments_dataset.csv",
        {
            "order_id": "string",
            "payment_sequential": "Int64",
            "payment_type": "string",
            "payment_installments": "Int64",
            "payment_value": "float64",
        },
        [],
        "order_payments",
    ),
    (
        "olist_order_reviews_dataset.csv",
        {
            "review_id": "string",
            "order_id": "string",
            "review_score": "Int64",
            "review_comment_title": "string",
            "review_comment_message": "string",
        },
        ["review_creation_date", "review_answer_timestamp"],
        "order_reviews",
    ),
    (
        "olist_orders_dataset.csv",
        {
            "order_id": "string",
            "customer_id": "string",
            "order_status": "string",
        },
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "orders",
    ),
    (
        "olist_products_dataset.csv",
        {
            "product_id": "string",
            "product_category_name": "string",
            "product_name_lenght": "Int64",
            "product_description_lenght": "Int64",
            "product_photos_qty": "Int64",
            "product_weight_g": "Int64",
            "product_length_cm": "Int64",
            "product_height_cm": "Int64",
            "product_width_cm": "Int64",
        },
        [],
        "products",
    ),
    (
        "olist_sellers_dataset.csv",
        {
            "seller_id": "string",
            "seller_zip_code_prefix": "Int64",
            "seller_city": "string",
            "seller_state": "string",
        },
        [],
        "sellers",
    ),
    (
        "product_category_name_translation.csv",
        {
            "product_category_name": "string",
            "product_category_name_english": "string",
        },
        [],
        "category_translation",
    ),
]


def main() -> int:
    BRONZE.mkdir(parents=True, exist_ok=True)
    print(f"{'table':<22} {'rows':>10}  {'cols':>5}  out")
    print("-" * 70)
    total_rows = 0
    for fname, dtypes, dates, out_name in TABLES:
        src = RAW / fname
        if not src.exists():
            print(f"[FAIL] missing raw file: {src}")
            return 2
        df = pd.read_csv(src, dtype=dtypes, parse_dates=dates)
        out = BRONZE / f"{out_name}.parquet"
        df.to_parquet(out, index=False)
        total_rows += len(df)
        print(f"{out_name:<22} {len(df):>10,}  {len(df.columns):>5}  {out.name}")
    print("-" * 70)
    print(f"Total rows across {len(TABLES)} tables: {total_rows:,}")
    print(f"Bronze: {BRONZE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

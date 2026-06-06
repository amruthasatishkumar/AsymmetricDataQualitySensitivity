"""One-shot sanity check: apply each corruption family to a 2000-row sample and
print row/col/NaN counts plus one human-readable change per family.
"""
from __future__ import annotations

import pandas as pd

from corruption import apply, list_families


def main() -> int:
    df = (
        pd.read_parquet("data/silver/order_features.parquet")
        .sample(2000, random_state=0)
        .reset_index(drop=True)
    )
    print(f"baseline rows={len(df)}, cols={len(df.columns)}, NaN={df.isna().sum().sum():,}")
    print()
    print(f"{'family':<26} | rows | cols |  NaN     | sample change")
    print("-" * 100)
    for fam in list_families():
        out = apply(df, fam, severity=0.20, seed=42)
        nan = out.isna().sum().sum()
        sample = ""
        if fam == "unit_mix":
            sample = f"total_price max -> {out['total_price'].max():,.1f}"
        elif fam == "type_flip":
            sample = f"total_price dtype -> {out['total_price'].dtype}"
        elif fam == "encoding_noise":
            v = out["customer_state"].iloc[0]
            sample = f"customer_state[0] -> {v!r}"
        elif fam == "schema_drift":
            renamed = [c for c in out.columns if c.startswith("col_42_")]
            sample = f"{len(renamed)} cols renamed (e.g. {renamed[:2]})"
        elif fam == "label_noise_categorical":
            diff = (df["customer_state"].astype(str) != out["customer_state"].astype(str)).sum()
            sample = f"customer_state differs in {diff} rows"
        elif fam == "duplicate_rows":
            sample = f"+{len(out) - len(df)} dup rows"
        elif fam == "outlier_injection":
            sample = f"total_price max -> {out['total_price'].max():,.1f}"
        elif fam == "missing_injection":
            sample = f"NaN added: {nan - df.isna().sum().sum():,}"
        print(f"{fam:<26} | {len(out):>4} | {len(out.columns):>4} | {nan:>7,} | {sample}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

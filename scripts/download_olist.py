"""Download the Olist Brazilian E-Commerce dataset.

Tries kagglehub first (uses ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars).
If that fails, prints manual-download instructions.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
License: CC BY-NC-SA 4.0

Run:
    python scripts/download_olist.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"

EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

KAGGLE_REF = "olistbr/brazilian-ecommerce"
KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"


def already_present() -> bool:
    return all((RAW / f).exists() and (RAW / f).stat().st_size > 0 for f in EXPECTED_FILES)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_manifest() -> None:
    manifest = REPO / "data" / "raw_manifest.txt"
    lines = ["# Olist raw-file SHA-256 manifest (first 16 hex chars)", f"# source: {KAGGLE_URL}", ""]
    for f in EXPECTED_FILES:
        p = RAW / f
        lines.append(f"{hash_file(p)}  {p.stat().st_size:>10}  {f}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nManifest written: {manifest}")
    for line in lines[3:]:
        print(line)


def manual_instructions() -> int:
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD FALLBACK")
    print("=" * 70)
    print(f"1. Open {KAGGLE_URL}")
    print("2. Click 'Download' (top right). Sign in if prompted.")
    print(f"3. Unzip the archive into: {RAW}")
    print("4. Rerun this script to verify and generate the manifest.")
    print("=" * 70)
    return 2


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)

    if already_present():
        print(f"All 9 CSVs already present in {RAW}. Skipping download.")
        write_manifest()
        return 0

    try:
        import kagglehub  # type: ignore
    except ImportError:
        print("[INFO] kagglehub not installed; falling back to manual download.")
        return manual_instructions()

    try:
        print(f"Downloading {KAGGLE_REF} via kagglehub ...")
        cached_path = Path(kagglehub.dataset_download(KAGGLE_REF))
        print(f"kagglehub cached the dataset at: {cached_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] kagglehub failed: {e}")
        print("       Common cause: no Kaggle credentials. Either set KAGGLE_USERNAME +")
        print("       KAGGLE_KEY env vars, or place ~/.kaggle/kaggle.json, OR use manual.")
        return manual_instructions()

    # Copy from kagglehub cache into our data/raw/ for reproducibility
    copied = 0
    for fname in EXPECTED_FILES:
        srcs = list(cached_path.rglob(fname))
        if not srcs:
            print(f"[WARN] {fname} not found inside cached dataset; structure may differ.")
            continue
        dst = RAW / fname
        shutil.copy2(srcs[0], dst)
        copied += 1
    print(f"Copied {copied}/{len(EXPECTED_FILES)} CSVs into {RAW}")

    if not already_present():
        print("[FAIL] Some files missing after download. See messages above.")
        return manual_instructions()

    write_manifest()
    print("\n[OK] Olist dataset ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

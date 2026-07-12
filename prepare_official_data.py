"""Extract a reproducible all-asset window from the canonical G-Research JAY file.

Run with `.jay-venv/bin/python prepare_official_data.py`. That environment is
Python 3.11 because the JAY reader's macOS wheel is x86_64-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import datatable as dt
import pandas as pd


TRAIN_PATH = Path("data/official/orig_train.jay")
ASSETS_PATH = Path("data/official/orig_asset_details.jay")
OUTPUT_PATH = Path("data/official/train_last_180_days.parquet")
MANIFEST_PATH = Path("data/official/train_last_180_days.manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    train = dt.fread(TRAIN_PATH)
    max_timestamp = int(train[:, dt.max(dt.f.timestamp)][0, 0])
    start_timestamp = max_timestamp - args.days * 24 * 60 * 60
    window = train[dt.f.timestamp >= start_timestamp, :].to_pandas()
    assets = dt.fread(ASSETS_PATH).to_pandas()

    window = window.merge(assets, on="Asset_ID", how="left", validate="many_to_one")
    if window["Weight"].isna().any():
        raise ValueError("Every official train row must resolve to an asset weight.")
    window = window.rename(columns=str.lower)
    window["timestamp"] = pd.to_datetime(window["timestamp"], unit="s", utc=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(OUTPUT_PATH, index=False, compression="zstd")

    manifest = {
        "source_train": str(TRAIN_PATH),
        "source_train_sha256": sha256(TRAIN_PATH),
        "source_asset_details": str(ASSETS_PATH),
        "source_asset_details_sha256": sha256(ASSETS_PATH),
        "window_days": args.days,
        "window_start": str(window["timestamp"].min()),
        "window_end": str(window["timestamp"].max()),
        "rows": int(len(window)),
        "assets": int(window["asset_id"].nunique()),
        "asset_names": sorted(window["asset_name"].dropna().unique().tolist()),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

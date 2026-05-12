#!/usr/bin/env python3
"""
downsample_parquet.py

Downsamples a 0.25m parquet to 0.5m (or any target resolution)
using per-column aggregation:
  - dtm, dsm, object_height  -> mean
  - multi_return_ratio        -> mean
  - is_street, is_building    -> max (logical OR — if any sub-pixel is 1, output is 1)

Usage:
    python downsample_parquet.py input.parquet output.parquet
    python downsample_parquet.py input.parquet output.parquet --res 0.5
"""

import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Per-column aggregation method
AGGREGATIONS = {
    "dtm":                "mean",
    "dsm":                "mean",
    "object_height":      "mean",
    "multi_return_ratio": "mean",
    "is_street":          "max",
    "is_building":        "max",
    "is_building_edge": "max",
    "wall_depth_below": "max"
}


def main():
    parser = argparse.ArgumentParser(
        description="Downsample a parquet file to a coarser resolution."
    )
    parser.add_argument("input",  help="Input parquet file (e.g. 0.25m resolution)")
    parser.add_argument("output", help="Output parquet file")
    parser.add_argument("--res",  type=float, default=0.5,
                        help="Target resolution in metres (default: 0.5)")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pq.read_table(args.input).to_pandas()
    print(f"  Rows: {len(df):,}")

    # Snap each point to the nearest target-resolution grid cell
    # by rounding coordinates to the nearest multiple of res
    res = args.res
    df["x_cell"] = (df["x"] / res).round() * res
    df["y_cell"] = (df["y"] / res).round() * res

    print(f"Aggregating to {res}m resolution ...")

    # Build aggregation dict for columns that exist
    agg_dict = {}
    for col, method in AGGREGATIONS.items():
        if col in df.columns:
            agg_dict[col] = method

    if not agg_dict:
        sys.exit("Error: no recognised value columns found in parquet.")

    df_out = df.groupby(["x_cell", "y_cell"], sort=False).agg(agg_dict).reset_index()
    df_out = df_out.rename(columns={"x_cell": "x", "y_cell": "y"})

    # Re-binarize is_street and is_building after max aggregation
    for col in ("is_street", "is_building", "is_building_edge"):
        if col in df_out.columns:
            df_out[col] = (df_out[col] > 0.5).astype(np.float32)

    print(f"  Output rows: {len(df_out):,}")

    pq.write_table(
        pa.Table.from_pandas(df_out, preserve_index=False),
        args.output,
        compression="snappy"
    )
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
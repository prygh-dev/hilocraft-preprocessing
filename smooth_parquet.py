#!/usr/bin/env python3
"""
smooth_parquet.py

Smooths DSM and DTM in a parquet file using a median filter,
but only applies smoothing to non-building cells to preserve wall edges.

Must be run AFTER apply_osm_mask.py so that is_building is available.

Dependencies:
    pip install numpy pyarrow scipy

Usage:
    python smooth_parquet.py input.parquet output.parquet --res 0.25
    python smooth_parquet.py input.parquet output.parquet --res 0.25 --smooth-size 25
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import median_filter


def main():
    parser = argparse.ArgumentParser(
        description="Smooth DSM/DTM in a parquet file, preserving building walls."
    )
    parser.add_argument("input",  help="Input parquet file (must have is_building column)")
    parser.add_argument("output", help="Output parquet file")
    parser.add_argument("--res", type=float, required=True,
                        help="Grid resolution in metres (e.g. 0.25 or 0.5)")
    parser.add_argument("--smooth-size", type=int, default=5,
                        help="Median filter kernel size in pixels (default: 5). "
                             "To convert from metres: size = 2 * (radius_m / res) + 1")
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"Error: file not found: {args.input}")

    print(f"Reading {args.input} ...")
    df = pq.read_table(args.input).to_pandas()
    print(f"  Rows: {len(df):,}")

    if "is_building" not in df.columns:
        sys.exit("Error: is_building column not found. Run apply_osm_mask.py first.")

    # --- Compute grid indices directly from coordinates ---
    print("Computing grid indices ...")
    x = df["x"].values
    y = df["y"].values

    x_min = x.min()
    y_min = y.min()

    col_idx = np.round((x - x_min) / args.res).astype(int)
    row_idx = np.round((y - y_min) / args.res).astype(int)

    n_cols = col_idx.max() + 1
    n_rows = row_idx.max() + 1
    print(f"  Grid: {n_cols} x {n_rows} = {n_cols * n_rows:,} cells")

    # --- Scatter into 2D grids ---
    print("Building 2D grids ...")
    dsm_grid         = np.zeros((n_rows, n_cols), dtype=np.float32)
    dtm_grid         = np.zeros((n_rows, n_cols), dtype=np.float32)
    is_building_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    dsm_grid[row_idx, col_idx]         = df["dsm"].values
    dtm_grid[row_idx, col_idx]         = df["dtm"].values
    is_building_grid[row_idx, col_idx] = df["is_building"].values

    building_mask = is_building_grid > 0.5

    # --- Smooth DSM and DTM ---
    print(f"Applying median filter (size={args.smooth_size}) to non-building cells ...")
    print(f"  Smoothing radius: {(args.smooth_size - 1) / 2 * args.res:.2f}m")
    dsm_smoothed = median_filter(dsm_grid, size=args.smooth_size, mode="nearest")
    dtm_smoothed = median_filter(dtm_grid, size=args.smooth_size, mode="nearest")

    # Only apply smoothed values where is_building is false
    dsm_grid[~building_mask] = dsm_smoothed[~building_mask]
    dtm_grid[~building_mask] = dtm_smoothed[~building_mask]

    # --- Write smoothed values back to dataframe ---
    print("Writing smoothed values back ...")
    df["dsm"] = dsm_grid[row_idx, col_idx]
    df["dtm"] = dtm_grid[row_idx, col_idx]
    df["object_height"] = (df["dsm"] - df["dtm"]).clip(lower=0.0).astype(np.float32)

    # --- Write output ---
    print(f"Writing {args.output} ...")
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        args.output,
        compression="snappy"
    )
    print(f"Done. {len(df):,} rows written.")


if __name__ == "__main__":
    main()
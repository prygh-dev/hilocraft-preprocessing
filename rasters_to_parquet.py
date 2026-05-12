#!/usr/bin/env python3
"""
rasters_to_parquet.py

Combines aligned DSM, DTM, and MRR rasters into a single Parquet file.
All three rasters must be aligned to the same grid (use align_raster.py first,
using DTM as the reference).

NaN and nodata values are filled using the median of valid (non-zero, non-NaN)
neighboring cells in a 3x3 window. If all neighbors are also invalid, the value
is set to 0.

Dependencies:
    pip install rasterio numpy pyarrow scipy tqdm

Usage:
    python rasters_to_parquet.py \\
        --dsm downtown_DSM_aligned.tif \\
        --dtm downtown_DTM_4x.tif \\
        --mrr downtown_MRR_aligned.tif \\
        --out samples.parquet

    # With smoothing (recommended to fix potholes and rough terrain)
    python rasters_to_parquet.py --dsm ... --dtm ... --mrr ... --out ... --smooth-size 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from scipy.ndimage import generic_filter
from scipy.ndimage import median_filter

from tqdm import tqdm


SCHEMA = pa.schema([
    ("x",                  pa.float64()),
    ("y",                  pa.float64()),
    ("dsm",                pa.float32()),
    ("dtm",                pa.float32()),
    ("object_height",      pa.float32()),
    ("multi_return_ratio", pa.float32()),
])


def fill_nans_median(grid: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(grid) | (grid == 0.0)
    if not nan_mask.any():
        return grid

    tmp = np.where(nan_mask, 0.0, grid)
    smoothed = median_filter(tmp, size=3, mode="nearest")

    result = grid.copy()
    result[nan_mask] = smoothed[nan_mask]
    np.nan_to_num(result, nan=0.0, copy=False)
    return result


def read_full(src, nodata):
    """Read entire raster band as float32, replacing nodata with NaN."""
    data = src.read(1).astype(np.float32)
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Combine aligned DSM/DTM/MRR rasters into a Parquet file."
    )
    parser.add_argument("--dsm", required=True, help="DSM GeoTIFF (aligned to DTM)")
    parser.add_argument("--dtm", required=True, help="DTM GeoTIFF (reference grid)")
    parser.add_argument("--mrr", required=True, help="MRR GeoTIFF (aligned to DTM)")
    parser.add_argument("--out", default="samples.parquet", help="Output Parquet file")
    parser.add_argument("--chunk-rows", type=int, default=1000,
                        help="Number of raster rows to process at once when writing (default: 1000)")
    parser.add_argument("--smooth-size", type=int, default=0,
                        help="Median filter kernel size for smoothing DSM and DTM after NaN filling. "
                             "0 = no smoothing (default). Try 5 or 7 to fix potholes and rough terrain.")
    args = parser.parse_args()

    for p in (args.dsm, args.dtm, args.mrr):
        if not Path(p).exists():
            sys.exit(f"Error: file not found: {p}")

    print("Opening rasters ...")
    with (
        rasterio.open(args.dsm) as dsm_src,
        rasterio.open(args.dtm) as dtm_src,
        rasterio.open(args.mrr) as mrr_src,
    ):
        height    = dtm_src.height
        width     = dtm_src.width
        transform = dtm_src.transform

        print(f"  Grid size  : {width} x {height} = {width * height:,} pixels")
        print(f"  Resolution : {transform.a:.4f} x {abs(transform.e):.4f} m")

        print("Reading rasters into memory ...")
        dsm = read_full(dsm_src, dsm_src.nodata)
        dtm = read_full(dtm_src, dtm_src.nodata)
        mrr = read_full(mrr_src, mrr_src.nodata)

    print("Filling NaN/zero values with neighbor median ...")
    print("  Filling DSM ...")
    dsm = fill_nans_median(dsm)
    print("  Filling DTM ...")
    dtm = fill_nans_median(dtm)
    print("  Filling MRR ...")
    mrr = fill_nans_median(mrr)

    # Zero out any remaining NaN (completely isolated cells)
    np.nan_to_num(dsm, nan=0.0, copy=False)
    np.nan_to_num(dtm, nan=0.0, copy=False)
    np.nan_to_num(mrr, nan=0.0, copy=False)

    # --- Optional smoothing ---
    if args.smooth_size > 0:
        print(f"Smoothing DSM and DTM with median filter (size={args.smooth_size}) ...")
        print("  Smoothing DSM ...")
        dsm = median_filter(dsm, size=args.smooth_size, mode="nearest")
        print("  Smoothing DTM ...")
        dtm = median_filter(dtm, size=args.smooth_size, mode="nearest")
        print("  Done smoothing.")

    print(f"\nWriting {args.out} ...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    cols_arr = np.arange(width, dtype=np.float64)

    with pq.ParquetWriter(str(out_path), SCHEMA, compression="snappy") as writer:
        for row_start in tqdm(range(0, height, args.chunk_rows),
                              desc="Writing", unit="row-batch"):
            row_end = min(row_start + args.chunk_rows, height)
            n_rows  = row_end - row_start

            rows_arr = np.arange(row_start, row_end, dtype=np.float64)
            col_grid, row_grid = np.meshgrid(cols_arr, rows_arr)

            xs = (transform.c + col_grid * transform.a).ravel()
            ys = (transform.f + row_grid * transform.e).ravel()

            dsm_chunk = dsm[row_start:row_end, :].ravel()
            dtm_chunk = dtm[row_start:row_end, :].ravel()
            mrr_chunk = mrr[row_start:row_end, :].ravel()
            oh_chunk  = (dsm_chunk - dtm_chunk).clip(min=0.0)

            total_written += len(xs)

            writer.write_batch(pa.record_batch(
                {
                    "x":                  pa.array(xs),
                    "y":                  pa.array(ys),
                    "dsm":                pa.array(dsm_chunk),
                    "dtm":                pa.array(dtm_chunk),
                    "object_height":      pa.array(oh_chunk),
                    "multi_return_ratio": pa.array(mrr_chunk),
                },
                schema=SCHEMA,
            ))

    print(f"\nDone.")
    print(f"  Rows written : {total_written:,}")
    print(f"  File size    : {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
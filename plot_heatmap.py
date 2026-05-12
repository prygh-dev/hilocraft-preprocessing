#!/usr/bin/env python3
"""
plot_heatmap.py

Plot a heatmap of any column from a Parquet or binary file using x/y as coordinates.

Usage:
    python plot_heatmap.py samples.parquet object_height
    python plot_heatmap.py samples.bin --binary --column object_height
    python plot_heatmap.py samples.parquet dtm --cmap terrain
    python plot_heatmap.py samples.parquet --list
"""

import argparse
import sys
import struct

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

MAGIC = 0x4C494441

# Column order and format in the binary file
# d = float64 (8 bytes), f = float32 (4 bytes)
BINARY_COLUMNS = ["x", "y", "dtm", "object_height", "dsm", "multi_return_ratio", "is_street", "is_building", "is_building_edge", "wall_depth_below"]
BINARY_FORMATS = [">d", ">d", ">d", ">d", ">d", ">d", ">f", ">f", ">f", ">f"]
BINARY_SIZES   = [8, 8, 8, 8, 8, 8, 4, 4, 4, 4]
ROW_BYTES      = sum(BINARY_SIZES)


def read_binary(filepath):
    arrays = {col: [] for col in BINARY_COLUMNS}
    with open(filepath, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != MAGIC:
            sys.exit("Bad magic number — not a valid binary file.")
        row_count = struct.unpack(">I", f.read(4))[0]
        print(f"Row count: {row_count:,}")

        for i in range(row_count):
            for col, fmt, size in zip(BINARY_COLUMNS, BINARY_FORMATS, BINARY_SIZES):
                val, = struct.unpack(fmt, f.read(size))
                arrays[col].append(val)

            if i % 500_000 == 0:
                print(f"  Read {i:,} / {row_count:,} ...", end="\r")

    print()
    return {col: np.array(arrays[col]) for col in BINARY_COLUMNS}


def main():
    parser = argparse.ArgumentParser(description="Plot a heatmap of a Parquet or binary column.")
    parser.add_argument("file",   help="Input Parquet or binary file")
    parser.add_argument("column", nargs="?", default=None, help="Column to plot")
    parser.add_argument("--binary", action="store_true", help="Read file as binary instead of Parquet")
    parser.add_argument("--cmap",   default="inferno",   help="Matplotlib colormap (default: inferno)")
    parser.add_argument("--title",  default=None,        help="Plot title (default: column name)")
    parser.add_argument("--list",   action="store_true", help="List available columns and exit")
    args = parser.parse_args()

    if args.binary:
        data    = read_binary(args.file)
        columns = BINARY_COLUMNS
    else:
        df      = pq.read_table(args.file).to_pandas()
        data    = {col: df[col].values for col in df.columns}
        columns = list(df.columns)

    if args.list or args.column is None:
        print("Columns:", ", ".join(columns))
        if args.column is None and not args.list:
            print("Provide a column name as the second argument.")
        return

    if args.column not in data:
        sys.exit(f"Column '{args.column}' not found. Available: {', '.join(columns)}")

    x = data["x"]
    y = data["y"]
    z = data[args.column].astype(np.float64)

    # Filter NaNs
    valid = ~np.isnan(z)
    n_nan = (~valid).sum()
    if n_nan > 0:
        print(f"  Skipping {n_nan:,} NaN values in '{args.column}'")
    x = x[valid]
    y = y[valid]
    z = z[valid]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("red")

    sc = ax.scatter(x, y, c=z, cmap=args.cmap, s=0.5, linewidths=0)
    plt.colorbar(sc, ax=ax, label=args.column)
    ax.set_title(args.title or args.column)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
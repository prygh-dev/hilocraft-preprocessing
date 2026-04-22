#!/usr/bin/env python3
"""
sort_by_chunk.py

Sorts a parquet file by 16x16 block sections (8x8 metres at 0.5m resolution).
All rows belonging to the same section are written contiguously.

Usage:
    python sort_by_chunk.py input.parquet output.parquet
"""

import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    if len(sys.argv) < 3:
        print("Usage: python sort_by_chunk.py input.parquet output.parquet")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Reading {in_path} ...")
    df = pq.read_table(in_path).to_pandas()
    print(f"  Rows: {len(df):,}")

    x = df["x"].values
    y = df["y"].values

    min_x = x.min()
    max_y = y.max()

    # Compute which 16x16 block section each row belongs to.
    # Each block = 0.5m, so 16 blocks = 8m.
    # section_x = mcX // 16 = ((x - min_x) * 2) // 16 = (x - min_x) // 8
    section_x = ((x - min_x) / 8.0).astype(int)
    section_z = ((max_y - y) / 8.0).astype(int)

    df["_sx"] = section_x
    df["_sz"] = section_z

    print("Sorting by section ...")
    df = df.sort_values(["_sx", "_sz"]).drop(columns=["_sx", "_sz"])

    print(f"Writing {out_path} ...")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path, compression="snappy")

    print(f"Done. {len(df):,} rows written.")

if __name__ == "__main__":
    main()
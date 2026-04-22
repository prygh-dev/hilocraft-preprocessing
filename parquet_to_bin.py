#!/usr/bin/env python3
"""
parquet_to_binary.py

Converts a processed parquet file to a simple raw binary format
that can be read from Java without any parquet library.

Binary format:
    [4 bytes]  magic number: 0x4C494441  ("LIDA")
    [4 bytes]  number of rows (int32, big-endian)
    [N * 28 bytes] row data, each row:
        [8 bytes] x                 (float64, big-endian)
        [8 bytes] y                 (float64, big-endian)
        [4 bytes] dtm               (float32, big-endian)
        [4 bytes] object_height     (float32, big-endian)
        [4 bytes] multi_return_ratio (float32, big-endian)

Big-endian matches Java's DataInputStream defaults.

Dependencies:
    pip install pyarrow numpy

Usage:
    python parquet_to_binary.py input.parquet output.bin
    python parquet_to_binary.py input.parquet output.bin --no-mrr
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


MAGIC = 0x4C494441   # "LIDA"
ROW_BYTES = 56       # 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4


def main():
    parser = argparse.ArgumentParser(
        description="Convert a processed parquet file to raw binary for Java."
    )
    parser.add_argument("parquet", help="Input parquet file")
    parser.add_argument("output",  help="Output binary file")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    output_path  = Path(args.output)

    if not parquet_path.exists():
        sys.exit(f"Error: file not found: {parquet_path}")

    print(f"Reading {parquet_path} ...")
    df = pq.read_table(parquet_path).to_pandas()
    n  = len(df)
    print(f"  Rows: {n:,}")

    for col in ("x", "y", "dtm", "object_height", "dsm", "multi_return_ratio", "is_street", "is_building"):
        if col not in df.columns:
            sys.exit(f"Error: expected column '{col}' not found in parquet.")

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    dtm = df["dtm"].to_numpy(dtype=np.float64)
    oh = df["object_height"].to_numpy(dtype=np.float64)
    dsm = df["dsm"].to_numpy(dtype=np.float64)
    mrr = df["multi_return_ratio"].to_numpy(dtype=np.float64)
    is_street = df["is_street"].to_numpy(dtype=np.float32)
    is_building = df["is_building"].to_numpy(dtype=np.float32)

    print(f"Writing {output_path} ...")
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack(">I", MAGIC))   # 4 bytes magic
        f.write(struct.pack(">I", n))       # 4 bytes row count

        # Rows — pack in chunks of 100k for speed
        chunk = 100_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            buf = bytearray()
            for i in range(start, end):
                buf += struct.pack(">dddddd ff",
                    x[i], y[i], dtm[i], oh[i], dsm[i], mrr[i], is_street[i], is_building[i])
            f.write(buf)
            if (start // chunk) % 10 == 0:
                print(f"  {end:,} / {n:,} rows written ...", end="\r")

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\nDone. {output_path}  ({size_mb:.1f} MB)")
    print(f"  {n:,} rows × {ROW_BYTES} bytes = {n * ROW_BYTES / 1_048_576:.1f} MB data")
    print(f"  + 8 bytes header")


if __name__ == "__main__":
    main()
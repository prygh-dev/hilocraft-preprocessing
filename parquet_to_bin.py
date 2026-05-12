#!/usr/bin/env python3
"""
Binary format (64 bytes per row):
    [4 bytes]  magic 0x4C494441
    [4 bytes]  row count (int32 BE)
    Per row:
        [8] x, y, dtm, object_height, dsm, mrr (float64 BE)
        [4] is_street, is_building, is_building_edge, wall_depth_below (float32 BE)
"""

import argparse, struct, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

MAGIC = 0x4C494441
ROW_BYTES = 64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("output")
    args = parser.parse_args()

    df = pq.read_table(args.parquet).to_pandas()
    n = len(df)
    print(f"Rows: {n:,}")

    if "is_building_edge" not in df.columns:
        df["is_building_edge"] = 0.0
    if "wall_depth_below" not in df.columns:
        df["wall_depth_below"] = 0.0

    x  = df["x"].to_numpy(np.float64)
    y  = df["y"].to_numpy(np.float64)
    dtm = df["dtm"].to_numpy(np.float64)
    oh = df["object_height"].to_numpy(np.float64)
    dsm = df["dsm"].to_numpy(np.float64)
    mrr = df["multi_return_ratio"].to_numpy(np.float64)
    iss = df["is_street"].to_numpy(np.float32)
    isb = df["is_building"].to_numpy(np.float32)
    ise = df["is_building_edge"].to_numpy(np.float32)
    wdb = df["wall_depth_below"].to_numpy(np.float32)

    with open(args.output, "wb") as f:
        f.write(struct.pack(">I", MAGIC))
        f.write(struct.pack(">I", n))
        chunk = 100_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            buf = bytearray()
            for i in range(start, end):
                buf += struct.pack(">dddddd ffff",
                    x[i], y[i], dtm[i], oh[i], dsm[i], mrr[i],
                    iss[i], isb[i], ise[i], wdb[i])
            f.write(buf)
            print(f"  {end:,} / {n:,}", end="\r")
    print(f"\nDone -> {args.output}")

if __name__ == "__main__":
    main()
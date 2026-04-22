import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import sys


if(len(sys.argv) < 4):
    print("Usage: python rotate_parquet.py <input_parquet> <degrees> <output_parquet>")
    sys.exit(1)

in_parquet = sys.argv[1]
ANGLE_DEG = int(sys.argv[2])
out_parquet = sys.argv[3]

angle = np.radians(ANGLE_DEG)
cos_a, sin_a = np.cos(angle), np.sin(angle)

df = pq.read_table(in_parquet).to_pandas()

cx = df["x"].mean()
cy = df["y"].mean()

dx = df["x"] - cx
dy = df["y"] - cy

df["x"] = dx * cos_a - dy * sin_a + cx
df["y"] = dx * sin_a + dy * cos_a + cy

pq.write_table(pa.Table.from_pandas(df), out_parquet)
print("Done.")
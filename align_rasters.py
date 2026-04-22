#!/usr/bin/env python3
"""
align_raster.py

Reprojects and resamples a source raster to exactly match the grid
(extent, resolution, CRS) of a reference raster. Use this to align
DTM and MRR to the DSM grid before sampling.

Dependencies:
    pip install rasterio numpy

Usage:
    python align_raster.py reference.tif source.tif output.tif

Example:
    python align_raster.py downtown_DSM_4x.tif downtown_DTM_4x.tif downtown_DTM_aligned.tif
    python align_raster.py downtown_DSM_4x.tif downtown_MRR_4x.tif downtown_MRR_aligned.tif
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def main():
    if len(sys.argv) < 4:
        print("Usage: python align_raster.py reference.tif source.tif output.tif")
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    src_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    for p in (ref_path, src_path):
        if not p.exists():
            sys.exit(f"Error: file not found: {p}")

    print(f"Reference : {ref_path}")
    print(f"Source    : {src_path}")
    print(f"Output    : {out_path}")

    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs       = ref.crs
        ref_height    = ref.height
        ref_width     = ref.width
        ref_profile   = ref.profile.copy()
        print(f"  Reference grid : {ref_width} x {ref_height}  res={ref.res}")

    with rasterio.open(src_path) as src:
        print(f"  Source grid    : {src.width} x {src.height}  res={src.res}")
        nodata = src.nodata if src.nodata is not None else -9999.0

        dest = np.full((ref_height, ref_width), nodata, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            dst_nodata=nodata,
        )

    ref_profile.update(
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **ref_profile) as dst:
        dst.write(dest, 1)

    print(f"Done. Saved -> {out_path}")


if __name__ == "__main__":
    main()
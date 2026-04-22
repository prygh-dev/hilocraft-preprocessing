#!/usr/bin/env python3
"""
las_to_dsm_dtm.py

Converts a LAS/LAZ file with pre-classified ground points (class code 2)
into DSM and DTM GeoTIFFs at the maximum resolution the dataset supports.

Dependencies:
    pip install "laspy[lazrs]" numpy scipy rasterio

Usage:
    python las_to_dsm_dtm.py input.las
    python las_to_dsm_dtm.py input.las --res 0.5 --out-dir ./output --fill-voids
    python las_to_dsm_dtm.py input.las --epsg 32604   # if CRS not embedded in file
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.interpolate import NearestNDInterpolator


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def compute_optimal_resolution(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Estimate the ideal raster resolution from average point spacing.

    Derivation:
        density  = N / area                   (points per unit²)
        spacing  = 1 / sqrt(density)          (average distance between neighbors)
        → one cell per average spacing keeps every point in a distinct cell on average.

    Returns (resolution, density).
    """
    area = (x.max() - x.min()) * (y.max() - y.min())
    if area == 0:
        raise ValueError("Point cloud has zero area — cannot compute resolution.")
    density = len(x) / area
    res = 1.0 / np.sqrt(density)
    return res, density


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_max_z(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_min: float,
    y_max: float,
    res: float,
    cols: int,
    rows: int,
) -> np.ndarray:
    """
    Assign each point to a grid cell (col, row) and record the maximum Z.

    Strategy: sort points by Z ascending, then write Z values into the grid
    array sequentially. Because later writes overwrite earlier ones, the final
    value in each cell is the maximum Z seen — no explicit groupby needed.

    Returns a float32 array of shape (rows, cols) with np.nan for empty cells.
    """
    grid = np.full((rows, cols), np.nan, dtype=np.float32)

    col_idx = np.floor((x - x_min) / res).astype(np.int32)
    row_idx = np.floor((y_max - y) / res).astype(np.int32)

    # Clamp for floating-point edge points landing exactly on the boundary
    np.clip(col_idx, 0, cols - 1, out=col_idx)
    np.clip(row_idx, 0, rows - 1, out=row_idx)

    # Sort ascending by Z so that the max Z is written last (and sticks)
    sort_order = np.argsort(z)
    grid[row_idx[sort_order], col_idx[sort_order]] = z[sort_order].astype(np.float32)

    return grid


# ---------------------------------------------------------------------------
# Void filling
# ---------------------------------------------------------------------------

def fill_voids(grid: np.ndarray) -> np.ndarray:
    """
    Fill NaN cells via nearest-neighbor interpolation from valid neighbors.

    This is appropriate for DTM void filling where sparse ground points leave
    gaps. It preserves all existing values and only fills empty cells.
    """
    valid = ~np.isnan(grid)
    if valid.all():
        return grid

    rows, cols = grid.shape
    row_grid, col_grid = np.mgrid[0:rows, 0:cols]

    interp = NearestNDInterpolator(
        np.column_stack([row_grid[valid], col_grid[valid]]),
        grid[valid],
    )

    empty = ~valid
    grid = grid.copy()
    grid[empty] = interp(row_grid[empty], col_grid[empty])
    return grid


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_las(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CRS | None]:
    """
    Read a LAS/LAZ file and return (x, y, z, classification, crs).

    laspy 2.x ScaledArrayViews already apply the scale + offset from the
    header, so np.asarray() gives true geographic coordinates.
    """
    print(f"Reading {path} ...")
    las = laspy.read(path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    classification = np.asarray(las.classification, dtype=np.uint8)

    crs = None
    try:
        parsed = las.header.parse_crs()
        if parsed is not None:
            crs = CRS.from_user_input(parsed.to_wkt())
    except Exception:
        pass

    return x, y, z, classification, crs


def write_geotiff(
    path: Path,
    grid: np.ndarray,
    transform,
    crs: CRS | None,
    nodata: float = -9999.0,
) -> None:
    """Write a float32 grid to a LZW-compressed GeoTIFF."""
    out = grid.copy().astype(np.float32)
    out[np.isnan(out)] = nodata

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(out, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a classified LAS/LAZ file to DSM and DTM GeoTIFFs."
    )
    parser.add_argument("input", help="Path to input LAS/LAZ file")
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help="Output cell size in CRS units (default: auto from point density)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory for output GeoTIFFs (default: current directory)",
    )
    parser.add_argument(
        "--fill-voids",
        action="store_true",
        help="Fill empty cells using nearest-neighbor interpolation",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=None,
        help="EPSG code to assign if the file carries no CRS (e.g. 32604 for UTM zone 4N)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: file not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Read ---
    x, y, z, classification, crs = read_las(input_path)

    print(f"  Total points : {len(x):,}")
    print(f"  X range      : {x.min():.3f} – {x.max():.3f}")
    print(f"  Y range      : {y.min():.3f} – {y.max():.3f}")
    print(f"  Z range      : {z.min():.3f} – {z.max():.3f}")

    # CRS resolution
    if crs is not None:
        print(f"  CRS (file)   : EPSG:{crs.to_epsg() or 'unknown'}")
    elif args.epsg:
        crs = CRS.from_epsg(args.epsg)
        print(f"  CRS (--epsg) : EPSG:{args.epsg}")
    else:
        print(
            "  Warning: No CRS found. Output will be ungeoreferenced.\n"
            "           Re-run with --epsg <code> to assign a projection."
        )

    # --- Resolution ---
    auto_res, density = compute_optimal_resolution(x, y)
    res = args.res if args.res is not None else auto_res

    print(f"\n  Point density  : {density:.4f} pts/unit²")
    print(f"  Auto res       : {auto_res:.4f} units/px")
    print(f"  Using res      : {res:.4f} units/px")

    # --- Grid geometry ---
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    cols = int(np.ceil((x_max - x_min) / res)) + 1
    rows = int(np.ceil((y_max - y_min) / res)) + 1
    print(f"  Grid size      : {rows} rows × {cols} cols  ({rows * cols / 1e6:.1f}M cells)")

    transform = from_origin(x_min, y_max, res, res)

    # --- DSM ---
    print("\n[DSM] Rasterizing all points (max Z per cell) ...")
    dsm = rasterize_max_z(x, y, z, x_min, y_max, res, cols, rows)
    void_pct = 100.0 * np.isnan(dsm).sum() / dsm.size
    print(f"      Void cells : {void_pct:.2f}%")
    if args.fill_voids and void_pct > 0:
        print("      Filling voids ...")
        dsm = fill_voids(dsm)

    # --- DTM ---
    print("\n[DTM] Filtering to class-2 ground points ...")
    ground = classification == 2
    n_ground = ground.sum()
    print(f"      Ground points : {n_ground:,} ({100.0 * n_ground / len(x):.1f}% of total)")

    if n_ground == 0:
        sys.exit(
            "Error: No class-2 (ground) points found.\n"
            "       Check your LAS classification before running."
        )

    print("      Rasterizing (max Z per cell) ...")
    dtm = rasterize_max_z(
        x[ground], y[ground], z[ground],
        x_min, y_max, res, cols, rows,
    )
    void_pct = 100.0 * np.isnan(dtm).sum() / dtm.size
    print(f"      Void cells : {void_pct:.2f}%")
    if void_pct > 0:
        # DTM almost always needs void filling — ground points are sparser
        print("      Filling voids (recommended for DTM — use --fill-voids to apply to DSM too) ...")
        dtm = fill_voids(dtm)

    # --- Write ---
    stem = input_path.stem
    dsm_path = out_dir / f"{stem}_DSM.tif"
    dtm_path = out_dir / f"{stem}_DTM.tif"

    print(f"\nWriting DSM → {dsm_path}")
    write_geotiff(dsm_path, dsm, transform, crs)

    print(f"Writing DTM → {dtm_path}")
    write_geotiff(dtm_path, dtm, transform, crs)

    print("\nDone.")


if __name__ == "__main__":
    main()
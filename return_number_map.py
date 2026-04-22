#!/usr/bin/env python3
"""
plot_return_map.py

Reads a LAS/LAZ file and produces two rasters + heatmaps derived from
pulse return data:

  1. Multi-return ratio  — fraction of points per cell where number_of_returns > 1.
                           High values → vegetation (laser penetrates canopy layers).
                           Near zero   → hard surfaces like building rooftops.

  2. Mean return number  — average return_number per cell.
                           Complements the ratio: late returns (2nd, 3rd…) buried
                           inside a canopy push the mean up even when the ratio
                           is moderate.

Both are saved as GeoTIFFs and optionally plotted side-by-side.

Dependencies:
    pip install "laspy[lazrs]" numpy scipy rasterio matplotlib

Usage:
    python plot_return_map.py input.las
    python plot_return_map.py input.las --res 0.5 --out-dir ./output --save returns.png
    python plot_return_map.py input.las --epsg 32604 --no-plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Resolution (same logic as las_to_dsm_dtm.py)
# ---------------------------------------------------------------------------

def compute_optimal_resolution(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    area = (x.max() - x.min()) * (y.max() - y.min())
    if area == 0:
        raise ValueError("Point cloud has zero area.")
    density = len(x) / area
    return 1.0 / np.sqrt(density), density


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_return_stats(
    x: np.ndarray,
    y: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    x_min: float,
    y_max: float,
    res: float,
    cols: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each grid cell accumulate:
      - sum of (number_of_returns > 1)  → numerator for multi-return ratio
      - sum of return_number values      → numerator for mean return number
      - count of points                  → denominator for both

    Returns:
        multi_return_ratio  : float32 (rows × cols), NaN where empty
        mean_return_number  : float32 (rows × cols), NaN where empty
    """
    count          = np.zeros((rows, cols), dtype=np.int32)
    multi_sum      = np.zeros((rows, cols), dtype=np.int32)   # points where n_returns > 1
    return_num_sum = np.zeros((rows, cols), dtype=np.float64) # sum of return_number

    col_idx = np.clip(np.floor((x - x_min) / res).astype(np.int32), 0, cols - 1)
    row_idx = np.clip(np.floor((y_max - y) / res).astype(np.int32), 0, rows - 1)

    # Flat index for np.add.at (handles duplicate cell writes correctly)
    flat_idx = row_idx * cols + col_idx

    np.add.at(count.ravel(),          flat_idx, 1)
    np.add.at(multi_sum.ravel(),      flat_idx, (number_of_returns > 1).astype(np.int32))
    np.add.at(return_num_sum.ravel(), flat_idx, return_number.astype(np.float64))

    empty = count == 0

    multi_ratio = np.where(empty, np.nan, multi_sum  / np.where(empty, 1, count)).astype(np.float32)
    mean_ret    = np.where(empty, np.nan, return_num_sum / np.where(empty, 1, count)).astype(np.float32)

    return multi_ratio, mean_ret


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def build_vegetation_cmap() -> mcolors.LinearSegmentedColormap:
    """
    Cool (grey-blue) → warm green → vivid green for multi-return ratio.
    Near-zero (buildings/pavement) = cool grey.
    High (dense canopy) = saturated green.
    """
    colors = [
        (0.15, 0.15, 0.25),   # dark blue-grey  – single return / hard surface
        (0.20, 0.35, 0.45),   # slate blue
        (0.15, 0.55, 0.45),   # teal
        (0.25, 0.72, 0.25),   # medium green
        (0.10, 0.90, 0.10),   # vivid green     – dense multi-return canopy
    ]
    return mcolors.LinearSegmentedColormap.from_list("veg_ratio", colors)


def build_return_num_cmap() -> mcolors.LinearSegmentedColormap:
    """
    Purple → blue → cyan → yellow for mean return number.
    1st return only = purple (hard surface).
    Late returns = yellow (deep canopy penetration).
    """
    colors = [
        (0.35, 0.10, 0.50),   # purple   – first return dominated
        (0.10, 0.25, 0.80),   # blue
        (0.05, 0.70, 0.85),   # cyan
        (0.85, 0.90, 0.15),   # yellow   – high mean return number
    ]
    return mcolors.LinearSegmentedColormap.from_list("return_num", colors)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_geotiff(path: Path, grid: np.ndarray, transform, crs, nodata: float = -9999.0):
    out = grid.copy()
    out[np.isnan(out)] = nodata
    with rasterio.open(
        path, "w",
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
        dst.write(out.astype(np.float32), 1)


def build_extent(x_min, y_min, x_max, y_max):
    return x_min, x_max, y_min, y_max


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_return_maps(
    multi_ratio: np.ndarray,
    mean_ret: np.ndarray,
    extent: tuple,
    stem: str,
    save_path: Path | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
    fig.patch.set_facecolor("#12121f")

    bg = "#0a0a16"

    panels = [
        {
            "ax"   : axes[0],
            "data" : multi_ratio,
            "cmap" : build_vegetation_cmap(),
            "vmin" : 0.0,
            "vmax" : 1.0,
            "label": "Multi-return ratio  (0 = all single returns, 1 = all multi-return)",
            "title": "Multi-Return Ratio per Cell",
            "note" : "High → likely vegetation\nLow → likely hard surface (building / road)",
        },
        {
            "ax"   : axes[1],
            "data" : mean_ret,
            "cmap" : build_return_num_cmap(),
            "vmin" : 1.0,
            "vmax" : np.nanpercentile(mean_ret, 99),
            "label": "Mean return number",
            "title": "Mean Return Number per Cell",
            "note" : "High → late / deep returns (penetrating canopy)\nLow → first-return dominated (rooftop / pavement)",
        },
    ]

    for p in panels:
        ax = p["ax"]
        ax.set_facecolor(bg)

        im = ax.imshow(
            p["data"],
            cmap=p["cmap"],
            vmin=p["vmin"],
            vmax=p["vmax"],
            extent=extent,
            origin="upper",
            interpolation="bilinear",
            aspect="equal",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(p["label"], color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(p["title"], color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.set_xlabel("Easting", color="white", fontsize=9)
        ax.set_ylabel("Northing", color="white", fontsize=9)

        # Annotation note in bottom-left corner
        ax.text(
            0.02, 0.03, p["note"],
            transform=ax.transAxes,
            color="#aaaacc", fontsize=7.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.7, edgecolor="none"),
        )

    fig.suptitle(
        f"LiDAR Return Analysis — {stem}",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map LiDAR return statistics to a raster and heatmap."
    )
    parser.add_argument("input", help="Path to input LAS/LAZ file")
    parser.add_argument("--res",     type=float, default=None,
                        help="Cell size in CRS units (default: auto from point density)")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--save",    default=None,
                        help="Save plot to this path (e.g. returns.png); omit to display")
    parser.add_argument("--epsg",    type=int, default=None,
                        help="EPSG code if not embedded in LAS file")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting, only write GeoTIFFs")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: file not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Read ---
    print(f"Reading {input_path} ...")
    las = laspy.read(input_path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    return_number    = np.asarray(las.return_number,    dtype=np.uint8)
    number_of_returns = np.asarray(las.number_of_returns, dtype=np.uint8)

    print(f"  Total points      : {len(x):,}")
    print(f"  Return number     : min={return_number.min()}  max={return_number.max()}")
    print(f"  Number of returns : min={number_of_returns.min()}  max={number_of_returns.max()}")
    multi_pct = 100.0 * (number_of_returns > 1).sum() / len(x)
    print(f"  Multi-return pts  : {multi_pct:.1f}% of total")

    # CRS
    crs = None
    try:
        parsed = las.header.parse_crs()
        if parsed is not None:
            crs = CRS.from_user_input(parsed.to_wkt())
            print(f"  CRS (file)        : EPSG:{crs.to_epsg()}")
    except Exception:
        pass
    if crs is None and args.epsg:
        crs = CRS.from_epsg(args.epsg)
        print(f"  CRS (--epsg)      : EPSG:{args.epsg}")
    elif crs is None:
        print("  Warning: No CRS found. Use --epsg to assign one.")

    # Resolution + grid
    auto_res, density = compute_optimal_resolution(x, y)
    res = args.res if args.res is not None else auto_res
    print(f"\n  Point density  : {density:.4f} pts/unit²")
    print(f"  Using res      : {res:.4f} units/px")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    cols = int(np.ceil((x_max - x_min) / res)) + 1
    rows = int(np.ceil((y_max - y_min) / res)) + 1
    print(f"  Grid size      : {rows} rows × {cols} cols")

    transform = from_origin(x_min, y_max, res, res)

    # --- Rasterize ---
    print("\nRasterizing return statistics ...")
    multi_ratio, mean_ret = rasterize_return_stats(
        x, y, return_number, number_of_returns,
        x_min, y_max, res, cols, rows,
    )

    valid = ~np.isnan(multi_ratio)
    print(f"  Multi-return ratio : mean={np.nanmean(multi_ratio):.3f}  max={np.nanmax(multi_ratio):.3f}")
    print(f"  Mean return number : mean={np.nanmean(mean_ret):.3f}  max={np.nanmax(mean_ret):.3f}")

    # --- Write GeoTIFFs ---
    stem = input_path.stem
    mr_path  = out_dir / f"{stem}_multi_return_ratio.tif"
    mrn_path = out_dir / f"{stem}_mean_return_number.tif"

    print(f"\nWriting → {mr_path}")
    write_geotiff(mr_path, multi_ratio, transform, crs)

    print(f"Writing → {mrn_path}")
    write_geotiff(mrn_path, mean_ret, transform, crs)

    # --- Plot ---
    if not args.no_plot:
        print("\nPlotting ...")
        extent = build_extent(x_min, y_min, x_max, y_max)
        save_path = Path(args.save) if args.save else None
        plot_return_maps(multi_ratio, mean_ret, extent, stem, save_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
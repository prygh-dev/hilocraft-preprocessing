#!/usr/bin/env python3
"""
apply_osm_mask.py

Adds is_building, is_street, is_building_edge, and wall_depth_below columns.
Writes output parquet in chunks to avoid memory issues on large datasets.
"""

import sys
import argparse
import gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box as shapely_box
from pyproj import Transformer
from scipy.ndimage import binary_erosion, minimum_filter


ROAD_HALF_WIDTHS = {
    "motorway":       6.0,
    "trunk":          5.0,
    "primary":        4.5,
    "secondary":      4.0,
    "tertiary":       3.5,
    "residential":    3.0,
    "service":        2.5,
    "unclassified":   3.0,
    "footway":        1.5,
    "path":           1.0,
    "cycleway":       1.5,
    "living_street":  2.5,
}
DEFAULT_HALF_WIDTH = 3.0


def unrotate(x, y, angle_deg):
    angle = np.radians(-angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    x_out = dx * cos_a - dy * sin_a + cx
    y_out = dx * sin_a + dy * cos_a + cy
    return x_out, y_out


def fetch_buildings(bbox_polygon):
    print("Fetching OSM building footprints ...")
    buildings = ox.features_from_polygon(bbox_polygon, tags={"building": True})
    if buildings.empty:
        return None
    buildings = buildings.to_crs("EPSG:6635")
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings[["geometry"]].reset_index(drop=True)
    print(f"  {len(buildings)} polygon footprints")
    return buildings


def fetch_streets(bbox_polygon):
    print("Fetching OSM road centerlines ...")
    roads = ox.features_from_polygon(bbox_polygon, tags={"highway": True})
    if roads.empty:
        return None
    roads = roads.to_crs("EPSG:6635")
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads[["geometry", "highway"]].reset_index(drop=True)
    roads["highway"] = roads["highway"].apply(lambda h: h[0] if isinstance(h, list) else h)
    roads["half_width"] = roads["highway"].map(ROAD_HALF_WIDTHS).fillna(DEFAULT_HALF_WIDTH)
    roads["geometry"]   = roads.apply(lambda r: r.geometry.buffer(r["half_width"]), axis=1)
    roads = roads[["geometry"]].reset_index(drop=True)
    print(f"  {len(roads)} road polygons")
    return roads


def spatial_join_mask(gdf_points, polygons, label, chunk_size=1_000_000):
    print(f"Spatial join for {label} ...")
    inside = np.zeros(len(gdf_points), dtype=bool)
    for start in range(0, len(gdf_points), chunk_size):
        end = min(start + chunk_size, len(gdf_points))
        chunk = gdf_points.iloc[start:end].copy()
        joined = gpd.sjoin(chunk, polygons, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        inside[start:end] = ~joined["index_right"].isna()
        print(f"  {end:,} / {len(gdf_points):,} ...", end="\r")
    print(f"\n  Points {label}: {inside.sum():,}")
    return inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--res", type=float, default=0.25)
    parser.add_argument("--rotation-deg", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Rows per spatial join chunk")
    parser.add_argument("--write-chunk-size", type=int, default=5_000_000,
                        help="Rows per parquet write chunk (default: 5000000)")
    parser.add_argument("--erode-iterations", type=int, default=1)
    parser.add_argument("--height-drop-threshold", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pq.read_table(args.input).to_pandas()
    print(f"  Rows: {len(df):,}")

    x_orig = df["x"].values.copy()
    y_orig = df["y"].values.copy()

    if args.rotation_deg is not None:
        print(f"Unrotating by {args.rotation_deg} degrees ...")
        x_geo, y_geo = unrotate(x_orig, y_orig, args.rotation_deg)
    else:
        x_geo, y_geo = x_orig, y_orig

    transformer = Transformer.from_crs("EPSG:6635", "EPSG:4326", always_xy=True)
    lons, lats  = transformer.transform(x_geo, y_geo)
    bbox_polygon = shapely_box(lons.min(), lats.min(), lons.max(), lats.max())
    del lons, lats

    buildings = fetch_buildings(bbox_polygon)
    streets   = fetch_streets(bbox_polygon)

    gdf_points = gpd.GeoDataFrame(
        {"row_idx": np.arange(len(df))},
        geometry=gpd.points_from_xy(x_geo, y_geo),
        crs="EPSG:6635"
    )

    if streets is not None:
        inside_street = spatial_join_mask(gdf_points, streets, "inside streets", args.chunk_size)
    else:
        inside_street = np.zeros(len(df), dtype=bool)

    if buildings is not None:
        inside_building = spatial_join_mask(gdf_points, buildings, "inside buildings", args.chunk_size)
        df.loc[~inside_building & ~inside_street, "object_height"] = 0.0
    else:
        inside_building = np.zeros(len(df), dtype=bool)

    #if a point is both street and building, treat as street only
    overlap = inside_building & inside_street
    if overlap.any():
        print(f"  {overlap.sum():,} points are both street and building — treating as street")
        inside_building[overlap] = False

    del gdf_points
    gc.collect()

    # --- Compute edge mask + wall depths on a 2D grid ---
    print("Computing edge mask + wall depths ...")
    res = args.res
    x_min = x_geo.min()
    y_min = y_geo.min()
    col_idx = np.round((x_geo - x_min) / res).astype(int)
    row_idx = np.round((y_geo - y_min) / res).astype(int)
    n_cols = col_idx.max() + 1
    n_rows = row_idx.max() + 1
    print(f"  Grid: {n_cols} x {n_rows}")

    del x_orig, y_orig, x_geo, y_geo

    building_grid = np.zeros((n_rows, n_cols), dtype=bool)
    building_grid[row_idx, col_idx] = inside_building

    oh_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    oh_grid[row_idx, col_idx] = df["object_height"].values

    eroded = binary_erosion(building_grid, iterations=args.erode_iterations)
    footprint_edge = building_grid & ~eroded
    del eroded
    print(f"  Footprint edge: {footprint_edge.sum():,}")

    oh_building_only = np.where(building_grid, oh_grid, np.inf)
    neighbor_min = minimum_filter(oh_building_only, size=3, mode="constant", cval=np.inf)
    neighbor_min = np.where(np.isfinite(neighbor_min), neighbor_min, 0.0)
    del oh_building_only

    height_rise = oh_grid - neighbor_min
    height_edge = building_grid & (height_rise > args.height_drop_threshold)
    print(f"  Height edge: {height_edge.sum():,}")

    edge_grid = footprint_edge | height_edge
    inside_building_edge = edge_grid[row_idx, col_idx]
    print(f"  Total edge points: {inside_building_edge.sum():,}")

    wall_depth_grid = np.zeros_like(oh_grid)
    wall_depth_grid[footprint_edge] = oh_grid[footprint_edge]
    height_only = height_edge & ~footprint_edge
    wall_depth_grid[height_only] = height_rise[height_only]
    wall_depth_below = wall_depth_grid[row_idx, col_idx]

    # Free all the 2D grids
    del building_grid, oh_grid, neighbor_min, height_rise, height_edge
    del footprint_edge, edge_grid, wall_depth_grid, height_only
    del col_idx, row_idx
    gc.collect()

    # --- Write parquet in chunks to avoid memory blowup ---
    print(f"Writing {args.output} in chunks of {args.write_chunk_size:,} rows ...")

    schema = pa.schema([
        ("x",                  pa.float64()),
        ("y",                  pa.float64()),
        ("dtm",                pa.float64()),
        ("dsm",                pa.float64()),
        ("object_height",      pa.float64()),
        ("multi_return_ratio", pa.float64()),
        ("is_street",          pa.float32()),
        ("is_building",        pa.float32()),
        ("is_building_edge",   pa.float32()),
        ("wall_depth_below",   pa.float32()),
    ])

    total = len(df)
    cs = args.write_chunk_size

    with pq.ParquetWriter(args.output, schema, compression="snappy") as writer:
        for start in range(0, total, cs):
            end = min(start + cs, total)
            print(f"  Writing rows {start:,} - {end:,} ...", end="\r")

            batch = pa.record_batch({
                "x":                  pa.array(df["x"].values[start:end]),
                "y":                  pa.array(df["y"].values[start:end]),
                "dtm":                pa.array(df["dtm"].values[start:end]),
                "dsm":                pa.array(df["dsm"].values[start:end]),
                "object_height":      pa.array(df["object_height"].values[start:end]),
                "multi_return_ratio": pa.array(df["multi_return_ratio"].values[start:end]),
                "is_street":          pa.array(inside_street[start:end].astype(np.float32)),
                "is_building":        pa.array(inside_building[start:end].astype(np.float32)),
                "is_building_edge":   pa.array(inside_building_edge[start:end].astype(np.float32)),
                "wall_depth_below":   pa.array(wall_depth_below[start:end].astype(np.float32)),
            }, schema=schema)
            writer.write_batch(batch)

    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
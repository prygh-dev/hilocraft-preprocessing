#!/usr/bin/env python3
"""
apply_osm_mask.py

Fetches OSM building footprints and road centerlines for the parquet
bounding box, then adds is_building and is_street columns.

Supports rotated input data — pass --rotation-deg to unrotate before
applying the OSM mask, then transfers flags back to the rotated coords.

Usage:
    # Unrotated data
    python apply_osm_mask.py input.parquet output.parquet

    # Rotated data (pass the same angle used when rotating)
    python apply_osm_mask.py input.parquet output.parquet --rotation-deg 59
"""

import sys
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box as shapely_box
from pyproj import Transformer


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
    """Reverse a rotation applied about the centroid of the data."""
    angle = np.radians(-angle_deg)  # negate to reverse
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
        print("  No buildings found.")
        return None
    print(f"  Found {len(buildings)} OSM features")
    buildings = buildings.to_crs("EPSG:6635")
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings[["geometry"]].reset_index(drop=True)
    print(f"  {len(buildings)} polygon footprints after filtering")
    return buildings


def fetch_streets(bbox_polygon):
    print("Fetching OSM road centerlines ...")
    roads = ox.features_from_polygon(bbox_polygon, tags={"highway": True})
    if roads.empty:
        print("  No roads found.")
        return None
    print(f"  Found {len(roads)} OSM road features")
    roads = roads.to_crs("EPSG:6635")
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads[["geometry", "highway"]].reset_index(drop=True)
    roads["highway"] = roads["highway"].apply(
        lambda h: h[0] if isinstance(h, list) else h
    )
    roads["half_width"] = roads["highway"].map(ROAD_HALF_WIDTHS).fillna(DEFAULT_HALF_WIDTH)
    roads["geometry"]   = roads.apply(lambda r: r.geometry.buffer(r["half_width"]), axis=1)
    roads = roads[["geometry"]].reset_index(drop=True)
    print(f"  {len(roads)} road polygons after buffering")
    return roads


def spatial_join_mask(gdf_points, polygons, label):
    print(f"Running spatial join for {label} ...")
    joined = gpd.sjoin(gdf_points, polygons, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    inside = ~joined["index_right"].isna()
    print(f"  Points {label}: {inside.sum():,}")
    return inside.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",  help="Input parquet file")
    parser.add_argument("output", help="Output parquet file")
    parser.add_argument("--rotation-deg", type=float, default=None,
                        help="Rotation angle in degrees that was applied to the data. "
                             "If provided, coordinates are unrotated before OSM masking.")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pq.read_table(args.input).to_pandas()
    print(f"  Rows: {len(df):,}")

    x_orig = df["x"].values.copy()
    y_orig = df["y"].values.copy()

    # --- Unrotate if needed ---
    if args.rotation_deg is not None:
        print(f"Unrotating by {args.rotation_deg} degrees ...")
        x_geo, y_geo = unrotate(x_orig, y_orig, args.rotation_deg)
        print(f"  Unrotated bbox: ({x_geo.min():.3f}, {y_geo.min():.3f}) -> ({x_geo.max():.3f}, {y_geo.max():.3f})")
    else:
        x_geo, y_geo = x_orig, y_orig

    # --- Convert bounding box to WGS84 for OSM query ---
    transformer = Transformer.from_crs("EPSG:6635", "EPSG:4326", always_xy=True)
    lons, lats  = transformer.transform(x_geo, y_geo)
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()
    print(f"  Bbox (WGS84): ({min_lat:.6f}, {min_lon:.6f}) -> ({max_lat:.6f}, {max_lon:.6f})")

    bbox_polygon = shapely_box(min_lon, min_lat, max_lon, max_lat)

    # --- Fetch OSM data ---
    buildings = fetch_buildings(bbox_polygon)
    streets   = fetch_streets(bbox_polygon)

    # --- Build point GeoDataFrame using unrotated (real-world) coords ---
    gdf_points = gpd.GeoDataFrame(
        {"row_idx": np.arange(len(df))},
        geometry=gpd.points_from_xy(x_geo, y_geo),
        crs="EPSG:6635"
    )

    # --- Street mask ---
    if streets is not None:
        inside_street = spatial_join_mask(gdf_points, streets, "inside streets")
    else:
        inside_street = np.zeros(len(df), dtype=bool)

    # --- Building mask ---
    if buildings is not None:
        inside_building = spatial_join_mask(gdf_points, buildings, "inside buildings")
        #df.loc[~inside_building & ~inside_street, "object_height"] = 0.0
        df.loc[~inside_building, "object_height"] = 0.0
    else:
        inside_building = np.zeros(len(df), dtype=bool)

    # --- Add flag columns to original (rotated) dataframe ---
    df["is_street"]   = inside_street.astype(np.float32)
    df["is_building"] = inside_building.astype(np.float32)

    # --- Write output ---
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), args.output, compression="snappy")
    print(f"\nSaved -> {args.output}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
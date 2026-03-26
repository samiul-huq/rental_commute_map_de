from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from textwrap import dedent

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"
OSM_INPUT_DIR = ROOT / "input" / "osm"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_PBF = OSM_INPUT_DIR / "germany-latest.osm.pbf"
DEFAULT_OSMCONF = CONFIG_DIR / "osmconf_quality.ini"
DEFAULT_CACHE_DIR = INTERMEDIATE_DIR / "osm_quality_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OSM quality layers and aggregate them to Stadtteile."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PBF,
        help="Path to the source Germany-wide .osm.pbf file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for filtered OSM extracts and cached thematic layers.",
    )
    parser.add_argument(
        "--osmconf",
        type=Path,
        default=DEFAULT_OSMCONF,
        help="Path to the OSM configuration file used by ogr2ogr.",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Rebuild filtered extracts and cached OSM layers even if they already exist.",
    )
    return parser.parse_args()


def ensure_directories(cache_dir: Path) -> None:
    for path in [INTERMEDIATE_DIR, PROCESSED_DIR, cache_dir]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (ROOT / path)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Path not found: {candidate}")
    return candidate


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "green_pbf": cache_dir / "green_areas.osm.pbf",
        "street_trees_pbf": cache_dir / "street_trees.osm.pbf",
        "supermarkets_pbf": cache_dir / "supermarkets.osm.pbf",
        "hospitals_pbf": cache_dir / "hospitals.osm.pbf",
        "green_gpkg": INTERMEDIATE_DIR / "osm_green_areas.gpkg",
        "street_trees_gpkg": INTERMEDIATE_DIR / "osm_street_trees.gpkg",
        "supermarkets_gpkg": INTERMEDIATE_DIR / "osm_supermarkets.gpkg",
        "hospitals_gpkg": INTERMEDIATE_DIR / "osm_hospitals.gpkg",
    }


def osmium_filter(source_pbf: Path, output_pbf: Path, filters: list[str], force: bool) -> None:
    if output_pbf.exists() and not force:
        print(f"Using cached filtered extract: {output_pbf}")
        return
    if output_pbf.exists():
        output_pbf.unlink()
    run_command(
        [
            "osmium",
            "tags-filter",
            str(source_pbf),
            *filters,
            "-o",
            str(output_pbf),
            "-O",
            "--progress",
        ]
    )


def ogr_export(source_pbf: Path, output_gpkg: Path, layer_name: str, sql: str, osmconf_path: Path) -> None:
    env = os.environ.copy()
    env["OSM_CONFIG_FILE"] = str(osmconf_path)
    if output_gpkg.exists():
        output_gpkg.unlink()
    run_command(
        [
            "ogr2ogr",
            "--config",
            "OSM_CONFIG_FILE",
            str(osmconf_path),
            "-f",
            "GPKG",
            str(output_gpkg),
            str(source_pbf),
            "-dialect",
            "SQLITE",
            "-sql",
            sql,
            "-nln",
            layer_name,
            "-overwrite",
            "-progress",
        ],
        env=env,
    )


def export_green_areas(filtered_pbf: Path, output_gpkg: Path, osmconf_path: Path) -> gpd.GeoDataFrame:
    sql = (
        'SELECT osm_id, COALESCE("name:de", name) AS name, leisure, landuse, natural, geometry '
        "FROM multipolygons "
        "WHERE leisure IN ('park','garden','nature_reserve','recreation_ground') "
        "OR landuse IN ('forest','grass','meadow') "
        "OR natural = 'wood'"
    )
    ogr_export(filtered_pbf, output_gpkg, "green_areas", sql, osmconf_path)
    gdf = gpd.read_file(output_gpkg, layer="green_areas")
    gdf = gdf.loc[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    gdf["green_type"] = (
        gdf["leisure"].fillna("").where(gdf["leisure"].fillna("").ne(""), gdf["landuse"]).where(
            lambda s: s.fillna("").ne(""),
            gdf["natural"],
        )
    )
    gdf.to_file(output_gpkg, driver="GPKG", layer="green_areas")
    return gdf


def export_point_like_features(
    filtered_pbf: Path,
    output_gpkg: Path,
    output_layer: str,
    field_name: str,
    field_value: str,
    osmconf_path: Path,
) -> gpd.GeoDataFrame:
    points_sql = (
        f'SELECT osm_id, COALESCE("name:de", name) AS name, {field_name}, geometry '
        f"FROM points WHERE {field_name} = '{field_value}'"
    )
    polygons_sql = (
        f'SELECT osm_id, COALESCE("name:de", name) AS name, {field_name}, geometry '
        f"FROM multipolygons WHERE {field_name} = '{field_value}'"
    )

    tmp_points = output_gpkg.with_name(f"{output_gpkg.stem}_points.gpkg")
    tmp_polygons = output_gpkg.with_name(f"{output_gpkg.stem}_polygons.gpkg")
    ogr_export(filtered_pbf, tmp_points, f"{output_layer}_points", points_sql, osmconf_path)
    ogr_export(filtered_pbf, tmp_polygons, f"{output_layer}_polygons", polygons_sql, osmconf_path)

    point_gdf = gpd.read_file(tmp_points, layer=f"{output_layer}_points")
    polygon_gdf = gpd.read_file(tmp_polygons, layer=f"{output_layer}_polygons")

    if not polygon_gdf.empty:
        polygon_gdf = polygon_gdf.loc[~polygon_gdf.geometry.is_empty & polygon_gdf.geometry.notna()].copy()
        invalid = ~polygon_gdf.geometry.is_valid
        if invalid.any():
            polygon_gdf.loc[invalid, "geometry"] = polygon_gdf.loc[invalid, "geometry"].buffer(0)
        polygon_gdf["geometry"] = polygon_gdf.geometry.representative_point()

    frames: list[gpd.GeoDataFrame] = []
    if not point_gdf.empty:
        point_gdf = point_gdf.loc[~point_gdf.geometry.is_empty & point_gdf.geometry.notna()].copy()
        frames.append(point_gdf)
    if not polygon_gdf.empty:
        frames.append(polygon_gdf)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)
        gdf = gdf.drop_duplicates(subset=["osm_id"], keep="first").copy()
    else:
        gdf = gpd.GeoDataFrame(columns=["osm_id", "name", field_name, "geometry"], geometry="geometry", crs="EPSG:4326")

    for tmp_path in [tmp_points, tmp_polygons]:
        if tmp_path.exists():
            tmp_path.unlink()

    gdf.to_file(output_gpkg, driver="GPKG", layer=output_layer)
    return gdf


def export_street_trees(filtered_pbf: Path, output_gpkg: Path, osmconf_path: Path) -> gpd.GeoDataFrame:
    points_sql = (
        'SELECT osm_id, COALESCE("name:de", name) AS name, man_made, other_tags, geometry '
        "FROM points "
        "WHERE other_tags LIKE '%\"natural\"=>\"tree\"%' "
        "OR other_tags LIKE '%\"natural\"=>\"tree_row\"%' "
        "OR man_made = 'tree_row'"
    )
    lines_sql = (
        'SELECT osm_id, COALESCE("name:de", name) AS name, man_made, other_tags, geometry '
        "FROM lines "
        "WHERE other_tags LIKE '%\"natural\"=>\"tree_row\"%' "
        "OR other_tags LIKE '%\"man_made\"=>\"tree_row\"%' "
        "OR man_made = 'tree_row'"
    )
    multilines_sql = (
        'SELECT osm_id, COALESCE("name:de", name) AS name, other_tags, geometry '
        "FROM multilinestrings "
        "WHERE other_tags LIKE '%\"natural\"=>\"tree_row\"%' "
        "OR other_tags LIKE '%\"man_made\"=>\"tree_row\"%'"
    )

    tmp_points = output_gpkg.with_name(f"{output_gpkg.stem}_points.gpkg")
    tmp_lines = output_gpkg.with_name(f"{output_gpkg.stem}_lines.gpkg")
    tmp_multilines = output_gpkg.with_name(f"{output_gpkg.stem}_multilines.gpkg")
    ogr_export(filtered_pbf, tmp_points, "street_trees_points", points_sql, osmconf_path)
    ogr_export(filtered_pbf, tmp_lines, "street_trees_lines", lines_sql, osmconf_path)
    ogr_export(filtered_pbf, tmp_multilines, "street_trees_multilines", multilines_sql, osmconf_path)

    point_gdf = gpd.read_file(tmp_points, layer="street_trees_points")
    line_gdf = gpd.read_file(tmp_lines, layer="street_trees_lines")
    multiline_gdf = gpd.read_file(tmp_multilines, layer="street_trees_multilines")

    frames: list[gpd.GeoDataFrame] = []
    if not point_gdf.empty:
        point_gdf = point_gdf.loc[~point_gdf.geometry.is_empty & point_gdf.geometry.notna()].copy()
        point_gdf["tree_source"] = np.where(
            point_gdf["other_tags"].fillna("").str.contains('"natural"=>"tree_row"|\"man_made\"=>\"tree_row\"', regex=True),
            "tree_row",
            "tree_point",
        )
        frames.append(point_gdf)
    for line_frame in [line_gdf, multiline_gdf]:
        if not line_frame.empty:
            line_frame = line_frame.loc[~line_frame.geometry.is_empty & line_frame.geometry.notna()].copy()
            line_frame["geometry"] = line_frame.geometry.representative_point()
            line_frame["tree_source"] = "tree_row"
            frames.append(line_frame)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)
        gdf = gdf.drop_duplicates(subset=["osm_id"], keep="first").copy()
        urban_mask = gdf["other_tags"].fillna("").str.contains(
            '"denotation"=>"urban"|\"denotation\"=>\"street_tree\"',
            regex=True,
        )
        row_mask = gdf["tree_source"].eq("tree_row")
        gdf = gdf.loc[urban_mask | row_mask].copy()
    else:
        gdf = gpd.GeoDataFrame(
            columns=["osm_id", "name", "man_made", "other_tags", "tree_source", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    for tmp_path in [tmp_points, tmp_lines, tmp_multilines]:
        if tmp_path.exists():
            tmp_path.unlink()

    gdf.to_file(output_gpkg, driver="GPKG", layer="street_trees")
    return gdf




def load_stdt_base() -> gpd.GeoDataFrame:
    path = PROCESSED_DIR / "rent_transit_by_stdt.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"Could not find required stage-2 output: {path}")
    gdf = gpd.read_file(path)
    if "stdt_id" not in gdf.columns:
        raise ValueError(f"{path.name} does not contain 'stdt_id'.")
    gdf["stdt_id"] = gdf["stdt_id"].astype("string")
    return gdf


def build_or_load_osm_layers(
    args: argparse.Namespace,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    source_pbf = resolve_existing_path(args.input)
    osmconf = resolve_existing_path(args.osmconf)
    paths = cache_paths(args.cache_dir)
    missing_layers = [path for key, path in paths.items() if key.endswith("_gpkg") and not path.exists()]

    if missing_layers or args.force_reparse:
        print("\n=== OSM Quality Extraction ===")
        print("source pbf:", source_pbf)
        osmium_filter(
            source_pbf,
            paths["green_pbf"],
            [
                "wr/leisure=park,garden,nature_reserve,recreation_ground",
                "wr/landuse=forest,grass,meadow",
                "wr/natural=wood",
            ],
            args.force_reparse,
        )
        osmium_filter(
            source_pbf,
            paths["street_trees_pbf"],
            [
                "nwr/natural=tree,tree_row",
                "nwr/man_made=tree_row",
            ],
            args.force_reparse,
        )
        osmium_filter(
            source_pbf,
            paths["supermarkets_pbf"],
            ["nwr/shop=supermarket"],
            args.force_reparse,
        )
        osmium_filter(
            source_pbf,
            paths["hospitals_pbf"],
            ["nwr/amenity=hospital"],
            args.force_reparse,
        )
        green = export_green_areas(paths["green_pbf"], paths["green_gpkg"], osmconf)
        street_trees = export_street_trees(paths["street_trees_pbf"], paths["street_trees_gpkg"], osmconf)
        supermarkets = export_point_like_features(
            paths["supermarkets_pbf"],
            paths["supermarkets_gpkg"],
            "supermarkets",
            "shop",
            "supermarket",
            osmconf,
        )
        hospitals = export_point_like_features(
            paths["hospitals_pbf"],
            paths["hospitals_gpkg"],
            "hospitals",
            "amenity",
            "hospital",
            osmconf,
        )
    else:
        green = gpd.read_file(paths["green_gpkg"], layer="green_areas")
        street_trees = gpd.read_file(paths["street_trees_gpkg"], layer="street_trees")
        supermarkets = gpd.read_file(paths["supermarkets_gpkg"], layer="supermarkets")
        hospitals = gpd.read_file(paths["hospitals_gpkg"], layer="hospitals")
    return green, street_trees, supermarkets, hospitals


def aggregate_green_space(stdt_gdf: gpd.GeoDataFrame, green_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if green_gdf.empty:
        return pd.DataFrame(
            columns=[
                "stdt_id",
                "green_space_count",
                "green_space_area_m2",
                "green_space_share",
                "green_space_patch_density_km2",
            ]
        )

    local_crs = "EPSG:3035"
    stdt_local = stdt_gdf[["stdt_id", "geometry"]].to_crs(local_crs).copy()
    stdt_local["stdt_area_m2"] = stdt_local.geometry.area
    green_local = green_gdf.to_crs(local_crs).copy()
    green_local = green_local.loc[(~green_local.geometry.is_empty) & green_local.geometry.notna()].copy()

    intersection = gpd.overlay(
        stdt_local[["stdt_id", "stdt_area_m2", "geometry"]],
        green_local[["osm_id", "green_type", "geometry"]],
        how="intersection",
    )
    if intersection.empty:
        return pd.DataFrame(
            columns=[
                "stdt_id",
                "green_space_count",
                "green_space_area_m2",
                "green_space_share",
                "green_space_patch_density_km2",
            ]
        )

    intersection["green_area_part_m2"] = intersection.geometry.area
    green_df = (
        intersection.groupby("stdt_id", dropna=False)
        .agg(
            green_space_count=("osm_id", "nunique"),
            green_space_area_m2=("green_area_part_m2", "sum"),
            stdt_area_m2=("stdt_area_m2", "first"),
        )
        .reset_index()
    )
    green_df["green_space_share"] = np.where(
        green_df["stdt_area_m2"] > 0,
        green_df["green_space_area_m2"] / green_df["stdt_area_m2"],
        np.nan,
    )
    green_df["green_space_patch_density_km2"] = np.where(
        green_df["stdt_area_m2"] > 0,
        green_df["green_space_count"] / (green_df["stdt_area_m2"] / 1_000_000.0),
        np.nan,
    )
    return green_df.drop(columns=["stdt_area_m2"])


def aggregate_point_counts(
    stdt_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    metric_name: str,
) -> pd.DataFrame:
    if poi_gdf.empty:
        return pd.DataFrame(columns=["stdt_id", metric_name])

    if poi_gdf.crs != stdt_gdf.crs:
        poi_gdf = poi_gdf.to_crs(stdt_gdf.crs)
    joined = gpd.sjoin(
        poi_gdf[["osm_id", "geometry"]],
        stdt_gdf[["stdt_id", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    if joined.empty:
        return pd.DataFrame(columns=["stdt_id", metric_name])

    return (
        joined.groupby("stdt_id", dropna=False)["osm_id"]
        .nunique()
        .rename(metric_name)
        .reset_index()
    )




def merge_quality_metrics(
    stdt_gdf: gpd.GeoDataFrame,
    green_df: pd.DataFrame,
    street_tree_df: pd.DataFrame,
    supermarket_df: pd.DataFrame,
    hospital_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    merged = stdt_gdf.copy()
    merged = merged.merge(green_df, on="stdt_id", how="left")
    merged = merged.merge(street_tree_df, on="stdt_id", how="left")
    merged = merged.merge(supermarket_df, on="stdt_id", how="left")
    merged = merged.merge(hospital_df, on="stdt_id", how="left")

    local = merged.to_crs("EPSG:3035").copy()
    local["stdt_area_m2"] = local.geometry.area
    merged["stdt_area_km2"] = (local["stdt_area_m2"] / 1_000_000.0).astype(float)

    for col in [
        "green_space_count",
        "green_space_area_m2",
        "street_tree_count",
        "supermarket_count",
        "hospital_count",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["green_space_share"] = pd.to_numeric(merged.get("green_space_share"), errors="coerce")
    merged["green_space_patch_density_km2"] = pd.to_numeric(
        merged.get("green_space_patch_density_km2"),
        errors="coerce",
    )
    merged["street_tree_density_km2"] = np.where(
        merged["stdt_area_km2"] > 0,
        merged["street_tree_count"] / merged["stdt_area_km2"],
        np.nan,
    )
    keep_cols = [
        col
        for col in [
            "stdt_id",
            "stdt_name",
            "municipality_name",
            "population",
            "geometry",
            "stdt_area_km2",
            "green_space_share",
            "green_space_patch_density_km2",
            "street_tree_density_km2",
            "supermarket_count",
            "hospital_count",
        ]
        if col in merged.columns
    ]
    return merged[keep_cols].copy()


def main() -> int:
    args = parse_args()
    ensure_directories(args.cache_dir)

    stdt_gdf = load_stdt_base()
    green_gdf, street_trees_gdf, supermarkets_gdf, hospitals_gdf = build_or_load_osm_layers(args)

    print("\n=== OSM Quality Inputs ===")
    print("stadtteile:", len(stdt_gdf))
    print("green areas:", len(green_gdf))
    print("street trees:", len(street_trees_gdf))
    print("supermarkets:", len(supermarkets_gdf))
    print("hospitals:", len(hospitals_gdf))

    green_df = aggregate_green_space(stdt_gdf, green_gdf)
    street_tree_df = aggregate_point_counts(stdt_gdf, street_trees_gdf, "street_tree_count")
    supermarket_df = aggregate_point_counts(stdt_gdf, supermarkets_gdf, "supermarket_count")
    hospital_df = aggregate_point_counts(stdt_gdf, hospitals_gdf, "hospital_count")

    quality_gdf = merge_quality_metrics(stdt_gdf, green_df, street_tree_df, supermarket_df, hospital_df)
    quality_df = pd.DataFrame(quality_gdf.drop(columns="geometry"))

    quality_df.to_csv(PROCESSED_DIR / "rent_transit_quality_by_stdt.csv", index=False, encoding="utf-8")
    quality_gdf.to_file(PROCESSED_DIR / "rent_transit_quality_by_stdt.gpkg", driver="GPKG")

    print("\n=== OSM Quality Summary ===")
    print("stadtteile with green space:", int(pd.to_numeric(quality_df["green_space_share"], errors="coerce").fillna(0).gt(0).sum()))
    print("stadtteile with street trees:", int(pd.to_numeric(quality_df["street_tree_density_km2"], errors="coerce").fillna(0).gt(0).sum()))
    print("stadtteile with supermarkets:", int(pd.to_numeric(quality_df["supermarket_count"], errors="coerce").fillna(0).gt(0).sum()))
    print("stadtteile with hospitals:", int(pd.to_numeric(quality_df["hospital_count"], errors="coerce").fillna(0).gt(0).sum()))
    print("outputs written:")
    print(f" - {(PROCESSED_DIR / 'rent_transit_quality_by_stdt.csv').relative_to(ROOT)}")
    print(f" - {(PROCESSED_DIR / 'rent_transit_quality_by_stdt.gpkg').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

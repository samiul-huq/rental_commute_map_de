from __future__ import annotations

import argparse
import json
import os
import subprocess
import warnings
from pathlib import Path
from textwrap import dedent

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "input"
CONFIG_DIR = ROOT / "config"
PROCESSED_DIR = ROOT / "data" / "processed"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
FIGURES_DIR = ROOT / "outputs" / "figures"
MAPS_DIR = ROOT / "outputs" / "maps"

STADTTEILE_INPUT_DIR = INTERMEDIATE_DIR / "stadtteile_shapes"
OSM_INPUT_DIR = INPUT_DIR / "osm"
DEFAULT_STADTTEILE_PARQUET = STADTTEILE_INPUT_DIR / "germany_stadtteile.parquet"
DEFAULT_STADTTEILE_CACHE_DIR = INTERMEDIATE_DIR / "stadtteile_cache"
DEFAULT_OSM_PBF = OSM_INPUT_DIR / "germany-latest.osm.pbf"
DEFAULT_OSMCONF_STADTTEILE = CONFIG_DIR / "osmconf_stadtteile.ini"
DEFAULT_NUTS_GPKG = INPUT_DIR / "nuts3_shapes" / "NUTS_RG_01M_2024_3035.gpkg"

MUNICIPALITY_LEVELS = {"4", "6", "8"}
STADTTEIL_LEVELS = {"9", "10"}

SCHEMA_REFERENCE = [
    ("osm_id", "string", "OSM", "Format: r12345678; stable relation ID"),
    ("name", "string", "OSM tag", "German name of the Stadtteil"),
    ("name_de", "string", "OSM tag", "Explicit German name if mapped separately"),
    ("admin_level", "string", "OSM tag", '"9" (Stadtbezirk) or "10" (Stadtteil)'),
    ("ref", "string", "OSM tag", "Local reference code if mapped"),
    ("ags", "string", "OSM tag", "Amtlicher Gemeindeschluessel; sparse"),
    ("population", "float", "OSM tag", "Sparse; official portals are usually better"),
    ("wikipedia", "string", "OSM tag", "Wikipedia article link if mapped"),
    ("wikidata", "string", "OSM tag", "Wikidata QID if mapped"),
    ("municipality_name", "string", "Spatial join", "Best parent name from admin_level 4/6/8"),
    ("municipality_id", "string", "Spatial join", "Parent city OSM relation ID"),
    ("admin_level_muni", "string", "Spatial join", 'Chosen parent admin level: "4", "6", or "8"'),
    ("geometry", "MultiPolygon", "OSM", "WGS84 (EPSG:4326)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stadtteil rent outputs, with optional Stadtteil boundary rebuild from the Germany-wide OSM PBF."
    )
    parser.add_argument(
        "--rebuild-stadtteile",
        action="store_true",
        help="Rebuild data/intermediate/stadtteile_shapes/germany_stadtteile.parquet from OSM before rent aggregation.",
    )
    parser.add_argument(
        "--force-reparse-stadtteile",
        action="store_true",
        help="Ignore cached extracted Stadtteil boundary layers and rebuild them from OSM.",
    )
    parser.add_argument(
        "--validate-stadtteile-only",
        action="store_true",
        help="Validate the Stadtteil parquet and exit without building rent outputs.",
    )
    parser.add_argument(
        "--no-validation-map",
        action="store_true",
        help="Skip the validation PNG when using --validate-stadtteile-only.",
    )
    parser.add_argument(
        "--osm-pbf",
        type=Path,
        default=DEFAULT_OSM_PBF,
        help="Germany-wide OSM PBF used to rebuild Stadtteil boundaries when needed.",
    )
    parser.add_argument(
        "--osmconf-stadtteile",
        type=Path,
        default=DEFAULT_OSMCONF_STADTTEILE,
        help="OSM configuration file used to export Stadtteil boundaries from the PBF.",
    )
    parser.add_argument(
        "--stadtteile-cache-dir",
        type=Path,
        default=DEFAULT_STADTTEILE_CACHE_DIR,
        help="Cache directory for intermediate Stadtteil boundary extraction artifacts.",
    )
    parser.add_argument(
        "--nuts-path",
        type=Path,
        default=DEFAULT_NUTS_GPKG,
        help="Germany NUTS GPKG used to clip administrative outputs to Germany.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for path in [PROCESSED_DIR, INTERMEDIATE_DIR, FIGURES_DIR, MAPS_DIR, STADTTEILE_INPUT_DIR, OSM_INPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (ROOT / path)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Path not found: {candidate}")
    return candidate


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def stadtteile_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "stadtteile": cache_dir / "stadtteile_raw.parquet",
        "municipalities": cache_dir / "municipalities_raw.parquet",
        "admin_gpkg": cache_dir / "admin_boundaries_raw.gpkg",
        "admin_boundaries_pbf": cache_dir / "admin_boundaries.osm.pbf",
        "admin_levels_pbf": cache_dir / "admin_levels_4_6_8_9_10.osm.pbf",
    }


def load_germany_boundary(nuts_path: Path) -> gpd.GeoDataFrame:
    nuts = gpd.read_file(nuts_path)
    nuts = nuts.loc[(nuts["CNTR_CODE"] == "DE") & (nuts["LEVL_CODE"] == 0)].copy()
    if nuts.empty:
        raise ValueError(f"Could not find Germany boundary in {nuts_path}")
    return nuts.to_crs("EPSG:4326")


def extract_stadtteil_boundaries(
    source_pbf: Path,
    cache_dir: Path,
    osmconf_path: Path,
    force_reparse: bool,
) -> tuple[Path, Path]:
    paths = stadtteile_cache_paths(cache_dir)
    ensure_dir(cache_dir)

    if not force_reparse and paths["stadtteile"].exists() and paths["municipalities"].exists():
        print(f"Using cached Stadtteil boundaries from {cache_dir}")
        return paths["stadtteile"], paths["municipalities"]

    sql = (
        "SELECT osm_id, name, name_de, admin_level, ref, "
        "de_amtlicher_gemeindeschluessel AS ags, place, population, "
        "wikipedia, wikidata, geometry "
        "FROM multipolygons "
        "WHERE osm_way_id IS NULL "
        "AND boundary = 'administrative' "
        "AND admin_level IN ('4','6','8','9','10')"
    )

    print("Filtering administrative relations with osmium tags-filter")
    run_command(
        [
            "osmium",
            "tags-filter",
            str(source_pbf),
            "r/boundary=administrative",
            "-o",
            str(paths["admin_boundaries_pbf"]),
            "-O",
            "--progress",
        ]
    )

    print("Filtering admin levels 4, 6, 8, 9, and 10 with osmium tags-filter")
    run_command(
        [
            "osmium",
            "tags-filter",
            str(paths["admin_boundaries_pbf"]),
            "r/admin_level=4,6,8,9,10",
            "-o",
            str(paths["admin_levels_pbf"]),
            "-O",
            "--progress",
        ]
    )

    env = os.environ.copy()
    env["OSM_CONFIG_FILE"] = str(osmconf_path)
    run_command(
        [
            "ogr2ogr",
            "--config",
            "OSM_CONFIG_FILE",
            str(osmconf_path),
            "-f",
            "GPKG",
            str(paths["admin_gpkg"]),
            str(paths["admin_levels_pbf"]),
            "-dialect",
            "SQLITE",
            "-sql",
            sql,
            "-nln",
            "admin_boundaries",
            "-overwrite",
            "-progress",
        ],
        env=env,
    )

    admin_gdf = gpd.read_file(paths["admin_gpkg"], layer="admin_boundaries")
    admin_gdf["osm_id"] = admin_gdf["osm_id"].astype(str).radd("r")

    stadtteile_gdf = admin_gdf.loc[admin_gdf["admin_level"].isin(STADTTEIL_LEVELS)].copy()
    municipalities_gdf = admin_gdf.loc[admin_gdf["admin_level"].isin(MUNICIPALITY_LEVELS)].copy()
    municipalities_gdf = municipalities_gdf.rename(
        columns={
            "name": "municipality_name",
            "osm_id": "municipality_id",
            "admin_level": "admin_level_muni",
        }
    )[["municipality_name", "municipality_id", "admin_level_muni", "geometry"]]
    municipalities_gdf["municipality_area"] = municipalities_gdf.geometry.area

    stadtteile_gdf.to_parquet(paths["stadtteile"])
    municipalities_gdf.to_parquet(paths["municipalities"])

    print(f"Cached {len(stadtteile_gdf)} stadtteile and {len(municipalities_gdf)} municipalities")
    return paths["stadtteile"], paths["municipalities"]


def build_stadtteil_outputs_from_cache(cache_dir: Path, nuts_path: Path) -> gpd.GeoDataFrame:
    paths = stadtteile_cache_paths(cache_dir)
    stadtteile_gdf = gpd.read_parquet(paths["stadtteile"])
    municipalities_gdf = gpd.read_parquet(paths["municipalities"])

    stadtteile_gdf = stadtteile_gdf.loc[stadtteile_gdf.geometry.notna() & ~stadtteile_gdf.geometry.is_empty].copy()
    municipalities_gdf = municipalities_gdf.loc[
        municipalities_gdf.geometry.notna() & ~municipalities_gdf.geometry.is_empty
    ].copy()

    stadtteile_gdf["geometry"] = stadtteile_gdf.geometry.buffer(0)
    municipalities_gdf["geometry"] = municipalities_gdf.geometry.buffer(0)
    stadtteile_gdf["population"] = pd.to_numeric(stadtteile_gdf["population"], errors="coerce")

    germany_boundary = load_germany_boundary(nuts_path)
    stadtteile_gdf["_point"] = stadtteile_gdf.geometry.representative_point()
    stadtteile_gdf = gpd.sjoin(
        stadtteile_gdf,
        germany_boundary[["geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    stadtteile_points = stadtteile_gdf.set_geometry("_point")
    joined = gpd.sjoin(
        stadtteile_points[["osm_id", "_point"]],
        municipalities_gdf[
            ["municipality_name", "municipality_id", "admin_level_muni", "municipality_area", "geometry"]
        ],
        how="left",
        predicate="within",
    )
    joined["admin_level_rank"] = pd.to_numeric(joined["admin_level_muni"], errors="coerce")
    joined = joined.sort_values(
        by=["osm_id", "admin_level_rank", "municipality_area"],
        ascending=[True, False, True],
    )
    joined = joined.drop_duplicates(subset="osm_id", keep="first")

    return stadtteile_gdf.drop(columns=["_point"]).merge(
        joined[["osm_id", "municipality_name", "municipality_id", "admin_level_muni"]],
        on="osm_id",
        how="left",
    )


def export_stadtteil_input_files(gdf: gpd.GeoDataFrame) -> None:
    gdf.to_parquet(DEFAULT_STADTTEILE_PARQUET)


def ensure_stadtteil_boundaries(args: argparse.Namespace) -> Path:
    parquet_path = DEFAULT_STADTTEILE_PARQUET
    if parquet_path.exists() and not args.rebuild_stadtteile:
        return parquet_path

    source_pbf = resolve_existing_path(args.osm_pbf)
    osmconf_path = resolve_existing_path(args.osmconf_stadtteile)
    nuts_path = resolve_existing_path(args.nuts_path)
    cache_dir = args.stadtteile_cache_dir.resolve()

    print("\n=== Stadtteil Boundary Build ===")
    print("source pbf:", source_pbf)
    print("osm config:", osmconf_path)
    print("cache dir:", cache_dir)

    extract_stadtteil_boundaries(
        source_pbf=source_pbf,
        cache_dir=cache_dir,
        osmconf_path=osmconf_path,
        force_reparse=args.force_reparse_stadtteile,
    )
    stadtteile_gdf = build_stadtteil_outputs_from_cache(cache_dir, nuts_path)
    export_stadtteil_input_files(stadtteile_gdf)
    print("stadtteil input rebuilt:", parquet_path)
    return parquet_path


def find_stadtteile_path(args: argparse.Namespace) -> Path:
    return ensure_stadtteil_boundaries(args)


def read_stadtteile(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    gdf = gdf.loc[gdf.geometry.notna()].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["admin_level_num"] = pd.to_numeric(gdf.get("admin_level"), errors="coerce")
    gdf["stdt_id"] = gdf["osm_id"].astype("string")
    gdf["stdt_name"] = (
        gdf.get("name_de", pd.Series(pd.NA, index=gdf.index))
        .fillna(gdf.get("name", pd.Series(pd.NA, index=gdf.index)))
        .astype("string")
    )
    gdf["municipality_name"] = gdf.get("municipality_name", pd.Series(pd.NA, index=gdf.index)).astype("string")
    if gdf["admin_level_num"].notna().any():
        deepest_by_muni = gdf.groupby("municipality_name")["admin_level_num"].transform("max")
        gdf = gdf.loc[gdf["admin_level_num"].fillna(-1).eq(deepest_by_muni.fillna(-1))].copy()
    area_crs = "EPSG:3035"
    local = gdf.to_crs(area_crs).copy()
    local["area_m2"] = local.geometry.area
    local = (
        local.sort_values(["municipality_name", "stdt_name", "area_m2"], ascending=[True, True, True])
        .drop_duplicates(subset=["municipality_name", "stdt_name"], keep="first")
        .copy()
    )
    return local.to_crs(gdf.crs)


def validate_stadtteile(path: Path, render_map: bool) -> None:
    gdf = gpd.read_parquet(path)
    print(f"Loaded dataset: {path}")
    print(f"Total features: {len(gdf)}")
    print("\nBy admin_level:")
    print(gdf["admin_level"].value_counts(dropna=False))
    print("\nTop 10 cities by Stadtteil count:")
    print(gdf.groupby("municipality_name", dropna=False).size().sort_values(ascending=False).head(10))
    print("\nCoverage check — features with empty name:")
    print((gdf["name"].fillna("") == "").sum())
    print("\nFeatures with AGS code:")
    print((gdf["ags"].fillna("") != "").sum())
    print("\nCurrent dtypes:")
    print(gdf.dtypes)
    print("\nSchema reference:")
    for column, dtype, source, notes in SCHEMA_REFERENCE:
        print(f"- {column}: {dtype} | {source} | {notes}")
    if render_map:
        out_path = FIGURES_DIR / "germany_stadtteile_validation.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 14))
        gdf.plot(ax=ax, edgecolor="white", linewidth=0.2, column="admin_level", legend=True)
        ax.set_title("Germany Stadtteile (admin_level 9 & 10)")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved validation map to {out_path}")


def read_rent_grid() -> gpd.GeoDataFrame:
    path = INPUT_DIR / "rental_price" / "Zensus 2022 - Kaltmieten.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    gdf = gpd.read_file(path)
    gdf = gdf.rename(
        columns={
            "gitter_id_100m": "grid_id_100m",
            "durchschnmieteqm": "mean_rent_per_m2",
        }
    ).copy()
    gdf["mean_rent_per_m2"] = pd.to_numeric(gdf["mean_rent_per_m2"], errors="coerce")
    gdf = gdf.loc[gdf.geometry.notna() & gdf["mean_rent_per_m2"].notna()].copy()
    return gdf[["grid_id_100m", "mean_rent_per_m2", "x_mp_100m", "y_mp_100m", "geometry"]]


def aggregate_grid_to_stadtteile(grid_gdf: gpd.GeoDataFrame, stdt_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    if grid_gdf.crs != stdt_gdf.crs:
        grid_gdf = grid_gdf.to_crs(stdt_gdf.crs)

    joined = gpd.sjoin(
        grid_gdf,
        stdt_gdf[["stdt_id", "stdt_name", "municipality_name", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])

    agg_df = (
        joined.groupby(["stdt_id", "stdt_name", "municipality_name"], dropna=False)
        .agg(
            grid_cell_count=("grid_id_100m", "nunique"),
            mean_rent_per_m2=("mean_rent_per_m2", "mean"),
            median_rent_per_m2=("mean_rent_per_m2", "median"),
            std_rent_per_m2=("mean_rent_per_m2", "std"),
            q25_rent_per_m2=("mean_rent_per_m2", lambda s: s.quantile(0.25)),
            q75_rent_per_m2=("mean_rent_per_m2", lambda s: s.quantile(0.75)),
        )
        .reset_index()
        .sort_values(["median_rent_per_m2", "grid_cell_count"], ascending=[False, False])
    )

    merged_gdf = stdt_gdf.merge(agg_df, on=["stdt_id", "stdt_name", "municipality_name"], how="left")
    return agg_df, merged_gdf


def save_static_map(gdf: gpd.GeoDataFrame, column: str, title: str, output_path: Path, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 14))
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        linewidth=0.05,
        edgecolor="#555555",
        legend=True,
        missing_kwds={"color": "#e3e3e3", "edgecolor": "#9a9a9a", "hatch": "///", "label": "No rent data"},
    )
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_folium_map(gdf: gpd.GeoDataFrame) -> None:
    web_gdf = gdf.to_crs("EPSG:4326").copy()
    minx, miny, maxx, maxy = web_gdf.total_bounds
    center = [float((miny + maxy) / 2), float((minx + maxx) / 2)]
    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")
    folium.Choropleth(
        geo_data=json.loads(web_gdf.to_json()),
        data=web_gdf,
        columns=["stdt_id", "median_rent_per_m2"],
        key_on="feature.properties.stdt_id",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.15,
        nan_fill_color="#d9d9d9",
        legend_name="Median rent per m² (EUR)",
    ).add_to(fmap)
    folium.GeoJson(
        data=json.loads(web_gdf.to_json()),
        style_function=lambda _: {"fillOpacity": 0, "color": "#666666", "weight": 0.2},
        tooltip=folium.GeoJsonTooltip(
            fields=["stdt_name", "municipality_name", "grid_cell_count", "mean_rent_per_m2", "median_rent_per_m2"],
            aliases=["Stadtteil", "Municipality", "100m grid cells", "Mean rent/m²", "Median rent/m²"],
            localize=True,
            sticky=False,
        ),
    ).add_to(fmap)
    fmap.save(MAPS_DIR / "germany_stdt_median_rent_stdt.html")


def write_readme(
    grid_path: Path,
    stadtteile_path: Path,
    grid_gdf: gpd.GeoDataFrame,
    matched: int,
    missing: int,
    rebuilt_from_osm: bool,
) -> None:
    text = dedent(
        f"""
        # Rent Aggregation by Stadtteil

        ## Inputs
        - Rent grid: `{grid_path.relative_to(ROOT)}`
        - Stadtteil boundaries: `{stadtteile_path.relative_to(ROOT)}`

        ## Method
        - Load the Zensus 2022 100m rent grid.
        - Use `durchschnmieteqm` as the source rent-per-m² field.
        - Stadtteil boundaries are read from `data/intermediate/stadtteile_shapes/germany_stadtteile.parquet`.
        - If that file is missing, stage 1 can rebuild it from the Germany-wide OSM PBF.
        - Spatially assign 100m grid points to Stadtteil polygons.
        - Aggregate grid-cell rent values by Stadtteil.

        ## Coverage
        - Input grid points: `{len(grid_gdf):,}`
        - Stadtteile with rent data: `{matched:,}`
        - Stadtteile without rent data: `{missing:,}`
        - Stadtteil boundaries rebuilt from OSM in this run: `{"yes" if rebuilt_from_osm else "no"}`

        ## Main output fields
        - `grid_cell_count`
        - `mean_rent_per_m2`
        - `median_rent_per_m2`
        - `std_rent_per_m2`
        - `q25_rent_per_m2`
        - `q75_rent_per_m2`

        ## Output files
        - `data/processed/rent_by_stdt.csv`
                - `data/processed/rent_by_stdt.geojson`
        - `data/processed/rent_by_stdt.gpkg`
        - `outputs/figures/germany_stdt_median_rent_stdt.png`
        - `outputs/figures/germany_stdt_grid_cell_count_stdt.png`
        - `outputs/maps/germany_stdt_median_rent_stdt.html`
        """
    ).strip()
    (ROOT / "README_rent_by_stdt.md").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_directories()
    grid_path = INPUT_DIR / "rental_price" / "Zensus 2022 - Kaltmieten.geojson"
    had_existing_stadtteile = DEFAULT_STADTTEILE_PARQUET.exists()
    stadtteile_path = find_stadtteile_path(args)

    if args.validate_stadtteile_only:
        validate_stadtteile(stadtteile_path, render_map=not args.no_validation_map)
        return 0

    grid_gdf = read_rent_grid()
    stdt_gdf = read_stadtteile(stadtteile_path)

    print("\n=== Rent Grid Inspection ===")
    print("shape:", grid_gdf.shape)
    print("columns:", grid_gdf.columns.tolist())
    print("crs:", grid_gdf.crs)

    print("\n=== Stadtteil Boundary Inspection ===")
    print("shape:", stdt_gdf.shape)
    print("columns:", stdt_gdf.columns.tolist())
    print("crs:", stdt_gdf.crs)

    agg_df, merged_gdf = aggregate_grid_to_stadtteile(grid_gdf, stdt_gdf)
    agg_df.to_csv(PROCESSED_DIR / "rent_by_stdt.csv", index=False, encoding="utf-8")
    merged_gdf.to_file(PROCESSED_DIR / "rent_by_stdt.geojson", driver="GeoJSON")
    merged_gdf.to_file(PROCESSED_DIR / "rent_by_stdt.gpkg", driver="GPKG")

    save_static_map(
        merged_gdf,
        "median_rent_per_m2",
        "Germany Stadtteil Median Rent per m² (Zensus 2022)",
        FIGURES_DIR / "germany_stdt_median_rent_stdt.png",
        "YlOrRd",
    )
    save_static_map(
        merged_gdf,
        "grid_cell_count",
        "Germany Stadtteil 100m Grid Cell Count",
        FIGURES_DIR / "germany_stdt_grid_cell_count_stdt.png",
        "Blues",
    )
    build_folium_map(merged_gdf)

    matched = int(merged_gdf["median_rent_per_m2"].notna().sum())
    missing = int(merged_gdf["median_rent_per_m2"].isna().sum())
    write_readme(
        grid_path=grid_path,
        stadtteile_path=stadtteile_path,
        grid_gdf=grid_gdf,
        matched=matched,
        missing=missing,
        rebuilt_from_osm=(not had_existing_stadtteile) or args.rebuild_stadtteile,
    )

    print("\n=== Final Summary ===")
    print("grid_cells_used:", len(grid_gdf))
    print("stadtteile_with_rent:", matched)
    print("stadtteile_without_rent:", missing)
    print("outputs_saved_under:")
    print(f"  - {PROCESSED_DIR.relative_to(ROOT)}")
    print(f"  - {FIGURES_DIR.relative_to(ROOT)}")
    print(f"  - {MAPS_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

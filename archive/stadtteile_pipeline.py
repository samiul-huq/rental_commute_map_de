from __future__ import annotations

import argparse
import subprocess
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PBF = BASE_DIR / "stadtteile-pipeline" / "germany-latest.osm.pbf"
DEFAULT_NUTS_GPKG = BASE_DIR / "stadtteile-pipeline" / "NUTS_RG_01M_2024_3035.gpkg"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_CACHE_DIR = BASE_DIR / "cache"
DEFAULT_OSMCONF = BASE_DIR / "osmconf_stadtteile.ini"
MUNICIPALITY_LEVELS = {"4", "6", "8"}
STADTTEIL_LEVELS = {"9", "10"}


def add_common_paths(
    parser: argparse.ArgumentParser,
    include_output: bool,
    include_input: bool = True,
) -> None:
    if include_input:
        parser.add_argument(
            "--input",
            type=Path,
            default=DEFAULT_PBF,
            help="Path to the source .osm.pbf file.",
        )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for intermediate cached layers.",
    )
    if include_output:
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help="Directory for final exports.",
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (BASE_DIR / path)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Path not found: {candidate}")
    return candidate


def resolve_output_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (BASE_DIR / path)
    return candidate.resolve()

def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def load_germany_boundary() -> gpd.GeoDataFrame:
    nuts = gpd.read_file(
        DEFAULT_NUTS_GPKG,
        layer="NUTS_RG_01M_2024_3035.gpkg",
        where="CNTR_CODE = 'DE' AND LEVL_CODE = 0",
    )
    if nuts.empty:
        raise ValueError("Could not find Germany boundary in NUTS file")
    return nuts.to_crs("EPSG:4326")


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "stadtteile": cache_dir / "stadtteile_raw.parquet",
        "municipalities": cache_dir / "municipalities_raw.parquet",
        "admin_gpkg": cache_dir / "admin_boundaries_raw.gpkg",
        "admin_boundaries_pbf": cache_dir / "admin_boundaries.osm.pbf",
        "admin_levels_pbf": cache_dir / "admin_levels_4_6_8_9_10.osm.pbf",
    }


def extract_boundaries(source_pbf: Path, cache_dir: Path, force_reparse: bool) -> tuple[Path, Path]:
    paths = cache_paths(cache_dir)
    ensure_dir(cache_dir)

    if (
        not force_reparse
        and paths["stadtteile"].exists()
        and paths["municipalities"].exists()
    ):
        print(f"Using cached boundaries from {cache_dir}")
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

    print("Filtering admin levels 6, 8, 9, and 10 with osmium tags-filter")
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

    run_command(
        [
            "ogr2ogr",
            "--config",
            "OSM_CONFIG_FILE",
            str(DEFAULT_OSMCONF),
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
        ]
    )

    admin_gdf = gpd.read_file(paths["admin_gpkg"], layer="admin_boundaries")
    admin_gdf["osm_id"] = admin_gdf["osm_id"].astype(str).radd("r")

    stadtteile_gdf = admin_gdf[admin_gdf["admin_level"].isin(STADTTEIL_LEVELS)].copy()
    municipalities_gdf = admin_gdf[admin_gdf["admin_level"].isin(MUNICIPALITY_LEVELS)].copy()
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


def load_cache(cache_path: Path) -> gpd.GeoDataFrame:
    return gpd.read_parquet(cache_path)


def build_outputs(
    stadtteile_gdf: gpd.GeoDataFrame,
    municipalities_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    stadtteile_gdf = stadtteile_gdf[
        stadtteile_gdf.geometry.notna() & ~stadtteile_gdf.geometry.is_empty
    ].copy()
    municipalities_gdf = municipalities_gdf[
        municipalities_gdf.geometry.notna() & ~municipalities_gdf.geometry.is_empty
    ].copy()

    stadtteile_gdf["geometry"] = stadtteile_gdf.geometry.buffer(0)
    municipalities_gdf["geometry"] = municipalities_gdf.geometry.buffer(0)
    stadtteile_gdf["population"] = pd.to_numeric(
        stadtteile_gdf["population"], errors="coerce"
    )
    germany_boundary = load_germany_boundary()
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


def export_outputs(
    gdf: gpd.GeoDataFrame,
    output_dir: Path,
    skip_geojson: bool,
) -> dict[str, Path]:
    ensure_dir(output_dir)

    outputs = {
        "gpkg": output_dir / "germany_stadtteile.gpkg",
        "parquet": output_dir / "germany_stadtteile.parquet",
        "csv": output_dir / "germany_stadtteile_nogeom.csv",
    }

    print(f"Writing {outputs['gpkg']}")
    gdf.to_file(outputs["gpkg"], driver="GPKG", layer="stadtteile")

    if not skip_geojson:
        outputs["geojson"] = output_dir / "germany_stadtteile.geojson"
        print(f"Writing {outputs['geojson']}")
        gdf.to_file(outputs["geojson"], driver="GeoJSON")

    print(f"Writing {outputs['parquet']}")
    gdf.to_parquet(outputs["parquet"])

    print(f"Writing {outputs['csv']}")
    gdf.drop(columns=["geometry"]).to_csv(outputs["csv"], index=False)

    return outputs


def assemble_outputs(cache_dir: Path, output_dir: Path, skip_geojson: bool) -> dict[str, Path]:
    paths = cache_paths(cache_dir)
    stadtteile_gdf = load_cache(paths["stadtteile"])
    municipalities_gdf = load_cache(paths["municipalities"])

    print("Joining stadtteile to municipalities")
    final_gdf = build_outputs(stadtteile_gdf, municipalities_gdf)

    print(f"Final dataset: {len(final_gdf)} features")
    print(final_gdf[["osm_id", "name", "admin_level", "municipality_name"]].head(10))

    return export_outputs(final_gdf, output_dir=output_dir, skip_geojson=skip_geojson)

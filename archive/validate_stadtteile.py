from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "output" / "germany_stadtteile.parquet"
DEFAULT_MAP = BASE_DIR / "output" / "germany_stadtteile_map.png"

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
        description="Validate and inspect germany_stadtteile parquet output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to germany_stadtteile.parquet",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=DEFAULT_MAP,
        help="Path to save the validation map PNG",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip map rendering",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (BASE_DIR / path)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found: {candidate}")
    return candidate


def print_schema_reference() -> None:
    print("\nSchema reference:")
    for column, dtype, source, notes in SCHEMA_REFERENCE:
        print(f"- {column}: {dtype} | {source} | {notes}")


def render_map(gdf: gpd.GeoDataFrame, map_output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for map output. Install it with "
            r".\stadtteile-env\Scripts\python.exe -m pip install matplotlib"
        ) from exc

    map_output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    gdf.plot(ax=ax, edgecolor="white", linewidth=0.2, column="admin_level", legend=True)
    ax.set_title("Germany Stadtteile (admin_level 9 & 10)")
    plt.savefig(map_output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved map to {map_output}")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.input)
    map_output = args.map_output if args.map_output.is_absolute() else (BASE_DIR / args.map_output)
    map_output = map_output.resolve()

    gdf = gpd.read_parquet(dataset_path)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Total features: {len(gdf)}")

    print("\nBy admin_level:")
    print(gdf["admin_level"].value_counts(dropna=False))

    print("\nTop 10 cities by Stadtteil count:")
    print(
        gdf.groupby("municipality_name", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    empty_names = (gdf["name"].fillna("") == "").sum()
    ags_present = (gdf["ags"].fillna("") != "").sum()

    print("\nCoverage check — features with empty name:")
    print(empty_names)

    print("\nFeatures with AGS code:")
    print(ags_present)

    print("\nCurrent dtypes:")
    print(gdf.dtypes)

    print_schema_reference()

    if not args.no_plot:
        render_map(gdf, map_output)


if __name__ == "__main__":
    main()

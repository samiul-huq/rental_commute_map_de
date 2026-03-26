from __future__ import annotations

import argparse

from stadtteile_pipeline import (
    add_common_paths,
    assemble_outputs,
    cache_paths,
    resolve_existing_path,
    resolve_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble final Stadtteile outputs from cached boundary layers."
    )
    add_common_paths(parser, include_output=True, include_input=False)
    parser.add_argument(
        "--skip-geojson",
        action="store_true",
        help="Skip GeoJSON export to save time and disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = resolve_output_path(args.cache_dir)
    output_dir = resolve_output_path(args.output_dir)

    paths = cache_paths(cache_dir)
    resolve_existing_path(paths["stadtteile"])
    resolve_existing_path(paths["municipalities"])

    assemble_outputs(cache_dir=cache_dir, output_dir=output_dir, skip_geojson=args.skip_geojson)
    print("Output assembly complete")


if __name__ == "__main__":
    main()

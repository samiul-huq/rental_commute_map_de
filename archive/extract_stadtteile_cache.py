from __future__ import annotations

import argparse

from stadtteile_pipeline import (
    add_common_paths,
    extract_boundaries,
    resolve_existing_path,
    resolve_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and cache municipality and stadtteil boundary layers."
    )
    add_common_paths(parser, include_output=False)
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Ignore cached extracted boundaries and parse the PBF again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_pbf = resolve_existing_path(args.input)
    cache_dir = resolve_output_path(args.cache_dir)

    extract_boundaries(
        source_pbf=source_pbf,
        cache_dir=cache_dir,
        force_reparse=args.force_reparse,
    )
    print("Cache extraction complete")


if __name__ == "__main__":
    main()

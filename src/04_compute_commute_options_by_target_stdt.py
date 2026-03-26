from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_commute_module():
    module_path = Path(__file__).with_name("05_rank_commute_locations_by_workplace_stdt.py")
    spec = importlib.util.spec_from_file_location("commute_rank_stdt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load commute ranking module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


commute = load_commute_module()


def main() -> int:
    commute.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    commute.COMMUTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    args = commute.parse_args_with_config()
    lat, lon = commute.parse_coordinate_from_config(args)
    args.transfer_radius_m = commute.normalize_transfer_radius_m(args.transfer_radius_m, args.min_transfer_radius_m, args.max_transfer_radius_m)
    args.max_transfers = max(0, int(args.max_transfers))

    district, _ = commute.detect_workplace_district(lat, lon, args.nuts_path)
    district_stdt_gdf = commute.load_district_stdt_geodata(district)
    district_stdt_set = set(district_stdt_gdf["stdt_id"].dropna().tolist())
    if not district_stdt_set:
        raise ValueError("No Stadtteil polygons intersect the detected district.")

    assigned_stops = commute.load_assigned_stops(district_stdt_set)
    all_gtfs_stops = commute.load_all_gtfs_stops()
    district_gtfs_stops = commute.load_all_gtfs_stops_in_district(district, all_gtfs_stops)
    stop_neighbor_lookup = commute.build_stop_neighbor_lookup(district_gtfs_stops, args.transfer_radius_m)
    if assigned_stops.empty:
        raise ValueError("No Stadtteil-assigned GTFS stops were found inside the detected district.")

    calendar_df = pd.read_csv(commute.GTFS_DIR / "calendar.txt", dtype={"service_id": "string"})
    calendar_dates_df = pd.read_csv(commute.GTFS_DIR / "calendar_dates.txt", dtype={"service_id": "string"})
    service_date = pd.Timestamp(args.date) if args.date else commute.choose_default_service_date(calendar_df)

    arrival_start_sec = commute.parse_gtfs_time_to_seconds(args.arrival_start)
    arrival_end_sec = commute.parse_gtfs_time_to_seconds(args.arrival_end)
    if arrival_start_sec is None or arrival_end_sec is None or arrival_end_sec <= arrival_start_sec:
        raise ValueError("Arrival window must be valid and end after start.")

    active_service_ids = commute.active_service_ids_for_date(calendar_df, calendar_dates_df, service_date)
    workplace_stops = commute.load_workplace_stops(lat, lon, args, all_gtfs_stops)
    if workplace_stops.empty:
        raise ValueError("No GTFS stops were found within the configured workplace distance.")

    stop_coord_lookup = {
        str(row.stop_id): (float(row.stop_lat), float(row.stop_lon))
        for row in district_gtfs_stops[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates("stop_id").itertuples(index=False)
    }
    stop_coord_lookup.update(
        {
            str(row.stop_id): (float(row.stop_lat), float(row.stop_lon))
            for row in assigned_stops[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates("stop_id").itertuples(index=False)
        }
    )
    stop_coord_lookup.update(
        {
            str(row.stop_id): (float(row.stop_lat), float(row.stop_lon))
            for row in workplace_stops[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates("stop_id").itertuples(index=False)
        }
    )

    trip_routes = commute.load_active_local_trip_routes(active_service_ids)
    relevant_stop_ids = (
        set(district_gtfs_stops["stop_id"].astype(str))
        | set(workplace_stops["stop_id"].astype(str))
        | set(assigned_stops["stop_id"].astype(str))
    )
    relevant_stop_times = commute.load_relevant_stop_times(
        active_trip_ids=set(trip_routes["trip_id"].astype(str)),
        relevant_stop_ids=relevant_stop_ids,
    )
    destination_occurrences = commute.collect_destination_trip_occurrences(
        workplace_stops,
        trip_routes,
        arrival_start_sec,
        arrival_end_sec,
        relevant_stop_times,
    )
    if destination_occurrences.empty:
        raise ValueError("No active local trips reach the workplace stop set in the selected arrival window.")

    trip2_stop_times = relevant_stop_times.loc[
        relevant_stop_times["trip_id"].isin(destination_occurrences["trip_id"].astype("string"))
    ].copy()
    direct_df, transfer_opportunities = commute.build_direct_options(destination_occurrences, trip2_stop_times, assigned_stops)

    computed_max_transfers = max(3, int(args.max_transfers))
    transfer_df = commute.build_transfer_options(
        downstream_opportunities=transfer_opportunities,
        trip_routes=trip_routes,
        assigned_stops=assigned_stops,
        stop_coord_lookup=stop_coord_lookup,
        stop_neighbor_lookup=stop_neighbor_lookup,
        relevant_stop_times=relevant_stop_times,
        max_transfers=computed_max_transfers,
        min_transfer_min=args.min_transfer_min,
        max_transfer_wait_min=args.max_transfer_wait_min,
        target_lat=lat,
        target_lon=lon,
    )

    options_df = pd.concat([direct_df, transfer_df], ignore_index=True)
    profile = commute.routing_profile_dict(args, lat, lon, service_date, computed_max_transfers)
    bundle_dir = commute.commute_cache_bundle_dir(profile)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        **profile,
        "district_nuts_id": str(district.iloc[0]["NUTS_ID"]),
        "district_name": str(district.iloc[0]["NUTS_NAME"]),
        "district_stadtteile": int(len(district_stdt_gdf)),
        "district_gtfs_stops": int(len(district_gtfs_stops)),
        "target_stop_count": int(len(workplace_stops)),
        "active_local_trips": int(trip_routes["trip_id"].nunique()),
        "destination_occurrences": int(len(destination_occurrences)),
        "option_count": int(len(options_df)),
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    options_df.to_pickle(bundle_dir / "options_raw.pkl")
    destination_occurrences.to_pickle(bundle_dir / "destination_occurrences.pkl")
    relevant_stop_times.to_pickle(bundle_dir / "relevant_stop_times.pkl")
    assigned_stops.to_pickle(bundle_dir / "assigned_stops.pkl")
    workplace_stops.to_pickle(bundle_dir / "workplace_stops.pkl")
    district_gtfs_stops.to_file(bundle_dir / "district_gtfs_stops.gpkg", driver="GPKG")

    print("\n=== Commute Compute Cache ===")
    print("target location coordinate:", f"{lat:.6f}, {lon:.6f}")
    print("district:", district.iloc[0]["NUTS_ID"], "-", district.iloc[0]["NUTS_NAME"])
    print("service date:", service_date.date())
    print("computed max transfers:", computed_max_transfers)
    print("active local trips on date:", trip_routes["trip_id"].nunique())
    print("destination trip occurrences in window:", len(destination_occurrences))
    print("direct options:", len(direct_df))
    print("transfer options:", len(transfer_df))
    print("cache bundle:", bundle_dir.relative_to(commute.ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

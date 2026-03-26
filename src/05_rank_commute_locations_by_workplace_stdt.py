from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import linear
import folium
from shapely.geometry import LineString, Point


ROOT = Path(__file__).resolve().parents[1]
GTFS_DIR = ROOT / "input" / "gtfs"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
PROCESSED_DIR = ROOT / "data" / "processed"
MAPS_DIR = ROOT / "outputs" / "maps"
COMMUTE_CACHE_DIR = PROCESSED_DIR / "commute_compute_cache_stdt"
DEFAULT_CONFIG_PATH = ROOT / "config" / "commute_defaults.json"

REQUIRED_CONFIG_KEYS = {
    "workplace_coordinate",
    "date",
    "arrival_start",
    "arrival_end",
    "max_workplace_stop_distance_m",
    "max_transfers",
    "transfer_radius_m",
    "min_transfer_radius_m",
    "max_transfer_radius_m",
    "min_transfer_min",
    "max_transfer_wait_min",
    "transfer_hate",
    "rent_importance",
    "time_importance",
    "frequency_importance",
    "green_space_importance",
    "supermarket_importance",
    "hospital_importance",
    "typical_size_m2",
    "nuts_path",
    "top_n",
}


def load_config_values(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        raise ValueError("A config path is required.")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    missing_keys = sorted(REQUIRED_CONFIG_KEYS - set(data))
    if missing_keys:
        raise ValueError(f"Config file is missing required keys: {', '.join(missing_keys)}")
    return data


def parse_args_with_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank residential Stadtteile in the target district using GTFS schedules, rent, and district-scoped mapping."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to a JSON config file with default values.")
    args = parser.parse_args()
    config_values = load_config_values(args.config)
    config_values["nuts_path"] = Path(config_values["nuts_path"])
    config_values["config"] = args.config
    return argparse.Namespace(**config_values)


def parse_coordinate_from_config(args: argparse.Namespace) -> tuple[float, float]:
    coordinate_text = str(getattr(args, "workplace_coordinate", "") or "").strip()
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", coordinate_text)
    if not match:
        raise ValueError('Config value "workplace_coordinate" must look like "51.46479, 7.01156".')
    return float(match.group(1)), float(match.group(2))


def parse_gtfs_time_to_seconds(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    if not text or text == "nan":
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_hhmm(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    total = int(round(float(value)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def route_style(route_type_value: object) -> tuple[str, int]:
    try:
        route_type = int(float(route_type_value))
    except (TypeError, ValueError):
        route_type = 3
    if route_type == 0:
        return "#e34a33", 4
    if route_type == 1:
        return "#5e3c99", 4
    if route_type == 2:
        return "#1b7837", 5
    if route_type == 3:
        return "#2b8cbe", 3
    if route_type in {4, 5, 6, 7}:
        return "#636363", 3
    return "#756bb1", 3


def route_color(route_key: object, route_type_value: object) -> tuple[str, int]:
    return route_style(route_type_value)


def build_line_key(route_label: object, route_type_value: object) -> str:
    label = str(route_label or "").strip().upper()
    route_type = str(route_type_value or "").strip()
    return f"{route_type}|{label}"


def distance_to_target_for_stop(
    stop_id: object,
    stop_coord_lookup: dict[str, tuple[float, float]],
    target_lat: float,
    target_lon: float,
) -> float | None:
    coords = stop_coord_lookup.get(str(stop_id))
    if coords is None:
        return None
    return haversine_m(float(coords[0]), float(coords[1]), target_lat, target_lon)


def classify_local_route(route_type_value: object, route_label: object) -> bool:
    label = str(route_label or "").strip().upper()
    try:
        route_type = int(float(route_type_value))
    except (TypeError, ValueError):
        return False
    if route_type in {0, 1, 3, 4, 5, 6, 7}:
        return True
    if route_type == 2:
        if label.startswith(("IC", "ICE", "EC", "ECE", "THA", "NJ", "FLX")):
            return False
        if label.startswith(("S", "U", "RB", "RE", "IRE", "MEX", "RS")):
            return True
        return True
    return False


def choose_default_service_date(calendar_df: pd.DataFrame) -> pd.Timestamp:
    calendar_df = calendar_df.copy()
    calendar_df["start_date"] = pd.to_datetime(calendar_df["start_date"].astype(str), format="%Y%m%d")
    first_start = calendar_df["start_date"].min()
    for offset in range(14):
        date = first_start + pd.Timedelta(days=offset)
        if date.weekday() < 5:
            return date
    return first_start


def clamp_preference(value: object) -> float:
    return float(max(0, min(10, int(value))))


def normalize_transfer_radius_m(value: object, min_radius_m: object, max_radius_m: object) -> float:
    min_radius = float(min_radius_m)
    max_radius = float(max_radius_m)
    if max_radius < min_radius:
        raise ValueError("max_transfer_radius_m must be greater than or equal to min_transfer_radius_m.")
    return float(max(min_radius, min(max_radius, float(value))))


def resolve_preferences(args: argparse.Namespace) -> dict[str, float]:
    return {
        "rent": clamp_preference(args.rent_importance),
        "time": clamp_preference(args.time_importance),
        "frequency": clamp_preference(args.frequency_importance),
        "transfer": clamp_preference(args.transfer_hate),
        "green_space": clamp_preference(args.green_space_importance),
        "supermarket": clamp_preference(args.supermarket_importance),
        "hospital": clamp_preference(args.hospital_importance),
    }


def resolve_transfer_penalty_minutes(args: argparse.Namespace) -> float:
    return clamp_preference(args.transfer_hate)


def routing_profile_dict(
    args: argparse.Namespace,
    lat: float,
    lon: float,
    service_date: pd.Timestamp,
    computed_max_transfers: int,
) -> dict[str, Any]:
    return {
        "workplace_coordinate": f"{lat:.8f},{lon:.8f}",
        "service_date": str(service_date.date()),
        "arrival_start": str(args.arrival_start),
        "arrival_end": str(args.arrival_end),
        "max_workplace_stop_distance_m": float(args.max_workplace_stop_distance_m),
        "computed_max_transfers": int(computed_max_transfers),
        "transfer_radius_m": float(args.transfer_radius_m),
        "min_transfer_min": float(args.min_transfer_min),
        "max_transfer_wait_min": float(args.max_transfer_wait_min),
        "nuts_path": str(Path(args.nuts_path)),
    }


def commute_cache_bundle_dir(profile: dict[str, Any]) -> Path:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    profile_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return COMMUTE_CACHE_DIR / profile_hash


def normalize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - min_value) / (max_value - min_value)


def active_service_ids_for_date(
    calendar_df: pd.DataFrame,
    calendar_dates_df: pd.DataFrame,
    service_date: pd.Timestamp,
) -> set[str]:
    calendar_df = calendar_df.copy()
    calendar_dates_df = calendar_dates_df.copy()
    calendar_df["start_date"] = pd.to_datetime(calendar_df["start_date"].astype(str), format="%Y%m%d")
    calendar_df["end_date"] = pd.to_datetime(calendar_df["end_date"].astype(str), format="%Y%m%d")
    calendar_dates_df["date"] = pd.to_datetime(calendar_dates_df["date"].astype(str), format="%Y%m%d")

    weekday_column = service_date.day_name().lower()
    base = calendar_df.loc[
        (calendar_df["start_date"] <= service_date)
        & (calendar_df["end_date"] >= service_date)
        & (calendar_df[weekday_column] == 1),
        "service_id",
    ]
    active = set(base.astype(str))

    exceptions = calendar_dates_df.loc[calendar_dates_df["date"] == service_date, ["service_id", "exception_type"]]
    for row in exceptions.itertuples(index=False):
        if int(row.exception_type) == 1:
            active.add(str(row.service_id))
        elif int(row.exception_type) == 2:
            active.discard(str(row.service_id))
    return active


def standardize_area_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def detect_workplace_district(lat: float, lon: float, nuts_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nuts = gpd.read_file(nuts_path)
    nuts = nuts.loc[(nuts["CNTR_CODE"] == "DE") & (nuts["LEVL_CODE"] == 3)].copy()
    if nuts.empty:
        raise ValueError(f"No German NUTS3 polygons found in {nuts_path}.")

    point_wgs84 = gpd.GeoDataFrame({"workplace": [1]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    point_nuts = point_wgs84.to_crs(nuts.crs)
    matched = gpd.sjoin(point_nuts, nuts, how="left", predicate="within")
    if matched["NUTS_ID"].isna().all():
        raise ValueError("Workplace coordinate does not fall inside any German NUTS3 district in the provided package.")

    district_id = matched.iloc[0]["NUTS_ID"]
    district = nuts.loc[nuts["NUTS_ID"] == district_id].copy()
    return district, point_wgs84


def load_district_stdt_geodata(district: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    source_path = PROCESSED_DIR / "rent_transit_quality_by_stdt.gpkg"
    if not source_path.exists():
        raise FileNotFoundError(f"Could not find required stage-3 quality output: {source_path}")

    stdt_gdf = gpd.read_file(source_path)
    if "stdt_id" not in stdt_gdf.columns:
        raise ValueError(f"{source_path.name} does not contain a 'stdt_id' column.")
    stdt_gdf["stdt_id"] = standardize_area_id(stdt_gdf["stdt_id"])
    if "stdt_name" not in stdt_gdf.columns:
        if "name_de" in stdt_gdf.columns or "name" in stdt_gdf.columns:
            stdt_gdf["stdt_name"] = stdt_gdf.get("name_de", pd.Series(pd.NA, index=stdt_gdf.index)).fillna(
                stdt_gdf.get("name", pd.Series(pd.NA, index=stdt_gdf.index))
            ).astype("string")
        else:
            stdt_gdf["stdt_name"] = stdt_gdf["stdt_id"].astype("string")
    if "municipality_name" not in stdt_gdf.columns:
        stdt_gdf["municipality_name"] = pd.NA

    district_in_area_crs = district.to_crs(stdt_gdf.crs)
    district_geom = district_in_area_crs.geometry.union_all()
    district_stdt = stdt_gdf.loc[stdt_gdf.geometry.notna() & stdt_gdf.intersects(district_geom)].copy()
    district_stdt["geometry"] = district_stdt.geometry.intersection(district_geom)
    district_stdt = district_stdt.loc[~district_stdt.geometry.is_empty].copy()
    return district_stdt.set_geometry("geometry")


def load_rent_data(area_filter: set[str]) -> pd.DataFrame:
    rent = pd.read_csv(PROCESSED_DIR / "rent_transit_quality_by_stdt.csv", dtype={"stdt_id": "string", "stdt_name": "string"})
    keep_cols = [
        "stdt_id",
        "stdt_name",
        "municipality_name",
        "population",
        "stdt_area_km2",
        "grid_cell_count",
        "mean_rent_per_m2",
        "median_rent_per_m2",
        "std_rent_per_m2",
        "q25_rent_per_m2",
        "q75_rent_per_m2",
        "green_space_share",
        "green_space_patch_density_km2",
        "street_tree_density_km2",
        "supermarket_count",
        "hospital_count",
    ]
    rent = rent[[col for col in keep_cols if col in rent.columns]].copy()
    rent["stdt_id"] = standardize_area_id(rent["stdt_id"])
    return rent.loc[rent["stdt_id"].isin(area_filter)].copy()


def load_assigned_stops(area_filter: set[str]) -> pd.DataFrame:
    stops = pd.read_csv(
        INTERMEDIATE_DIR / "gtfs_stops_with_stdt.csv.gz",
        dtype={"stop_id": "string", "stop_name": "string", "parent_station": "string", "stdt_id": "string", "stdt_name": "string"},
        compression="gzip",
    )
    stops["stdt_id"] = standardize_area_id(stops["stdt_id"])
    stops = stops.loc[stops["stdt_id"].isin(area_filter)].copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    return stops.loc[stops["stop_lat"].notna() & stops["stop_lon"].notna()].copy()


def load_all_gtfs_stops() -> pd.DataFrame:
    stops = pd.read_csv(
        GTFS_DIR / "stops.txt",
        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
        dtype={"stop_id": "string", "stop_name": "string", "parent_station": "string"},
    )
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops["location_type"] = pd.to_numeric(stops["location_type"], errors="coerce")
    stops = stops.loc[
        stops["stop_lat"].notna()
        & stops["stop_lon"].notna()
        & (stops["location_type"].isna() | stops["location_type"].eq(0))
    ].copy()
    return stops


def load_all_gtfs_stops_in_district(district: gpd.GeoDataFrame, all_stops: pd.DataFrame) -> gpd.GeoDataFrame:
    stops_gdf = gpd.GeoDataFrame(all_stops.copy(), geometry=gpd.points_from_xy(all_stops["stop_lon"], all_stops["stop_lat"]), crs="EPSG:4326")
    district_stops = gpd.sjoin(stops_gdf.to_crs(district.crs), district[["geometry"]], how="inner", predicate="within")
    return district_stops.drop(columns=["index_right"]).to_crs("EPSG:4326")


def build_stop_neighbor_lookup(stops_gdf: gpd.GeoDataFrame, radius_m: float) -> dict[str, set[str]]:
    if stops_gdf.empty:
        return {}
    local = stops_gdf.to_crs("EPSG:3035").copy()
    local["stop_id"] = local["stop_id"].astype("string")
    sindex = local.sindex
    lookup: dict[str, set[str]] = {}
    for row in local.itertuples(index=False):
        bounds = row.geometry.buffer(radius_m).bounds
        candidate_idx = list(sindex.intersection(bounds))
        candidates = local.iloc[candidate_idx]
        near = candidates.loc[candidates.geometry.distance(row.geometry) <= radius_m, "stop_id"].astype(str)
        lookup[str(row.stop_id)] = set(near.tolist())
    return lookup


def load_workplace_stops(lat: float, lon: float, args: argparse.Namespace, all_stops: pd.DataFrame) -> pd.DataFrame:
    stops = all_stops.copy()
    stops["distance_to_work_m"] = stops.apply(
        lambda row: haversine_m(lat, lon, float(row["stop_lat"]), float(row["stop_lon"])),
        axis=1,
    )
    stops = stops.loc[stops["distance_to_work_m"] <= args.max_workplace_stop_distance_m].copy()
    return stops.sort_values("distance_to_work_m")


def clip_stop_dataframe_to_district(stops: pd.DataFrame, district: gpd.GeoDataFrame) -> pd.DataFrame:
    if stops.empty:
        return stops.copy()
    stops_gdf = gpd.GeoDataFrame(
        stops.copy(),
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )
    district_wgs84 = district.to_crs("EPSG:4326")
    clipped = gpd.sjoin(stops_gdf, district_wgs84[["geometry"]], how="inner", predicate="within")
    return clipped.drop(columns=["geometry", "index_right"])


def load_active_local_trip_routes(active_service_ids: set[str]) -> pd.DataFrame:
    routes = pd.read_csv(
        GTFS_DIR / "routes.txt",
        usecols=["route_id", "route_type", "route_short_name", "route_long_name", "agency_id"],
        dtype="string",
    )
    routes["route_label"] = (
        routes["route_short_name"].fillna("").str.strip().where(
            routes["route_short_name"].fillna("").str.strip().ne(""),
            routes["route_long_name"],
        )
    )
    routes["route_label"] = routes["route_label"].fillna(routes["route_id"])
    routes["line_key"] = routes.apply(lambda row: build_line_key(row["route_label"], row["route_type"]), axis=1)
    routes["is_local"] = routes.apply(lambda row: classify_local_route(row["route_type"], row["route_label"]), axis=1)

    trips = pd.read_csv(GTFS_DIR / "trips.txt", usecols=["trip_id", "route_id", "service_id"], dtype="string")
    trips = trips.loc[trips["service_id"].isin(active_service_ids)].copy()
    trips = trips.merge(routes, on="route_id", how="left")
    trips = trips.loc[trips["is_local"] == True].copy()
    return trips[["trip_id", "route_id", "route_label", "line_key", "agency_id", "route_type"]]


def collect_destination_trip_occurrences(
    workplace_stops: pd.DataFrame,
    trip_routes: pd.DataFrame,
    arrival_start_sec: float,
    arrival_end_sec: float,
    relevant_stop_times: pd.DataFrame,
) -> pd.DataFrame:
    workplace_stop_ids = set(workplace_stops["stop_id"].tolist())
    active_trip_ids = set(trip_routes["trip_id"].tolist())
    destination_occurrences = relevant_stop_times.loc[
        relevant_stop_times["trip_id"].isin(active_trip_ids)
        & relevant_stop_times["stop_id"].isin(workplace_stop_ids)
        & relevant_stop_times["arrival_sec"].notna()
        & relevant_stop_times["arrival_sec"].between(arrival_start_sec, arrival_end_sec)
    ].copy()
    if destination_occurrences.empty:
        return destination_occurrences

    destination_occurrences = destination_occurrences.merge(
        workplace_stops[["stop_id", "stop_name", "distance_to_work_m"]].rename(
            columns={"stop_id": "work_stop_id", "stop_name": "work_stop_name"}
        ),
        left_on="stop_id",
        right_on="work_stop_id",
        how="left",
    )
    destination_occurrences = destination_occurrences.merge(trip_routes, on="trip_id", how="left")
    destination_occurrences = destination_occurrences.rename(columns={"stop_sequence": "dest_sequence"})
    return destination_occurrences


def load_relevant_stop_times(
    active_trip_ids: set[str],
    relevant_stop_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for chunk in pd.read_csv(
        GTFS_DIR / "stop_times.txt",
        usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"],
        dtype={
            "trip_id": "string",
            "stop_id": "string",
            "stop_sequence": "Int64",
            "arrival_time": "string",
            "departure_time": "string",
        },
        chunksize=2_000_000,
    ):
        chunk = chunk.loc[chunk["trip_id"].isin(active_trip_ids) & chunk["stop_id"].isin(relevant_stop_ids)].copy()
        if not chunk.empty:
            rows.append(chunk)

    if not rows:
        return pd.DataFrame(columns=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time", "arrival_sec", "departure_sec"])

    stop_times = pd.concat(rows, ignore_index=True)
    stop_times["trip_id"] = stop_times["trip_id"].astype("string")
    stop_times["stop_id"] = stop_times["stop_id"].astype("string")
    stop_times["arrival_sec"] = stop_times["arrival_time"].map(parse_gtfs_time_to_seconds)
    stop_times["departure_sec"] = stop_times["departure_time"].map(parse_gtfs_time_to_seconds)
    return stop_times


def build_direct_options(
    destination_occurrences: pd.DataFrame,
    trip_stop_times: pd.DataFrame,
    assigned_stops: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stop_lookup = assigned_stops.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon", "stdt_id"]].to_dict("index")
    trip_stop_times = trip_stop_times.copy()
    trip_stop_times["arrival_sec"] = trip_stop_times["arrival_time"].map(parse_gtfs_time_to_seconds)
    trip_stop_times["departure_sec"] = trip_stop_times["departure_time"].map(parse_gtfs_time_to_seconds)

    direct_records = []
    transfer_opportunities = []
    trip_groups = {trip_id: df.sort_values("stop_sequence") for trip_id, df in trip_stop_times.groupby("trip_id")}
    for occ in destination_occurrences.itertuples(index=False):
        trip_df = trip_groups.get(occ.trip_id)
        if trip_df is None:
            continue
        earlier = trip_df.loc[trip_df["stop_sequence"] < occ.dest_sequence].copy()
        if earlier.empty:
            continue

        for row in earlier.itertuples(index=False):
            stop_meta = stop_lookup.get(str(row.stop_id))
            if stop_meta is None:
                continue
            if row.departure_sec is None or occ.arrival_sec is None or occ.arrival_sec <= row.departure_sec:
                continue
            travel_minutes = (float(occ.arrival_sec) - float(row.departure_sec)) / 60.0
            direct_records.append(
                {
                    "stdt_id": stop_meta["stdt_id"],
                    "origin_stop_id": row.stop_id,
                    "origin_stop_name": stop_meta["stop_name"],
                    "origin_departure_sec": float(row.departure_sec),
                    "arrival_sec": float(occ.arrival_sec),
                    "travel_time_min": travel_minutes,
                    "transfer_count": 0,
                    "route_path": str(occ.route_label),
                    "first_route_label": occ.route_label,
                    "second_route_label": None,
                    "first_route_id": occ.route_id,
                    "first_line_key": occ.line_key,
                    "second_route_id": None,
                    "second_line_key": None,
                    "final_work_stop_name": occ.work_stop_name,
                    "final_work_stop_id": occ.work_stop_id,
                    "final_work_stop_distance_m": float(occ.distance_to_work_m),
                    "feeder_trip_chain_ids": "",
                    "feeder_route_chain_ids": "",
                    "feeder_route_chain_labels": "",
                    "feeder_alight_stop_chain_ids": "",
                    "option_key": f"{stop_meta['stdt_id']}|{row.stop_id}|{occ.trip_id}|{int(row.departure_sec)}|0",
                }
            )

            transfer_opportunities.append(
                {
                    "trip2_id": occ.trip_id,
                    "trip2_route_id": occ.route_id,
                    "trip2_route_label": occ.route_label,
                    "trip2_line_key": occ.line_key,
                    "transfer_stop_id": row.stop_id,
                    "transfer_departure_sec": float(row.departure_sec),
                    "arrival_sec": float(occ.arrival_sec),
                    "transfer_count": 0,
                    "route_path": str(occ.route_label),
                    "route2_label": occ.route_label,
                    "work_stop_name": occ.work_stop_name,
                    "work_stop_id": occ.work_stop_id,
                    "work_stop_distance_m": float(occ.distance_to_work_m),
                    "direct_transfer_stop_id": row.stop_id,
                    "final_feeder_trip_id": None,
                    "final_feeder_route_id": None,
                    "final_feeder_route_label": None,
                    "final_feeder_line_key": None,
                    "feeder_alight_stop_id": None,
                    "feeder_trip_chain_ids": "",
                    "feeder_route_chain_ids": "",
                    "feeder_route_chain_labels": "",
                    "feeder_alight_stop_chain_ids": "",
                }
            )

    return pd.DataFrame(direct_records), pd.DataFrame(transfer_opportunities)


def build_transfer_options(
    downstream_opportunities: pd.DataFrame,
    trip_routes: pd.DataFrame,
    assigned_stops: pd.DataFrame,
    stop_coord_lookup: dict[str, tuple[float, float]],
    stop_neighbor_lookup: dict[str, set[str]],
    relevant_stop_times: pd.DataFrame,
    max_transfers: int,
    min_transfer_min: float,
    max_transfer_wait_min: float,
    target_lat: float,
    target_lon: float,
) -> pd.DataFrame:
    if downstream_opportunities.empty or max_transfers <= 0:
        return pd.DataFrame()

    stop_lookup = assigned_stops.set_index("stop_id")[["stop_name", "stdt_id"]].to_dict("index")
    trip_meta_lookup = trip_routes.set_index("trip_id")[["route_label", "route_id", "line_key"]].to_dict("index")
    direct_transfer_summary = (
        downstream_opportunities.assign(
            trip2_line_key=downstream_opportunities["trip2_line_key"].astype("string"),
            transfer_departure_minute_bin=(pd.to_numeric(downstream_opportunities["transfer_departure_sec"], errors="coerce") // 60).astype("Int64"),
        )
        .groupby("direct_transfer_stop_id", dropna=False)
        .agg(
            downstream_direct_line_count=("trip2_line_key", "nunique"),
            downstream_direct_trip_count=("trip2_id", "nunique"),
            downstream_direct_minute_count=("transfer_departure_minute_bin", "nunique"),
        )
        .reset_index()
    )
    direct_transfer_summary_lookup = direct_transfer_summary.set_index("direct_transfer_stop_id").to_dict("index")
    active_trip_ids = set(trip_routes["trip_id"].tolist())
    all_transfer_records: list[dict[str, Any]] = []
    current_opportunities = downstream_opportunities.copy()
    min_transfer_sec = min_transfer_min * 60.0
    max_transfer_wait_sec = max_transfer_wait_min * 60.0

    for next_transfer_count in range(1, max_transfers + 1):
        if current_opportunities.empty:
            break

        opportunity_lookup: dict[str, list[Any]] = defaultdict(list)
        candidate_arrival_stop_ids: set[str] = set()
        for opp in current_opportunities.itertuples(index=False):
            nearby_stops = stop_neighbor_lookup.get(str(opp.transfer_stop_id), {str(opp.transfer_stop_id)})
            for nearby_stop_id in nearby_stops:
                opportunity_lookup[str(nearby_stop_id)].append(opp)
                candidate_arrival_stop_ids.add(str(nearby_stop_id))

        trip_occurrences = relevant_stop_times.loc[
            relevant_stop_times["trip_id"].isin(active_trip_ids)
            & relevant_stop_times["stop_id"].isin(candidate_arrival_stop_ids)
            & relevant_stop_times["arrival_sec"].notna()
        ].copy()
        if trip_occurrences.empty:
            break

        trip_ids = set(trip_occurrences["trip_id"].astype(str).tolist())
        full_trip_stop_times = relevant_stop_times.loc[relevant_stop_times["trip_id"].isin(trip_ids)].copy()
        trip_groups = {trip_id: df.sort_values("stop_sequence") for trip_id, df in full_trip_stop_times.groupby("trip_id")}

        candidate_connections_by_trip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        next_opportunity_records: list[dict[str, Any]] = []
        for occ in trip_occurrences.itertuples(index=False):
            candidates = opportunity_lookup.get(str(occ.stop_id), [])
            if not candidates:
                continue

            earliest = occ.arrival_sec + min_transfer_sec
            latest = occ.arrival_sec + max_transfer_wait_sec
            feasible = [
                opp
                for opp in candidates
                if opp.transfer_departure_sec >= earliest
                and opp.transfer_departure_sec <= latest
                and str(opp.trip2_id) != str(occ.trip_id)
            ]
            if not feasible:
                continue
            upstream_meta = trip_meta_lookup.get(str(occ.trip_id), {})
            upstream_route_label = upstream_meta.get("route_label")
            upstream_route_id = upstream_meta.get("route_id")
            upstream_line_key = upstream_meta.get("line_key")
            feasible_candidates: list[dict[str, Any]] = []
            for downstream in feasible:
                downstream_line_key = getattr(downstream, "trip2_line_key", None)
                if upstream_line_key and downstream_line_key and str(upstream_line_key) == str(downstream_line_key):
                    continue
                route_path = f"{upstream_route_label} > {downstream.route_path}" if upstream_route_label else str(downstream.route_path)
                route_chain = [part.strip() for part in str(route_path).split(">") if part and part.strip()]
                first_route_label = route_chain[0] if route_chain else None
                second_route_label = route_chain[1] if len(route_chain) > 1 else None
                direct_transfer_stop_id = str(downstream.direct_transfer_stop_id)
                if next_transfer_count == 1:
                    final_feeder_trip_id = str(occ.trip_id)
                    final_feeder_route_id = upstream_route_id
                    final_feeder_route_label = upstream_route_label
                    final_feeder_line_key = upstream_line_key
                    feeder_alight_stop_id = str(occ.stop_id)
                else:
                    final_feeder_trip_id = str(downstream.final_feeder_trip_id)
                    final_feeder_route_id = downstream.final_feeder_route_id
                    final_feeder_route_label = downstream.final_feeder_route_label
                    final_feeder_line_key = downstream.final_feeder_line_key
                    feeder_alight_stop_id = str(downstream.feeder_alight_stop_id)

                downstream_trip_chain = [value for value in str(getattr(downstream, "feeder_trip_chain_ids", "") or "").split("|") if value]
                downstream_route_id_chain = [value for value in str(getattr(downstream, "feeder_route_chain_ids", "") or "").split("|") if value]
                downstream_route_label_chain = [value for value in str(getattr(downstream, "feeder_route_chain_labels", "") or "").split("|") if value]
                downstream_alight_chain = [value for value in str(getattr(downstream, "feeder_alight_stop_chain_ids", "") or "").split("|") if value]
                direct_transfer_summary_row = direct_transfer_summary_lookup.get(direct_transfer_stop_id, {})

                feasible_candidates.append(
                    {
                        "arrival_stop_id": str(occ.stop_id),
                        "arrival_stop_sequence": int(occ.stop_sequence),
                        "arrival_stop_arrival_sec": float(occ.arrival_sec),
                        "downstream": downstream,
                        "downstream_line_key": downstream_line_key,
                        "route_path": route_path,
                        "first_route_label": first_route_label,
                        "second_route_label": second_route_label,
                        "direct_transfer_stop_id": direct_transfer_stop_id,
                        "final_feeder_trip_id": final_feeder_trip_id,
                        "final_feeder_route_id": final_feeder_route_id,
                        "final_feeder_route_label": final_feeder_route_label,
                        "final_feeder_line_key": final_feeder_line_key,
                        "feeder_alight_stop_id": feeder_alight_stop_id,
                        "feeder_trip_chain_ids": "|".join([str(occ.trip_id), *downstream_trip_chain]),
                        "feeder_route_chain_ids": "|".join([str(upstream_route_id or ""), *downstream_route_id_chain]).strip("|"),
                        "feeder_route_chain_labels": "|".join([str(upstream_route_label or ""), *downstream_route_label_chain]).strip("|"),
                        "feeder_alight_stop_chain_ids": "|".join([str(occ.stop_id), *downstream_alight_chain]),
                        "direct_transfer_distance_to_target_m": distance_to_target_for_stop(
                            direct_transfer_stop_id,
                            stop_coord_lookup,
                            target_lat,
                            target_lon,
                        ),
                        "selected_transfer_wait_min": (float(downstream.transfer_departure_sec) - float(occ.arrival_sec)) / 60.0,
                        "downstream_direct_line_count": int(direct_transfer_summary_row.get("downstream_direct_line_count", 0) or 0),
                        "downstream_direct_trip_count": int(direct_transfer_summary_row.get("downstream_direct_trip_count", 0) or 0),
                        "downstream_direct_minute_count": int(direct_transfer_summary_row.get("downstream_direct_minute_count", 0) or 0),
                    }
                )

            if feasible_candidates:
                candidate_connections_by_trip[str(occ.trip_id)].extend(feasible_candidates)

        for trip_id, trip_candidates in candidate_connections_by_trip.items():
            trip_df = trip_groups.get(trip_id)
            if trip_df is None or trip_df.empty:
                continue

            trip_candidates = sorted(
                trip_candidates,
                key=lambda item: (
                    item["arrival_stop_sequence"],
                    item["downstream"].arrival_sec,
                    item["downstream"].transfer_count,
                ),
            )
            arrival_sequences = [item["arrival_stop_sequence"] for item in trip_candidates]

            suffix_best_idx = [0] * len(trip_candidates)
            best_idx = len(trip_candidates) - 1
            suffix_best_idx[-1] = best_idx
            for idx in range(len(trip_candidates) - 2, -1, -1):
                current = trip_candidates[idx]
                best = trip_candidates[best_idx]
                current_key = (
                    float(current["downstream"].arrival_sec),
                    int(current["downstream"].transfer_count),
                    float(current["arrival_stop_arrival_sec"]),
                )
                best_key = (
                    float(best["downstream"].arrival_sec),
                    int(best["downstream"].transfer_count),
                    float(best["arrival_stop_arrival_sec"]),
                )
                if current_key <= best_key:
                    best_idx = idx
                suffix_best_idx[idx] = best_idx

            upstream_meta = trip_meta_lookup.get(str(trip_id), {})
            upstream_route_label = upstream_meta.get("route_label")
            upstream_route_id = upstream_meta.get("route_id")
            upstream_line_key = upstream_meta.get("line_key")

            for row in trip_df.itertuples(index=False):
                stop_meta = stop_lookup.get(str(row.stop_id))
                if stop_meta is None or row.departure_sec is None:
                    continue

                candidate_idx = bisect_right(arrival_sequences, int(row.stop_sequence))
                if candidate_idx >= len(trip_candidates):
                    continue
                origin_distance_to_target_m = distance_to_target_for_stop(
                    row.stop_id,
                    stop_coord_lookup,
                    target_lat,
                    target_lon,
                )
                available_candidates = trip_candidates[candidate_idx:]
                selected: dict[str, Any]
                if origin_distance_to_target_m is not None:
                    forward_candidates = [
                        item
                        for item in available_candidates
                        if item["direct_transfer_distance_to_target_m"] is not None
                        and float(item["direct_transfer_distance_to_target_m"]) <= float(origin_distance_to_target_m) + 100.0
                    ]
                    if forward_candidates:
                        selected = min(
                            forward_candidates,
                            key=lambda item: (
                                float(item["downstream"].arrival_sec),
                                int(item["downstream"].transfer_count),
                                float(item["direct_transfer_distance_to_target_m"]),
                            ),
                        )
                    else:
                        selected = trip_candidates[suffix_best_idx[candidate_idx]]
                else:
                    selected = trip_candidates[suffix_best_idx[candidate_idx]]
                best_downstream = selected["downstream"]
                if float(best_downstream.arrival_sec) <= float(row.departure_sec):
                    continue
                if (
                    origin_distance_to_target_m is not None
                    and selected["direct_transfer_distance_to_target_m"] is not None
                    and float(selected["direct_transfer_distance_to_target_m"]) > float(origin_distance_to_target_m) + 100.0
                ):
                    continue

                travel_minutes = (float(best_downstream.arrival_sec) - float(row.departure_sec)) / 60.0
                all_transfer_records.append(
                    {
                        "stdt_id": stop_meta["stdt_id"],
                        "origin_stop_id": row.stop_id,
                        "origin_stop_name": stop_meta["stop_name"],
                        "origin_departure_sec": float(row.departure_sec),
                        "arrival_sec": float(best_downstream.arrival_sec),
                        "travel_time_min": travel_minutes,
                        "transfer_count": next_transfer_count,
                        "route_path": selected["route_path"],
                        "first_route_label": selected["first_route_label"],
                        "second_route_label": selected["second_route_label"],
                        "first_route_id": upstream_route_id,
                        "first_line_key": upstream_line_key,
                        "second_route_id": best_downstream.trip2_route_id,
                        "second_line_key": selected["downstream_line_key"],
                        "final_work_stop_name": best_downstream.work_stop_name,
                        "final_work_stop_id": best_downstream.work_stop_id,
                        "final_work_stop_distance_m": float(best_downstream.work_stop_distance_m),
                        "direct_transfer_stop_id": selected["direct_transfer_stop_id"],
                        "final_feeder_trip_id": selected["final_feeder_trip_id"],
                        "final_feeder_route_id": selected["final_feeder_route_id"],
                        "final_feeder_route_label": selected["final_feeder_route_label"],
                        "final_feeder_line_key": selected["final_feeder_line_key"],
                        "feeder_alight_stop_id": selected["feeder_alight_stop_id"],
                        "selected_transfer_wait_min": selected["selected_transfer_wait_min"],
                        "downstream_direct_line_count": selected["downstream_direct_line_count"],
                        "downstream_direct_trip_count": selected["downstream_direct_trip_count"],
                        "downstream_direct_minute_count": selected["downstream_direct_minute_count"],
                        "feeder_trip_chain_ids": selected["feeder_trip_chain_ids"],
                        "feeder_route_chain_ids": selected["feeder_route_chain_ids"],
                        "feeder_route_chain_labels": selected["feeder_route_chain_labels"],
                        "feeder_alight_stop_chain_ids": selected["feeder_alight_stop_chain_ids"],
                        "option_key": f"{stop_meta['stdt_id']}|{row.stop_id}|{trip_id}|{int(row.departure_sec)}|{next_transfer_count}",
                    }
                )
                next_opportunity_records.append(
                    {
                        "trip2_id": trip_id,
                        "trip2_route_id": upstream_route_id,
                        "trip2_route_label": upstream_route_label,
                        "trip2_line_key": upstream_line_key,
                        "transfer_stop_id": row.stop_id,
                        "transfer_departure_sec": float(row.departure_sec),
                        "arrival_sec": float(best_downstream.arrival_sec),
                        "transfer_count": next_transfer_count,
                        "route_path": selected["route_path"],
                        "route2_label": upstream_route_label,
                        "work_stop_name": best_downstream.work_stop_name,
                        "work_stop_id": best_downstream.work_stop_id,
                        "work_stop_distance_m": float(best_downstream.work_stop_distance_m),
                        "direct_transfer_stop_id": selected["direct_transfer_stop_id"],
                        "final_feeder_trip_id": selected["final_feeder_trip_id"],
                        "final_feeder_route_id": selected["final_feeder_route_id"],
                        "final_feeder_route_label": selected["final_feeder_route_label"],
                        "final_feeder_line_key": selected["final_feeder_line_key"],
                        "feeder_alight_stop_id": selected["feeder_alight_stop_id"],
                        "selected_transfer_wait_min": selected["selected_transfer_wait_min"],
                        "downstream_direct_line_count": selected["downstream_direct_line_count"],
                        "downstream_direct_trip_count": selected["downstream_direct_trip_count"],
                        "downstream_direct_minute_count": selected["downstream_direct_minute_count"],
                        "feeder_trip_chain_ids": selected["feeder_trip_chain_ids"],
                        "feeder_route_chain_ids": selected["feeder_route_chain_ids"],
                        "feeder_route_chain_labels": selected["feeder_route_chain_labels"],
                        "feeder_alight_stop_chain_ids": selected["feeder_alight_stop_chain_ids"],
                    }
                )

        current_opportunities = pd.DataFrame(next_opportunity_records)

    return pd.DataFrame(all_transfer_records)


def aggregate_and_rank(
    options_df: pd.DataFrame,
    rent_df: pd.DataFrame,
    args: argparse.Namespace,
    window_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_labels = {
        "rent": "affordability",
        "time": "travel time",
        "frequency": "service richness",
        "transfer": "transfer comfort",
        "green_space": "green space",
        "supermarket": "supermarkets",
        "hospital": "hospitals",
    }

    preferences = resolve_preferences(args)
    transfer_penalty_min = resolve_transfer_penalty_minutes(args)
    if options_df.empty:
        raise ValueError("No feasible commute options were found for the selected workplace/date/window.")

    options_df["commute_generalized_cost"] = options_df["travel_time_min"] + options_df["transfer_count"] * transfer_penalty_min
    options_df = options_df.drop_duplicates(subset=["option_key"]).copy()
    options_df["journey_choice_key"] = (
        options_df["stdt_id"].astype("string")
        + "|"
        + options_df["origin_stop_id"].astype("string")
        + "|"
        + options_df["origin_departure_sec"].round().astype(int).astype(str)
    )
    options_df = (
        options_df.sort_values(
            ["journey_choice_key", "commute_generalized_cost", "travel_time_min", "transfer_count", "arrival_sec"],
            ascending=[True, True, True, True, True],
        )
        .drop_duplicates(subset=["journey_choice_key"], keep="first")
        .reset_index(drop=True)
    )
    options_df["departure_hhmm"] = options_df["origin_departure_sec"].map(seconds_to_hhmm)
    options_df["arrival_hhmm"] = options_df["arrival_sec"].map(seconds_to_hhmm)
    options_df["departure_minute_bin"] = (options_df["origin_departure_sec"] // 60).astype(int)
    options_df.to_csv(PROCESSED_DIR / "commute_options_by_workplace_stdt.csv", index=False, encoding="utf-8")

    grouped = []
    for stdt_id, group in options_df.groupby("stdt_id"):
        best = group.sort_values(["commute_generalized_cost", "travel_time_min", "transfer_count"]).iloc[0]
        direct_group = group.loc[group["transfer_count"] == 0]
        transfer_group = group.loc[group["transfer_count"] > 0]
        unique_departure_minutes = group["departure_minute_bin"].nunique()
        direct_line_count = direct_group["first_line_key"].dropna().astype("string").nunique() if not direct_group.empty else 0
        useful_origin_stop_count = group["origin_stop_id"].dropna().astype("string").nunique()
        transfer_line_count = transfer_group["first_line_key"].dropna().astype("string").nunique() if not transfer_group.empty else 0
        transfer_stop_count = transfer_group["direct_transfer_stop_id"].dropna().astype("string").nunique() if not transfer_group.empty else 0
        useful_line_count_total = pd.concat(
            [
                group["first_line_key"].dropna().astype("string"),
                group["second_line_key"].dropna().astype("string"),
            ],
            ignore_index=True,
        ).nunique()
        latest_departure_sec = float(group["origin_departure_sec"].max()) if group["origin_departure_sec"].notna().any() else np.nan
        latest_direct_departure_sec = (
            float(direct_group["origin_departure_sec"].max()) if not direct_group.empty and direct_group["origin_departure_sec"].notna().any() else np.nan
        )
        best_transfer_wait_min = (
            float(transfer_group["selected_transfer_wait_min"].min())
            if not transfer_group.empty and transfer_group["selected_transfer_wait_min"].notna().any()
            else np.nan
        )
        median_transfer_wait_min = (
            float(transfer_group["selected_transfer_wait_min"].median())
            if not transfer_group.empty and transfer_group["selected_transfer_wait_min"].notna().any()
            else np.nan
        )
        best_transfer_direct_line_count = (
            int(transfer_group["downstream_direct_line_count"].max())
            if not transfer_group.empty and transfer_group["downstream_direct_line_count"].notna().any()
            else 0
        )
        best_transfer_direct_trip_count = (
            int(transfer_group["downstream_direct_trip_count"].max())
            if not transfer_group.empty and transfer_group["downstream_direct_trip_count"].notna().any()
            else 0
        )
        best_transfer_direct_minute_count = (
            int(transfer_group["downstream_direct_minute_count"].max())
            if not transfer_group.empty and transfer_group["downstream_direct_minute_count"].notna().any()
            else 0
        )
        grouped.append(
            {
                "stdt_id": stdt_id,
                "departures_to_work_in_window": group["option_key"].nunique(),
                "departures_to_work_per_hour": group["option_key"].nunique() / window_hours,
                "useful_origin_stop_count": int(useful_origin_stop_count),
                "unique_departure_minutes_in_window": int(unique_departure_minutes),
                "unique_departure_minutes_per_hour": unique_departure_minutes / window_hours,
                "latest_departure_sec": latest_departure_sec,
                "latest_departure_time": seconds_to_hhmm(latest_departure_sec),
                "latest_direct_departure_sec": latest_direct_departure_sec,
                "latest_direct_departure_time": seconds_to_hhmm(latest_direct_departure_sec),
                "best_transfer_wait_min": best_transfer_wait_min,
                "median_transfer_wait_min": median_transfer_wait_min,
                "best_travel_time_min": best["travel_time_min"],
                "best_transfer_count": int(best["transfer_count"]),
                "best_generalized_cost": best["commute_generalized_cost"],
                "direct_line_count": int(direct_line_count),
                "transfer_line_count": int(transfer_line_count),
                "transfer_stop_count": int(transfer_stop_count),
                "useful_line_count_total": int(useful_line_count_total),
                "best_transfer_direct_line_count": int(best_transfer_direct_line_count),
                "best_transfer_direct_trip_count": int(best_transfer_direct_trip_count),
                "best_transfer_direct_minute_count": int(best_transfer_direct_minute_count),
                "best_departure_time": best["departure_hhmm"],
                "best_arrival_time": best["arrival_hhmm"],
                "best_route_path": best.get("route_path"),
                "best_first_route": best["first_route_label"],
                "best_second_route": best["second_route_label"],
                "best_first_route_id": best["first_route_id"],
                "best_second_route_id": best["second_route_id"],
                "best_work_stop_name": best["final_work_stop_name"],
                "best_work_stop_id": best["final_work_stop_id"],
                "best_work_stop_distance_m": best["final_work_stop_distance_m"],
                "best_direct_travel_time_min": direct_group["travel_time_min"].min() if not direct_group.empty else np.nan,
            }
        )

    ranking_df = pd.DataFrame(grouped).merge(rent_df, on="stdt_id", how="left")
    ranking_df["estimated_monthly_rent_typical"] = ranking_df["median_rent_per_m2"] * args.typical_size_m2
    observed_rent = pd.to_numeric(ranking_df["median_rent_per_m2"], errors="coerce")
    observed_travel = pd.to_numeric(ranking_df["best_travel_time_min"], errors="coerce")
    observed_transfer = pd.to_numeric(ranking_df["best_transfer_count"], errors="coerce")
    observed_direct_lines = pd.to_numeric(ranking_df["direct_line_count"], errors="coerce")
    observed_useful_lines = pd.to_numeric(ranking_df["useful_line_count_total"], errors="coerce")
    observed_departure_minutes = pd.to_numeric(ranking_df["unique_departure_minutes_per_hour"], errors="coerce")
    observed_latest_departure = pd.to_numeric(ranking_df["latest_departure_sec"], errors="coerce")
    observed_transfer_direct_lines = pd.to_numeric(ranking_df["best_transfer_direct_line_count"], errors="coerce")
    observed_transfer_direct_minutes = pd.to_numeric(ranking_df["best_transfer_direct_minute_count"], errors="coerce")
    observed_transfer_wait = pd.to_numeric(ranking_df["median_transfer_wait_min"], errors="coerce")
    observed_green_share = pd.to_numeric(ranking_df["green_space_share"], errors="coerce")
    observed_green_patch_density = np.log1p(
        pd.to_numeric(ranking_df["green_space_patch_density_km2"], errors="coerce").fillna(0.0)
    )
    observed_street_tree_density_area = np.log1p(
        pd.to_numeric(ranking_df["street_tree_density_km2"], errors="coerce").fillna(0.0)
    )
    observed_supermarkets = np.log1p(pd.to_numeric(ranking_df["supermarket_count"], errors="coerce").fillna(0.0))
    observed_hospitals = np.log1p(pd.to_numeric(ranking_df["hospital_count"], errors="coerce").fillna(0.0))

    fallback_rent = observed_rent.median() if observed_rent.notna().any() else 0.0
    fallback_travel = (observed_travel.max() * 1.25 + 5.0) if observed_travel.notna().any() else 120.0
    fallback_transfer = max(2.0, float(observed_transfer.max())) if observed_transfer.notna().any() else 2.0
    fallback_latest_departure = observed_latest_departure.min() if observed_latest_departure.notna().any() else 0.0
    fallback_transfer_wait = observed_transfer_wait.max() if observed_transfer_wait.notna().any() else 0.0

    ranking_df["rent_good"] = 1.0 - normalize_series(observed_rent.fillna(fallback_rent))
    ranking_df["travel_good"] = 1.0 - normalize_series(observed_travel.fillna(fallback_travel))
    transfer_count_good = 1.0 - normalize_series(observed_transfer.fillna(fallback_transfer))
    transfer_robustness_good = 0.5 * normalize_series(observed_transfer_direct_lines.fillna(0.0)) + 0.5 * normalize_series(
        observed_transfer_direct_minutes.fillna(0.0)
    )
    transfer_wait_good = 1.0 - normalize_series(observed_transfer_wait.fillna(fallback_transfer_wait))
    ranking_df["transfer_good"] = (transfer_count_good + transfer_robustness_good + transfer_wait_good) / 3.0
    frequency_line_good = normalize_series(observed_direct_lines.fillna(0.0))
    frequency_useful_lines_good = normalize_series(observed_useful_lines.fillna(0.0))
    frequency_departure_minutes_good = normalize_series(observed_departure_minutes.fillna(0.0))
    frequency_latest_good = normalize_series(observed_latest_departure.fillna(fallback_latest_departure))
    ranking_df["frequency_good"] = (
        frequency_line_good
        + frequency_useful_lines_good
        + frequency_departure_minutes_good
        + frequency_latest_good
    ) / 4.0
    green_share_good = normalize_series(observed_green_share.fillna(0.0))
    green_patch_density_good = normalize_series(observed_green_patch_density)
    street_tree_density_area_good = normalize_series(observed_street_tree_density_area)
    ranking_df["green_space_good"] = (
        green_share_good
        + green_patch_density_good
        + street_tree_density_area_good
    ) / 3.0
    ranking_df["supermarket_good"] = normalize_series(observed_supermarkets)
    ranking_df["hospital_good"] = normalize_series(observed_hospitals)

    weighted_good_sum = (
        ranking_df["rent_good"] * preferences["rent"]
        + ranking_df["travel_good"] * preferences["time"]
        + ranking_df["frequency_good"] * preferences["frequency"]
        + ranking_df["transfer_good"] * preferences["transfer"]
        + ranking_df["green_space_good"] * preferences["green_space"]
        + ranking_df["supermarket_good"] * preferences["supermarket"]
        + ranking_df["hospital_good"] * preferences["hospital"]
    )
    total_weight = max(1.0, float(sum(preferences.values())))
    component_specs = [
        ("rent", "rent_good"),
        ("time", "travel_good"),
        ("frequency", "frequency_good"),
        ("transfer", "transfer_good"),
        ("green_space", "green_space_good"),
        ("supermarket", "supermarket_good"),
        ("hospital", "hospital_good"),
    ]
    for component_name, good_col in component_specs:
        ranking_df[f"score_{component_name}_contribution"] = (
            100.0 * ranking_df[good_col] * preferences[component_name] / total_weight
        )

    def build_score_summary(row: pd.Series) -> str:
        if bool(row.get("rent_data_missing", False)):
            return "Rent data missing; commute and amenity metrics are available, but the area is not ranked."
        active_components = [
            (
                component_labels[name],
                float(row[f"score_{name}_contribution"]),
                float(row[good_col]),
            )
            for name, good_col in component_specs
            if preferences[name] > 0
        ]
        if not active_components:
            return "All user preferences are set to zero."
        strengths = sorted(active_components, key=lambda item: item[1], reverse=True)[:2]
        weakest = min(active_components, key=lambda item: item[2])
        if len(strengths) == 1:
            return f"Main strength: {strengths[0][0]}. Weakest side: {weakest[0]}."
        return f"Main strengths: {strengths[0][0]} and {strengths[1][0]}. Weakest side: {weakest[0]}."

    ranking_df["ranking_score"] = 100.0 * weighted_good_sum / total_weight
    ranking_df["rent_data_missing"] = observed_rent.isna()
    ranking_df.loc[ranking_df["rent_data_missing"], "ranking_score"] = np.nan
    for component_name, _ in component_specs:
        ranking_df.loc[ranking_df["rent_data_missing"], f"score_{component_name}_contribution"] = np.nan
    ranking_df["score_summary"] = ranking_df.apply(build_score_summary, axis=1)
    ranking_df = ranking_df.sort_values(
        ["ranking_score", "best_travel_time_min", "median_rent_per_m2"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking_df["district_rank"] = np.where(
        ranking_df["ranking_score"].notna(),
        np.arange(1, len(ranking_df) + 1),
        np.nan,
    )
    valid_rank_mask = ranking_df["ranking_score"].notna()
    if valid_rank_mask.any():
        ranking_df.loc[valid_rank_mask, "district_rank"] = np.arange(1, int(valid_rank_mask.sum()) + 1)
    ranking_df["desirability_index"] = ranking_df["ranking_score"]

    ranking_df["preference_reason"] = ranking_df.apply(
        lambda row: (
            f"{'Direct' if row['best_transfer_count'] == 0 else '1-transfer'} trip, "
            f"{row['best_travel_time_min']:.1f} min, "
            f"{row['unique_departure_minutes_per_hour']:.2f} unique departure minutes/hour, "
            f"{int(row['useful_line_count_total'])} useful line groups, "
            f"latest departure {row['latest_departure_time'] if pd.notna(row['latest_departure_time']) else 'n/a'}, "
            f"rent {row['median_rent_per_m2']:.2f} EUR/m², "
            f"green share {100.0 * float(pd.to_numeric(row['green_space_share'], errors='coerce') if pd.notna(pd.to_numeric(row['green_space_share'], errors='coerce')) else 0.0):.1f}%, "
            f"green patch density {float(pd.to_numeric(row['green_space_patch_density_km2'], errors='coerce') if pd.notna(pd.to_numeric(row['green_space_patch_density_km2'], errors='coerce')) else 0.0):.2f}/km², "
            f"street tree density {float(pd.to_numeric(row['street_tree_density_km2'], errors='coerce') if pd.notna(pd.to_numeric(row['street_tree_density_km2'], errors='coerce')) else 0.0):.2f}/km²"
            if pd.notna(row["median_rent_per_m2"])
            else f"{'Direct' if row['best_transfer_count'] == 0 else '1-transfer'} trip, "
            f"{row['best_travel_time_min']:.1f} min, "
            f"{row['unique_departure_minutes_per_hour']:.2f} unique departure minutes/hour, "
            f"{int(row['useful_line_count_total'])} useful line groups, "
            f"latest departure {row['latest_departure_time'] if pd.notna(row['latest_departure_time']) else 'n/a'}"
        ),
        axis=1,
    )

    ranking_df.to_csv(PROCESSED_DIR / "commute_ranking_by_workplace_stdt.csv", index=False, encoding="utf-8")
    return ranking_df, options_df


def build_target_direct_layer_data(
    destination_occurrences: pd.DataFrame,
    district_stops_gdf: gpd.GeoDataFrame,
    assigned_stops: pd.DataFrame,
    window_hours: float,
    relevant_stop_times: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    empty_segments = gpd.GeoDataFrame(
        columns=[
            "route_id",
            "route_label",
            "line_key",
            "route_type",
            "trip_id",
            "target_stop_id",
            "target_stop_name",
            "arrivals_in_window",
            "arrivals_per_hour",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    empty_stops = pd.DataFrame(
        columns=[
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "stdt_id",
            "best_direct_frequency_per_hour",
            "best_time_to_target_min",
            "direct_routes",
        ]
    )
    if destination_occurrences.empty:
        return empty_segments, empty_stops

    full_occurrence_times = relevant_stop_times.loc[
        relevant_stop_times["trip_id"].isin(destination_occurrences["trip_id"].astype("string"))
    ].copy()
    if full_occurrence_times.empty:
        return empty_segments, empty_stops

    district_stop_lookup = district_stops_gdf.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon"]].to_dict("index")
    district_stop_ids = set(district_stops_gdf["stop_id"].astype(str))

    line_frequency = (
        destination_occurrences.assign(trip_id=destination_occurrences["trip_id"].astype("string"))
        .groupby(["line_key", "route_label", "route_type"], dropna=False)["trip_id"]
        .nunique()
        .reset_index(name="arrivals_in_window")
    )
    line_frequency["arrivals_per_hour"] = line_frequency["arrivals_in_window"] / max(window_hours, 1e-9)

    candidates = destination_occurrences.copy()
    candidates["trip_id"] = candidates["trip_id"].astype("string")
    candidates = candidates.sort_values(
        ["trip_id", "distance_to_work_m", "arrival_sec"],
        ascending=[True, True, True],
    )
    trip_occurrences = candidates.drop_duplicates(subset=["trip_id"], keep="first").copy()
    trip_occurrences = trip_occurrences.merge(line_frequency, on=["line_key", "route_label", "route_type"], how="left")

    rep_lookup = trip_occurrences.set_index("trip_id").to_dict("index")
    route_segments: list[dict[str, Any]] = []
    stop_summaries: dict[str, dict[str, Any]] = {}

    for trip_id, group in full_occurrence_times.loc[
        full_occurrence_times["trip_id"].isin(trip_occurrences["trip_id"].tolist())
    ].groupby("trip_id"):
        meta = rep_lookup.get(str(trip_id))
        if meta is None:
            continue
        upto_target = group.loc[group["stop_sequence"] <= meta["dest_sequence"]].sort_values("stop_sequence").copy()
        if upto_target.empty:
            continue

        prev_row = None
        for row in upto_target.itertuples(index=False):
            current_stop = district_stop_lookup.get(str(row.stop_id))
            if current_stop is None:
                prev_row = row
                continue

            if row.departure_sec is not None and meta["arrival_sec"] >= row.departure_sec:
                time_to_target_min = (float(meta["arrival_sec"]) - float(row.departure_sec)) / 60.0
            else:
                time_to_target_min = 0.0 if str(row.stop_id) == str(meta["work_stop_id"]) else np.nan

            summary = stop_summaries.setdefault(
                str(row.stop_id),
                {
                    "stop_id": str(row.stop_id),
                    "stop_name": current_stop["stop_name"],
                    "stop_lat": float(current_stop["stop_lat"]),
                    "stop_lon": float(current_stop["stop_lon"]),
                    "services": [],
                },
            )
            summary["services"].append(
                {
                    "route_label": str(meta["route_label"]),
                    "target_stop_name": str(meta["work_stop_name"]),
                    "time_to_target_min": float(time_to_target_min) if pd.notna(time_to_target_min) else np.nan,
                    "arrivals_per_hour": float(meta["arrivals_per_hour"]),
                }
            )

            if prev_row is not None:
                prev_stop = district_stop_lookup.get(str(prev_row.stop_id))
                if prev_stop is not None:
                    color, weight = route_color(meta["route_id"], meta["route_type"])
                    route_segments.append(
                        {
                            "route_id": meta["route_id"],
                            "route_label": meta["route_label"],
                            "line_key": meta["line_key"],
                            "route_type": meta["route_type"],
                            "trip_id": str(trip_id),
                            "target_stop_id": meta["work_stop_id"],
                            "target_stop_name": meta["work_stop_name"],
                            "arrivals_in_window": int(meta["arrivals_in_window"]),
                            "arrivals_per_hour": float(meta["arrivals_per_hour"]),
                            "from_stop_id": str(prev_row.stop_id),
                            "from_stop_name": prev_stop["stop_name"],
                            "to_stop_id": str(row.stop_id),
                            "to_stop_name": current_stop["stop_name"],
                            "route_path_summary": f"{meta['route_label']} direct to {meta['work_stop_name']}",
                            "geometry": LineString(
                                [
                                    (float(prev_stop["stop_lon"]), float(prev_stop["stop_lat"])),
                                    (float(current_stop["stop_lon"]), float(current_stop["stop_lat"])),
                                ]
                            ),
                            "color": color,
                            "weight": weight + 1,
                        }
                    )
            prev_row = row

    direct_segments = gpd.GeoDataFrame(route_segments, geometry="geometry", crs="EPSG:4326")
    if direct_segments.empty:
        return empty_segments, empty_stops
    direct_segments = direct_segments.drop_duplicates(
        subset=["line_key", "from_stop_id", "to_stop_id", "target_stop_id"]
    ).reset_index(drop=True)

    stdt_lookup = (
        assigned_stops[["stop_id", "stdt_id"]]
        .drop_duplicates(subset=["stop_id"])
        .assign(stop_id=lambda frame: frame["stop_id"].astype("string"))
        .set_index("stop_id")["stdt_id"]
        .to_dict()
    )

    stop_rows = []
    for stop_id, payload in stop_summaries.items():
        services = sorted(payload["services"], key=lambda item: (item["time_to_target_min"], item["route_label"]))
        stop_rows.append(
            {
                "stop_id": stop_id,
                "stop_name": payload["stop_name"],
                "stop_lat": payload["stop_lat"],
                "stop_lon": payload["stop_lon"],
                "stdt_id": stdt_lookup.get(stop_id),
                "best_direct_frequency_per_hour": max(service["arrivals_per_hour"] for service in services),
                "best_time_to_target_min": min(
                    service["time_to_target_min"] for service in services if pd.notna(service["time_to_target_min"])
                )
                if any(pd.notna(service["time_to_target_min"]) for service in services)
                else np.nan,
                "direct_routes": ", ".join(sorted({service["route_label"] for service in services})[:8]),
            }
        )

    direct_stops = pd.DataFrame(stop_rows).sort_values(["best_time_to_target_min", "stop_name"], ascending=[True, True])
    return direct_segments, direct_stops.reset_index(drop=True)


def build_used_feeder_layer_data(
    options_df: pd.DataFrame,
    district_stops_gdf: gpd.GeoDataFrame,
    relevant_stop_times: pd.DataFrame,
    transfer_radius_m: float,
    window_hours: float,
    direct_target_line_keys: set[str],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    empty_segments = gpd.GeoDataFrame(
        columns=[
            "route_id",
            "route_label",
            "trip_id",
            "direct_transfer_stop_id",
            "feeder_alight_stop_id",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    empty_markers = pd.DataFrame(
        columns=[
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "feasible_departures_per_hour",
            "best_total_time_to_target_min",
            "best_transfer_count",
            "route_label",
            "feeder_route_label",
            "marker_role",
            "radius_m",
        ]
    )

    transfer_options = options_df.loc[
        options_df["transfer_count"] > 0
        & options_df["final_feeder_trip_id"].notna()
        & options_df["feeder_alight_stop_id"].notna()
        & options_df["direct_transfer_stop_id"].notna()
    ].copy()
    if direct_target_line_keys:
        transfer_options = transfer_options.loc[
            ~transfer_options["final_feeder_line_key"].astype("string").isin(direct_target_line_keys)
        ].copy()
    if transfer_options.empty:
        return empty_segments, empty_markers

    district_stop_lookup = district_stops_gdf.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon"]].to_dict("index")
    district_stop_ids = set(district_stops_gdf["stop_id"].astype(str))

    representative_cols = [
        "final_feeder_trip_id",
        "final_feeder_route_id",
        "final_feeder_route_label",
        "final_feeder_line_key",
        "feeder_alight_stop_id",
        "direct_transfer_stop_id",
    ]
    representatives = (
        transfer_options.groupby(representative_cols, dropna=False)
        .agg(
            departures_in_window=("option_key", "nunique"),
            best_total_time_to_target_min=("travel_time_min", "min"),
            best_transfer_count=("transfer_count", "min"),
        )
        .reset_index()
    )

    stop_times = relevant_stop_times.loc[
        relevant_stop_times["trip_id"].isin(representatives["final_feeder_trip_id"].astype("string"))
    ].copy()
    if stop_times.empty:
        return empty_segments, empty_markers

    segment_rows: list[dict[str, Any]] = []
    transfer_marker_rows: list[dict[str, Any]] = []
    feeder_stop_rows: list[dict[str, Any]] = []

    for rep in representatives.itertuples(index=False):
        trip_id = str(rep.final_feeder_trip_id)
        trip_df = stop_times.loc[stop_times["trip_id"] == trip_id].sort_values("stop_sequence").copy()
        if trip_df.empty:
            continue
        alight_matches = trip_df.index[trip_df["stop_id"].astype(str) == str(rep.feeder_alight_stop_id)].tolist()
        if not alight_matches:
            continue
        alight_idx = alight_matches[0]
        upto_transfer = trip_df.loc[:alight_idx].copy()
        upto_transfer = upto_transfer.loc[upto_transfer["stop_id"].astype(str).isin(district_stop_ids)].copy()
        if len(upto_transfer) < 2:
            continue

        route_type_hint = 3
        final_line_key = str(getattr(rep, "final_feeder_line_key", "") or "")
        if "|" in final_line_key:
            route_type_hint = final_line_key.split("|", 1)[0]
        color, weight = route_color(rep.final_feeder_route_id, route_type_hint)
        prev_row = None
        representative_origin_name = None
        for row in upto_transfer.itertuples(index=False):
            current_stop = district_stop_lookup.get(str(row.stop_id))
            if current_stop is None:
                prev_row = row
                continue
            if representative_origin_name is None:
                representative_origin_name = current_stop["stop_name"]
            feeder_stop_rows.append(
                {
                    "stop_id": str(row.stop_id),
                    "stop_name": current_stop["stop_name"],
                    "stop_lat": float(current_stop["stop_lat"]),
                    "stop_lon": float(current_stop["stop_lon"]),
                    "route_label": rep.final_feeder_route_label,
                    "best_total_time_to_target_min": float(rep.best_total_time_to_target_min),
                    "feasible_departures_per_hour": float(rep.departures_in_window) / max(window_hours, 1e-9),
                    "transfer_stop_id": str(rep.direct_transfer_stop_id),
                }
            )
            if prev_row is not None:
                prev_stop = district_stop_lookup.get(str(prev_row.stop_id))
                if prev_stop is not None:
                    segment_rows.append(
                        {
                            "route_id": rep.final_feeder_route_id,
                            "route_label": rep.final_feeder_route_label,
                            "trip_id": trip_id,
                            "direct_transfer_stop_id": str(rep.direct_transfer_stop_id),
                            "feeder_alight_stop_id": str(rep.feeder_alight_stop_id),
                            "from_stop_id": str(prev_row.stop_id),
                            "from_stop_name": prev_stop["stop_name"],
                            "to_stop_id": str(row.stop_id),
                            "to_stop_name": current_stop["stop_name"],
                            "origin_stop_name": representative_origin_name or prev_stop["stop_name"],
                            "transfer_stop_name": district_stop_lookup.get(str(rep.direct_transfer_stop_id), {}).get("stop_name"),
                            "target_stop_name": None,
                            "route_path_summary": f"{rep.final_feeder_route_label} -> transfer",
                            "geometry": LineString(
                                [
                                    (float(prev_stop["stop_lon"]), float(prev_stop["stop_lat"])),
                                    (float(current_stop["stop_lon"]), float(current_stop["stop_lat"])),
                                ]
                            ),
                                "color": color,
                                "weight": max(2, weight - 1),
                        }
                    )
            prev_row = row

        direct_transfer_stop = district_stop_lookup.get(str(rep.direct_transfer_stop_id))
        if direct_transfer_stop is not None:
            transfer_marker_rows.append(
                {
                    "stop_id": str(rep.direct_transfer_stop_id),
                    "stop_name": direct_transfer_stop["stop_name"],
                    "stop_lat": float(direct_transfer_stop["stop_lat"]),
                    "stop_lon": float(direct_transfer_stop["stop_lon"]),
                    "feasible_departures_per_hour": float(rep.departures_in_window) / max(window_hours, 1e-9),
                    "best_total_time_to_target_min": float(rep.best_total_time_to_target_min),
                    "best_transfer_count": int(rep.best_transfer_count),
                    "feeder_route_label": rep.final_feeder_route_label,
                    "radius_m": float(transfer_radius_m),
                }
            )

    feeder_segments = gpd.GeoDataFrame(segment_rows, geometry="geometry", crs="EPSG:4326")
    transfer_markers = pd.DataFrame(transfer_marker_rows)
    feeder_stop_markers = pd.DataFrame(feeder_stop_rows)
    if transfer_markers.empty:
        return feeder_segments, empty_markers

    transfer_markers = (
        transfer_markers.groupby(["stop_id", "stop_name", "stop_lat", "stop_lon", "radius_m"], dropna=False)
        .agg(
            feasible_departures_per_hour=("feasible_departures_per_hour", "sum"),
            best_total_time_to_target_min=("best_total_time_to_target_min", "min"),
            best_transfer_count=("best_transfer_count", "min"),
            feeder_route_label=("feeder_route_label", lambda values: ", ".join(sorted({str(v) for v in values if pd.notna(v)})[:8])),
        )
        .reset_index()
        .sort_values(["feasible_departures_per_hour", "best_total_time_to_target_min"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if feeder_stop_markers.empty:
        feeder_stop_markers = empty_markers.copy()
    else:
        feeder_stop_markers = (
            feeder_stop_markers.groupby(["stop_id", "stop_name", "stop_lat", "stop_lon"], dropna=False)
            .agg(
                feasible_departures_per_hour=("feasible_departures_per_hour", "sum"),
                best_total_time_to_target_min=("best_total_time_to_target_min", "min"),
                route_label=("route_label", lambda values: ", ".join(sorted({str(v) for v in values if pd.notna(v)})[:8])),
            )
            .reset_index()
            .sort_values(["best_total_time_to_target_min", "stop_name"], ascending=[True, True])
            .reset_index(drop=True)
        )

    transfer_markers["marker_role"] = "transfer_stop"
    feeder_stop_markers["marker_role"] = "feeder_stop"
    feeder_stop_markers["radius_m"] = np.nan
    all_markers = pd.concat([feeder_stop_markers, transfer_markers], ignore_index=True, sort=False)
    all_markers = all_markers.drop_duplicates(subset=["stop_id", "marker_role"]).reset_index(drop=True)
    return feeder_segments, all_markers


def build_earlier_feeder_layer_data(
    options_df: pd.DataFrame,
    district_stops_gdf: gpd.GeoDataFrame,
    relevant_stop_times: pd.DataFrame,
    transfer_radius_m: float,
    window_hours: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    empty_segments = gpd.GeoDataFrame(
        columns=[
            "route_id",
            "route_label",
            "trip_id",
            "alight_stop_id",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    empty_markers = pd.DataFrame(
        columns=[
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "feasible_departures_per_hour",
            "best_total_time_to_target_min",
            "best_transfer_count",
            "feeder_route_label",
            "radius_m",
        ]
    )

    options = options_df.loc[
        options_df["transfer_count"] >= 2
        & options_df["feeder_trip_chain_ids"].notna()
        & options_df["feeder_alight_stop_chain_ids"].notna()
    ].copy()
    if options.empty:
        return empty_segments, empty_markers

    district_stop_lookup = district_stops_gdf.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon"]].to_dict("index")
    district_stop_ids = set(district_stops_gdf["stop_id"].astype(str))

    segment_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []

    for row in options.itertuples(index=False):
        trip_ids = [value for value in str(row.feeder_trip_chain_ids or "").split("|") if value]
        route_ids = [value for value in str(row.feeder_route_chain_ids or "").split("|") if value]
        route_labels = [value for value in str(row.feeder_route_chain_labels or "").split("|") if value]
        alight_stop_ids = [value for value in str(row.feeder_alight_stop_chain_ids or "").split("|") if value]
        if len(trip_ids) < 2 or len(alight_stop_ids) < 2:
            continue

        for idx in range(len(trip_ids) - 1):
            trip_id = str(trip_ids[idx])
            alight_stop_id = str(alight_stop_ids[idx])
            route_id = str(route_ids[idx]) if idx < len(route_ids) else None
            route_label = str(route_labels[idx]) if idx < len(route_labels) else None

            trip_df = relevant_stop_times.loc[relevant_stop_times["trip_id"] == trip_id].sort_values("stop_sequence").copy()
            if trip_df.empty:
                continue
            alight_matches = trip_df.index[trip_df["stop_id"].astype(str) == alight_stop_id].tolist()
            if not alight_matches:
                continue
            alight_idx = alight_matches[0]
            upto_transfer = trip_df.loc[:alight_idx].copy()
            upto_transfer = upto_transfer.loc[upto_transfer["stop_id"].astype(str).isin(district_stop_ids)].copy()
            if len(upto_transfer) < 2:
                continue

            prev_trip_row = None
            for trip_row in upto_transfer.itertuples(index=False):
                current_stop = district_stop_lookup.get(str(trip_row.stop_id))
                if current_stop is None:
                    prev_trip_row = trip_row
                    continue
                if prev_trip_row is not None:
                    prev_stop = district_stop_lookup.get(str(prev_trip_row.stop_id))
                    if prev_stop is not None:
                        color, weight = route_color(route_id, 3)
                        segment_rows.append(
                            {
                                "route_id": route_id,
                                "route_label": route_label,
                                "trip_id": trip_id,
                                "alight_stop_id": alight_stop_id,
                                "from_stop_id": str(prev_trip_row.stop_id),
                                "from_stop_name": prev_stop["stop_name"],
                                "to_stop_id": str(trip_row.stop_id),
                                "to_stop_name": current_stop["stop_name"],
                                "geometry": LineString(
                                    [
                                        (float(prev_stop["stop_lon"]), float(prev_stop["stop_lat"])),
                                        (float(current_stop["stop_lon"]), float(current_stop["stop_lat"])),
                                    ]
                                ),
                                "color": color,
                                "weight": max(2, weight - 1),
                            }
                        )
                prev_trip_row = trip_row

            transfer_stop = district_stop_lookup.get(alight_stop_id)
            if transfer_stop is not None:
                marker_rows.append(
                    {
                        "stop_id": alight_stop_id,
                        "stop_name": transfer_stop["stop_name"],
                        "stop_lat": float(transfer_stop["stop_lat"]),
                        "stop_lon": float(transfer_stop["stop_lon"]),
                        "feasible_departures_per_hour": 1.0 / max(window_hours, 1e-9),
                        "best_total_time_to_target_min": float(row.travel_time_min),
                        "best_transfer_count": int(row.transfer_count),
                        "feeder_route_label": route_label,
                        "radius_m": float(transfer_radius_m),
                    }
                )

    if segment_rows:
        earlier_segments = gpd.GeoDataFrame(segment_rows, geometry="geometry", crs="EPSG:4326")
    else:
        earlier_segments = empty_segments.copy()
    earlier_markers = pd.DataFrame(marker_rows)
    if earlier_markers.empty:
        return earlier_segments, empty_markers

    earlier_markers = (
        earlier_markers.groupby(["stop_id", "stop_name", "stop_lat", "stop_lon", "radius_m"], dropna=False)
        .agg(
            feasible_departures_per_hour=("feasible_departures_per_hour", "sum"),
            best_total_time_to_target_min=("best_total_time_to_target_min", "min"),
            best_transfer_count=("best_transfer_count", "min"),
            feeder_route_label=("feeder_route_label", lambda values: ", ".join(sorted({str(v) for v in values if pd.notna(v)})[:8])),
        )
        .reset_index()
        .sort_values(["feasible_departures_per_hour", "best_total_time_to_target_min"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return earlier_segments, earlier_markers


def build_district_heatmap_frame(
    district_stdt_gdf: gpd.GeoDataFrame,
    ranking_df: pd.DataFrame,
    args: argparse.Namespace,
) -> gpd.GeoDataFrame:
    preferences = resolve_preferences(args)
    merged = district_stdt_gdf.merge(ranking_df, on="stdt_id", how="left", suffixes=("", "_rank"))

    merged["commute_status"] = np.where(
        merged["best_travel_time_min"].notna(),
        "feasible_commute_found",
        "no_feasible_commute_found",
    )
    merged["rent_data_status"] = np.where(
        pd.to_numeric(merged["median_rent_per_m2"], errors="coerce").notna(),
        "rent_data_available",
        "rent_data_missing",
    )
    merged["rent_data_label"] = np.where(
        merged["rent_data_status"].eq("rent_data_available"),
        "Available",
        "Missing",
    )

    observed_rent = pd.to_numeric(merged["median_rent_per_m2"], errors="coerce")
    observed_travel = pd.to_numeric(merged["best_travel_time_min"], errors="coerce")
    observed_transfer = pd.to_numeric(merged["best_transfer_count"], errors="coerce")
    observed_direct_lines = pd.to_numeric(merged["direct_line_count"], errors="coerce")
    observed_useful_lines = pd.to_numeric(merged["useful_line_count_total"], errors="coerce")
    observed_departure_minutes = pd.to_numeric(merged["unique_departure_minutes_per_hour"], errors="coerce")
    observed_latest_departure = pd.to_numeric(merged["latest_departure_sec"], errors="coerce")
    observed_transfer_direct_lines = pd.to_numeric(merged["best_transfer_direct_line_count"], errors="coerce")
    observed_transfer_direct_minutes = pd.to_numeric(merged["best_transfer_direct_minute_count"], errors="coerce")
    observed_transfer_wait = pd.to_numeric(merged["median_transfer_wait_min"], errors="coerce")
    observed_green_share = pd.to_numeric(merged["green_space_share"], errors="coerce")
    observed_green_patch_density = np.log1p(
        pd.to_numeric(merged["green_space_patch_density_km2"], errors="coerce").fillna(0.0)
    )
    observed_street_tree_density_area = np.log1p(
        pd.to_numeric(merged["street_tree_density_km2"], errors="coerce").fillna(0.0)
    )
    observed_supermarkets = pd.to_numeric(merged["supermarket_count"], errors="coerce")
    observed_supermarkets = pd.to_numeric(merged["supermarket_count"], errors="coerce")
    observed_hospitals = pd.to_numeric(merged["hospital_count"], errors="coerce")

    fallback_rent = observed_rent.median() if observed_rent.notna().any() else 0.0
    fallback_travel = (observed_travel.max() * 1.25 + 5.0) if observed_travel.notna().any() else 120.0
    fallback_transfer = max(2.0, float(observed_transfer.max())) if observed_transfer.notna().any() else 2.0

    merged["rent_for_scale"] = observed_rent.fillna(fallback_rent)
    merged["travel_for_scale"] = observed_travel.fillna(fallback_travel)
    merged["transfer_for_scale"] = observed_transfer.fillna(fallback_transfer)
    merged["direct_lines_for_scale"] = observed_direct_lines.fillna(0.0)
    merged["useful_lines_for_scale"] = observed_useful_lines.fillna(0.0)
    merged["departure_minutes_for_scale"] = observed_departure_minutes.fillna(0.0)
    merged["latest_departure_for_scale"] = observed_latest_departure.fillna(observed_latest_departure.min() if observed_latest_departure.notna().any() else 0.0)
    merged["transfer_direct_lines_for_scale"] = observed_transfer_direct_lines.fillna(0.0)
    merged["transfer_direct_minutes_for_scale"] = observed_transfer_direct_minutes.fillna(0.0)
    merged["transfer_wait_for_scale"] = observed_transfer_wait.fillna(observed_transfer_wait.max() if observed_transfer_wait.notna().any() else 0.0)
    merged["green_share_for_scale"] = observed_green_share.fillna(0.0)
    merged["green_patch_density_for_scale"] = observed_green_patch_density
    merged["street_tree_density_area_for_scale"] = observed_street_tree_density_area
    merged["supermarkets_for_scale"] = observed_supermarkets.fillna(0.0)
    merged["hospitals_for_scale"] = observed_hospitals.fillna(0.0)

    rent_good = 1.0 - normalize_series(merged["rent_for_scale"])
    travel_good = 1.0 - normalize_series(merged["travel_for_scale"])
    transfer_count_good = 1.0 - normalize_series(merged["transfer_for_scale"])
    frequency_line_good = normalize_series(merged["direct_lines_for_scale"])
    frequency_useful_lines_good = normalize_series(merged["useful_lines_for_scale"])
    frequency_departure_minutes_good = normalize_series(merged["departure_minutes_for_scale"])
    frequency_latest_good = normalize_series(merged["latest_departure_for_scale"])
    transfer_direct_lines_good = normalize_series(merged["transfer_direct_lines_for_scale"])
    transfer_direct_minutes_good = normalize_series(merged["transfer_direct_minutes_for_scale"])
    transfer_wait_good = 1.0 - normalize_series(merged["transfer_wait_for_scale"])
    green_share_good = normalize_series(merged["green_share_for_scale"])
    green_patch_density_good = normalize_series(merged["green_patch_density_for_scale"])
    street_tree_density_area_good = normalize_series(merged["street_tree_density_area_for_scale"])
    supermarkets_good = normalize_series(np.log1p(merged["supermarkets_for_scale"]))
    hospitals_good = normalize_series(np.log1p(merged["hospitals_for_scale"]))

    frequency_good = (
        frequency_line_good
        + frequency_useful_lines_good
        + frequency_departure_minutes_good
        + frequency_latest_good
    ) / 4.0
    transfer_good = (
        transfer_count_good + transfer_direct_lines_good + transfer_direct_minutes_good + transfer_wait_good
    ) / 4.0
    green_good = (
        green_share_good
        + green_patch_density_good
        + street_tree_density_area_good
    ) / 3.0
    total_weight = max(1.0, float(sum(preferences.values())))
    merged["heatmap_score"] = 100.0 * (
        rent_good * preferences["rent"]
        + travel_good * preferences["time"]
        + frequency_good * preferences["frequency"]
        + transfer_good * preferences["transfer"]
        + green_good * preferences["green_space"]
        + supermarkets_good * preferences["supermarket"]
        + hospitals_good * preferences["hospital"]
    ) / total_weight
    merged.loc[observed_rent.isna(), "heatmap_score"] = np.nan

    merged["heatmap_label"] = merged.apply(
        lambda row: (
            f"Rent data missing; feasible commute, {row['best_travel_time_min']:.1f} min, "
            f"{row['useful_line_count_total']:.0f} useful line groups"
            if pd.isna(row["median_rent_per_m2"]) and pd.notna(row["best_travel_time_min"])
            else (
                f"Feasible commute, {row['best_travel_time_min']:.1f} min, "
                f"{row['useful_line_count_total']:.0f} useful line groups, rent {row['median_rent_per_m2']:.2f} EUR/m²"
                if pd.notna(row["best_travel_time_min"]) and pd.notna(row["median_rent_per_m2"])
                else (
                    "Rent data missing and no feasible commute found"
                    if pd.isna(row["median_rent_per_m2"])
                    else f"No feasible commute found, rent {row['median_rent_per_m2']:.2f} EUR/m²"
                )
            )
        ),
        axis=1,
    )
    return merged


def create_interactive_map(
    district: gpd.GeoDataFrame,
    district_heatmap_gdf: gpd.GeoDataFrame,
    target_point: gpd.GeoDataFrame,
    target_stops: pd.DataFrame,
    target_radius_m: float,
    direct_target_segments: gpd.GeoDataFrame,
    direct_target_stops: pd.DataFrame,
    transfer_segments: gpd.GeoDataFrame,
    transfer_markers: pd.DataFrame,
    earlier_feeder_segments: gpd.GeoDataFrame,
    earlier_feeder_markers: pd.DataFrame,
    output_path: Path,
) -> None:
    district_wgs84 = district.to_crs("EPSG:4326")
    ranking_wgs84 = district_heatmap_gdf.to_crs("EPSG:4326")
    bounds = district_wgs84.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    valid_scores = ranking_wgs84["heatmap_score"].dropna()
    if not valid_scores.empty:
        low_clip = float(valid_scores.quantile(0.10))
        high_clip = float(valid_scores.quantile(0.90))
        if high_clip <= low_clip:
            low_clip = float(valid_scores.min())
            high_clip = float(valid_scores.max())
        if high_clip <= low_clip:
            high_clip = low_clip + 1.0
        colormap = linear.RdYlGn_09.scale(low_clip, high_clip)
    else:
        colormap = linear.RdYlGn_09.scale(0.0, 100.0)
    colormap.caption = "District suitability score (green is better, gray = no rent data)"

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        value = feature["properties"].get("heatmap_score")
        rent_status = feature["properties"].get("rent_data_status")
        if rent_status == "rent_data_missing":
            return {"fillColor": "#bdbdbd", "color": "#6b6b6b", "weight": 1.0, "fillOpacity": 0.55}
        if value is None:
            return {"fillColor": "#d9d9d9", "color": "#5f5f5f", "weight": 1.0, "fillOpacity": 0.42}
        clipped_value = min(max(float(value), colormap.vmin), colormap.vmax)
        return {"fillColor": colormap(clipped_value), "color": "#4a4a4a", "weight": 0.9, "fillOpacity": 0.72}

    heat_layer = folium.FeatureGroup(name="District Stadtteil heatmap", show=True).add_to(fmap)
    folium.GeoJson(
        ranking_wgs84.to_json(),
        style_function=style_function,
        highlight_function=lambda _: {"weight": 2.0, "color": "#111111", "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "stdt_name",
                "municipality_name",
                "rent_data_label",
                "heatmap_score",
                "score_summary",
                "median_rent_per_m2",
                "green_space_share",
                "green_space_patch_density_km2",
                "street_tree_density_km2",
                "supermarket_count",
                "hospital_count",
                "best_travel_time_min",
                "departures_to_work_per_hour",
                "useful_line_count_total",
                "unique_departure_minutes_per_hour",
                "latest_departure_time",
                "best_transfer_count",
                "best_transfer_direct_line_count",
            ],
            aliases=[
                "Stadtteil",
                "Municipality",
                "Rent data",
                "District suitability score",
                "Why this area scores this way",
                "Median rent €/m²",
                "Green space share",
                "Green patch density / km²",
                "Street tree density / km²",
                "Supermarket count",
                "Hospital count",
                "Best travel time (min)",
                "Feasible departures/hour",
                "Useful line groups",
                "Unique departure minutes/hour",
                "Latest feasible departure",
                "Transfers",
                "Transfer direct lines",
            ],
            localize=True,
            sticky=False,
        ),
    ).add_to(heat_layer)
    colormap.add_to(fmap)

    location_layer = folium.FeatureGroup(name="Target location and nearest stops", show=True).add_to(fmap)
    work_row = target_point.iloc[0]
    folium.Marker(
        location=[work_row.geometry.y, work_row.geometry.x],
        tooltip="Target location",
        icon=folium.Icon(color="red", icon="briefcase", prefix="fa"),
    ).add_to(location_layer)
    folium.Circle(
        location=[work_row.geometry.y, work_row.geometry.x],
        radius=float(target_radius_m),
        color="#cb181d",
        weight=1.5,
        fill=False,
        opacity=0.7,
        tooltip=f"Target stop search radius: {float(target_radius_m):.0f} m",
    ).add_to(location_layer)
    for row in target_stops.itertuples(index=False):
        folium.CircleMarker(
            location=[row.stop_lat, row.stop_lon],
            radius=6,
            color="#7f0000",
            fill=True,
            fill_color="#ef3b2c",
            fill_opacity=0.9,
            tooltip=f"{row.stop_name} | {row.distance_to_work_m:.0f} m to target",
        ).add_to(location_layer)

    direct_layer = folium.FeatureGroup(name="Direct target services", show=False).add_to(fmap)
    for row in direct_target_segments.itertuples(index=False):
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in row.geometry.coords],
            color=row.color,
            weight=row.weight,
            opacity=0.8,
            tooltip=(
                f"{row.route_label} | direct to {row.target_stop_name} | "
                f"{row.arrivals_per_hour:.2f}/hour"
            ),
            popup=(
                f"Direct route: {row.route_label}<br>"
                f"To target stop: {row.target_stop_name}<br>"
                f"Frequency in selected window: {row.arrivals_per_hour:.2f}/hour"
            ),
        ).add_to(direct_layer)
    for row in direct_target_stops.itertuples(index=False):
        folium.CircleMarker(
            location=[row.stop_lat, row.stop_lon],
            radius=5,
            color="#08519c",
            fill=True,
            fill_color="#3182bd",
            fill_opacity=0.7,
            weight=0.8,
            tooltip=(
                f"{row.stop_name} | {row.best_direct_frequency_per_hour:.2f}/hour | "
                f"{row.best_time_to_target_min:.1f} min to target | "
                f"{row.direct_routes if row.direct_routes else 'n/a'}"
            ),
        ).add_to(direct_layer)

    transfer_layer = folium.FeatureGroup(name="Transfer feeder services", show=False).add_to(fmap)
    for row in transfer_segments.itertuples(index=False):
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in row.geometry.coords],
            color=row.color,
            weight=max(2, row.weight - 1),
            opacity=0.65,
            tooltip=f"{row.route_label} | feeder to {row.transfer_stop_name if row.transfer_stop_name else 'transfer'}",
            popup=(
                f"Feeder route: {row.route_label}<br>"
                f"Origin stop: {row.origin_stop_name if row.origin_stop_name else 'n/a'}<br>"
                f"Transfer stop: {row.transfer_stop_name if row.transfer_stop_name else 'n/a'}"
            ),
        ).add_to(transfer_layer)
    for row in transfer_markers.itertuples(index=False):
        if getattr(row, "marker_role", "") == "transfer_stop" and pd.notna(row.radius_m):
            folium.Circle(
                location=[row.stop_lat, row.stop_lon],
                radius=float(row.radius_m),
                color="#f16913",
                weight=1.2,
                fill=False,
                opacity=0.65,
                tooltip=f"Transfer radius: {float(row.radius_m):.0f} m",
            ).add_to(transfer_layer)
        marker_color = "#7f2704" if getattr(row, "marker_role", "") == "transfer_stop" else "#b15928"
        marker_fill = "#f16913" if getattr(row, "marker_role", "") == "transfer_stop" else "#fdae6b"
        folium.CircleMarker(
            location=[row.stop_lat, row.stop_lon],
            radius=5,
            color=marker_color,
            fill=True,
            fill_color=marker_fill,
            fill_opacity=0.8,
            weight=0.9,
            tooltip=(
                f"{row.stop_name} | {row.feasible_departures_per_hour:.2f}/hour | "
                f"{row.best_total_time_to_target_min:.1f} min to target | "
                f"{row.route_label if pd.notna(getattr(row, 'route_label', np.nan)) else row.feeder_route_label if pd.notna(getattr(row, 'feeder_route_label', np.nan)) else 'n/a'}"
            ),
            popup=(
                f"{row.stop_name}<br>"
                f"Feeder frequency: {row.feasible_departures_per_hour:.2f}/hour<br>"
                f"Best time to target: {row.best_total_time_to_target_min:.1f} min<br>"
                f"Role: {'Transfer stop' if getattr(row, 'marker_role', '') == 'transfer_stop' else 'Feeder-path stop'}"
            ),
        ).add_to(transfer_layer)

    if not earlier_feeder_segments.empty or not earlier_feeder_markers.empty:
        earlier_layer = folium.FeatureGroup(name="Earlier feeder services (2+ transfers)", show=False).add_to(fmap)
        for row in earlier_feeder_segments.itertuples(index=False):
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in row.geometry.coords],
                color=row.color,
                weight=row.weight,
                opacity=0.45,
                tooltip=f"{row.route_label} | earlier feeder service",
            ).add_to(earlier_layer)
        for row in earlier_feeder_markers.itertuples(index=False):
            folium.Circle(
                location=[row.stop_lat, row.stop_lon],
                radius=float(row.radius_m),
                color="#756bb1",
                weight=1.1,
                fill=False,
                opacity=0.6,
                tooltip=f"Transfer radius: {float(row.radius_m):.0f} m",
            ).add_to(earlier_layer)
            folium.CircleMarker(
                location=[row.stop_lat, row.stop_lon],
                radius=5,
                color="#54278f",
                fill=True,
                fill_color="#756bb1",
                fill_opacity=0.7,
                weight=0.9,
                tooltip=(
                    f"{row.stop_name} | {row.feasible_departures_per_hour:.2f}/hour | "
                    f"{row.best_total_time_to_target_min:.1f} min to target | "
                    f"{int(row.best_transfer_count)} transfer(s)"
                ),
                popup=(
                    f"{row.stop_name}<br>"
                    f"Earlier-feeder frequency: {row.feasible_departures_per_hour:.2f}/hour<br>"
                    f"Best time to target: {row.best_total_time_to_target_min:.1f} min<br>"
                    f"Transfers needed: {int(row.best_transfer_count)}<br>"
                    f"Earlier feeder routes: {row.feeder_route_label if row.feeder_route_label else 'n/a'}"
                ),
            ).add_to(earlier_layer)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    fmap.save(output_path)


def write_readme(
    args: argparse.Namespace,
    lat: float,
    lon: float,
    service_date: pd.Timestamp,
    district: gpd.GeoDataFrame,
    workplace_stops: pd.DataFrame,
) -> None:
    preferences = resolve_preferences(args)
    transfer_penalty_min = resolve_transfer_penalty_minutes(args)
    district_row = district.iloc[0]
    text = f"""# Commute Ranking by Target Location

Target location input:
- coordinate string: `{args.workplace_coordinate or f"{lat}, {lon}"}`
- latitude: `{lat}`
- longitude: `{lon}`

Detected district:
- NUTS3 id: `{district_row['NUTS_ID']}`
- name: `{district_row['NUTS_NAME']}`

Service date:
- `{service_date.date()}`

Arrival window at the target:
- `{args.arrival_start}` to `{args.arrival_end}`

Target-side stops used:
{workplace_stops[['stop_id', 'stop_name', 'distance_to_work_m']].to_string(index=False)}

Spatial scope:
- residential search space is restricted to Stadtteile intersecting the detected NUTS3 district
- every Stadtteil in the district is kept in the heatmap, even when no feasible commute is found under the current model

Transit option rules:
- the user selects the arrival window
- local transport includes bus, tram, U-Bahn/subway, ferry/funicular-like modes, and rail-like services including S-Bahn
- up to `{args.max_transfers}` transfers are considered
- transfers can happen between nearby stops within `{args.transfer_radius_m:.0f}` meters
- minimum transfer time: `{args.min_transfer_min:.1f}` minutes
- maximum transfer wait: `{args.max_transfer_wait_min:.1f}` minutes

User preference scales:
- rent importance: `{preferences['rent']:.0f}/10`
- time importance: `{preferences['time']:.0f}/10`
- frequency importance: `{preferences['frequency']:.0f}/10`
- transfer dislike: `{preferences['transfer']:.0f}/10`
- green space importance: `{preferences['green_space']:.0f}/10`
- supermarket importance: `{preferences['supermarket']:.0f}/10`
- hospital importance: `{preferences['hospital']:.0f}/10`

How the final score works:
- each top-level dimension is normalized to the district context before weighting
- the user preference values are then applied directly to those normalized dimensions
- there are no hidden category multipliers in the final ranking or heatmap
- within the frequency dimension, the script combines departures/hour, direct-line breadth, direct-stop breadth, and latest feasible departure
- within the transfer dimension, the script combines transfer count, downstream direct-service richness, and transfer wait quality
- transfer dislike is also used as a `{transfer_penalty_min:.2f}` minute-equivalent penalty when choosing between otherwise feasible itinerary options

Additional commute quality signals:
- `direct_line_count`: number of distinct direct target-serving line groups from the Stadtteil
- `useful_line_count_total`: number of distinct useful line groups across feasible options in the Stadtteil
- `unique_departure_minutes_per_hour`: departure-minute coverage across the Stadtteil
- `latest_departure_time`: latest feasible departure still reaching the target in the chosen window
- `best_transfer_direct_line_count`: number of direct line groups available at the chosen transfer stop
- `best_transfer_direct_minute_count`: downstream direct minute coverage at the chosen transfer stop
- `best_transfer_wait_min` / `median_transfer_wait_min`: selected transfer wait quality
- `green_space_share`: share of Stadtteil area covered by mapped green spaces
- `green_space_patch_density_km2`: count of mapped green-space patches per km² as a simple distributed-greenery proxy
- `street_tree_density_km2`: count of mapped urban tree points and tree-row features per km²
- `supermarket_count`: number of mapped supermarkets within the Stadtteil
- `hospital_count`: number of mapped hospitals within the Stadtteil

Heatmap scale:
- green is better
- the scale is applied to all Stadtteile in the district
- Stadtteile without a feasible commute are not blank; they receive a low score instead

Map behavior:
- the district heatmap colors Stadtteile by a district-wide suitability score
- the map always shows the target location and nearest target stops
- the direct target services layer shows complete direct services that reach the target-side stop set during the selected window
- the direct target layer includes route frequency in the selected window and stop-level time-to-target values
- the transfer feeder services layer only shows the final feeder approach that hands off into the direct target network
- the heatmap tooltip also shows Stadtteil-wide feasible departures per hour, useful line groups, departure-minute coverage, latest feasible departure, transfer-side direct-line richness, green space share, supermarket count, and hospital count
- if 2+ transfers are allowed and found, an earlier feeder layer is added for feeder legs that happen before the final transfer approach

Output files:
- `data/processed/commute_options_by_workplace_stdt.csv`
- `data/processed/commute_ranking_by_workplace_stdt.csv`
- `data/processed/commute_district_stdt.geojson`
- `data/processed/commute_target_direct_routes_stdt.geojson`
- `data/processed/commute_transfer_feeder_routes_stdt.geojson`
- `data/processed/commute_earlier_feeder_routes_stdt.geojson`
- `outputs/maps/commute_district_heatmap_stdt.html`
"""
    (ROOT / "README_commute_ranking_stdt.md").write_text(text + "\n", encoding="utf-8")


def load_cached_commute_bundle(bundle_dir: Path) -> dict[str, Any]:
    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing cached commute bundle at {bundle_dir}. Run src/04_compute_commute_options_by_target_stdt.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    district_gtfs_stops = gpd.read_file(bundle_dir / "district_gtfs_stops.gpkg")
    options_df = pd.read_pickle(bundle_dir / "options_raw.pkl")
    destination_occurrences = pd.read_pickle(bundle_dir / "destination_occurrences.pkl")
    relevant_stop_times = pd.read_pickle(bundle_dir / "relevant_stop_times.pkl")
    assigned_stops = pd.read_pickle(bundle_dir / "assigned_stops.pkl")
    workplace_stops = pd.read_pickle(bundle_dir / "workplace_stops.pkl")
    return {
        "metadata": metadata,
        "district_gtfs_stops": district_gtfs_stops,
        "options_df": options_df,
        "destination_occurrences": destination_occurrences,
        "relevant_stop_times": relevant_stop_times,
        "assigned_stops": assigned_stops,
        "workplace_stops": workplace_stops,
    }


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    COMMUTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    args = parse_args_with_config()
    lat, lon = parse_coordinate_from_config(args)
    args.transfer_radius_m = normalize_transfer_radius_m(args.transfer_radius_m, args.min_transfer_radius_m, args.max_transfer_radius_m)
    args.max_transfers = max(0, int(args.max_transfers))
    preferences = resolve_preferences(args)
    transfer_penalty_min = resolve_transfer_penalty_minutes(args)

    district, workplace_point = detect_workplace_district(lat, lon, args.nuts_path)
    district_stdt_gdf = load_district_stdt_geodata(district)
    district_stdt_set = set(district_stdt_gdf["stdt_id"].dropna().tolist())
    if not district_stdt_set:
        raise ValueError("No Stadtteil polygons intersect the detected district.")

    service_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    computed_max_transfers = max(3, int(args.max_transfers))
    profile = routing_profile_dict(args, lat, lon, service_date, computed_max_transfers)
    bundle_dir = commute_cache_bundle_dir(profile)
    cached = load_cached_commute_bundle(bundle_dir)
    cached_metadata = cached["metadata"]
    if int(cached_metadata.get("computed_max_transfers", computed_max_transfers)) < int(args.max_transfers):
        raise ValueError(
            f"Cached commute bundle only contains up to {cached_metadata.get('computed_max_transfers')} transfers. "
            "Run src/04_compute_commute_options_by_target_stdt.py again."
        )

    rent_df = load_rent_data(district_stdt_set)
    assigned_stops = cached["assigned_stops"]
    district_gtfs_stops = cached["district_gtfs_stops"]
    if assigned_stops.empty:
        raise ValueError("No Stadtteil-assigned GTFS stops were found inside the detected district.")

    arrival_start_sec = parse_gtfs_time_to_seconds(args.arrival_start)
    arrival_end_sec = parse_gtfs_time_to_seconds(args.arrival_end)
    if arrival_start_sec is None or arrival_end_sec is None or arrival_end_sec <= arrival_start_sec:
        raise ValueError("Arrival window must be valid and end after start.")
    window_hours = (arrival_end_sec - arrival_start_sec) / 3600.0

    workplace_stops = cached["workplace_stops"]
    if workplace_stops.empty:
        raise ValueError("No GTFS stops were found within the configured workplace distance.")
    workplace_stops_display = clip_stop_dataframe_to_district(workplace_stops, district)
    relevant_stop_times = cached["relevant_stop_times"]
    destination_occurrences = cached["destination_occurrences"]
    if destination_occurrences.empty:
        raise ValueError("No active local trips reach the workplace stop set in the selected arrival window.")
    options_df = cached["options_df"].copy()
    options_df = options_df.loc[pd.to_numeric(options_df["transfer_count"], errors="coerce").fillna(0).le(args.max_transfers)].copy()
    ranking_df, options_df = aggregate_and_rank(options_df, rent_df, args, window_hours)
    district_heatmap_gdf = build_district_heatmap_frame(district_stdt_gdf, ranking_df, args)

    direct_target_segments, direct_target_stops = build_target_direct_layer_data(
        destination_occurrences=destination_occurrences,
        district_stops_gdf=district_gtfs_stops,
        assigned_stops=assigned_stops,
        window_hours=window_hours,
        relevant_stop_times=relevant_stop_times,
    )
    direct_target_line_keys = {
        str(line_key) for line_key in destination_occurrences["line_key"].dropna().astype(str).unique().tolist()
    }

    transfer_segments, transfer_markers = build_used_feeder_layer_data(
        options_df=options_df,
        district_stops_gdf=district_gtfs_stops,
        relevant_stop_times=relevant_stop_times,
        transfer_radius_m=args.transfer_radius_m,
        window_hours=window_hours,
        direct_target_line_keys=direct_target_line_keys,
    )
    earlier_feeder_segments, earlier_feeder_markers = build_earlier_feeder_layer_data(
        options_df=options_df,
        district_stops_gdf=district_gtfs_stops,
        relevant_stop_times=relevant_stop_times,
        transfer_radius_m=args.transfer_radius_m,
        window_hours=window_hours,
    )

    export_keep_cols = [
        "stdt_id",
        "stdt_name",
        "municipality_name",
        "heatmap_score",
        "commute_status",
        "rent_data_status",
        "rent_data_label",
        "score_summary",
        "score_rent_contribution",
        "score_time_contribution",
        "score_frequency_contribution",
        "score_transfer_contribution",
        "score_green_space_contribution",
        "score_supermarket_contribution",
        "score_hospital_contribution",
        "median_rent_per_m2",
        "green_space_share",
        "green_space_patch_density_km2",
        "street_tree_density_km2",
        "supermarket_count",
        "hospital_count",
        "best_travel_time_min",
        "best_transfer_count",
        "best_transfer_direct_line_count",
        "departures_to_work_per_hour",
        "useful_line_count_total",
        "unique_departure_minutes_per_hour",
        "geometry",
    ]
    district_export_gdf = district_heatmap_gdf[[col for col in export_keep_cols if col in district_heatmap_gdf.columns]].copy()
    district_export_gdf.to_file(PROCESSED_DIR / "commute_district_stdt.geojson", driver="GeoJSON")
    direct_target_segments.to_file(PROCESSED_DIR / "commute_target_direct_routes_stdt.geojson", driver="GeoJSON")
    transfer_segments.to_file(PROCESSED_DIR / "commute_transfer_feeder_routes_stdt.geojson", driver="GeoJSON")
    if not earlier_feeder_segments.empty:
        earlier_feeder_segments.to_file(PROCESSED_DIR / "commute_earlier_feeder_routes_stdt.geojson", driver="GeoJSON")

    create_interactive_map(
        district=district,
        district_heatmap_gdf=district_heatmap_gdf,
        target_point=workplace_point,
        target_stops=workplace_stops_display,
        target_radius_m=args.max_workplace_stop_distance_m,
        direct_target_segments=direct_target_segments,
        direct_target_stops=direct_target_stops,
        transfer_segments=transfer_segments,
        transfer_markers=transfer_markers,
        earlier_feeder_segments=earlier_feeder_segments,
        earlier_feeder_markers=earlier_feeder_markers,
        output_path=MAPS_DIR / "commute_district_heatmap_stdt.html",
    )
    write_readme(
        args=args,
        lat=lat,
        lon=lon,
        service_date=service_date,
        district=district,
        workplace_stops=workplace_stops,
    )

    print("\n=== District Commute Ranking Summary ===")
    print("target location coordinate:", f"{lat:.6f}, {lon:.6f}")
    print("district:", district.iloc[0]["NUTS_ID"], "-", district.iloc[0]["NUTS_NAME"])
    print("service date:", service_date.date())
    print(
        "user preferences:",
        json.dumps(
            {
                "rent_importance": preferences["rent"],
                "time_importance": preferences["time"],
                "frequency_importance": preferences["frequency"],
                "transfer_penalty_min": transfer_penalty_min,
                "green_space_importance": preferences["green_space"],
                "supermarket_importance": preferences["supermarket"],
                "hospital_importance": preferences["hospital"],
            }
        ),
    )
    print("district Stadtteil polygons:", len(district_stdt_gdf))
    print("district GTFS stops:", len(district_gtfs_stops))
    print("target stops used:", len(workplace_stops))
    print("active local trips on date:", int(cached_metadata.get("active_local_trips", 0)))
    print("destination trip occurrences in window:", len(destination_occurrences))
    print("cached commute bundle:", bundle_dir.relative_to(ROOT))
    print("direct options:", int((pd.to_numeric(cached["options_df"]["transfer_count"], errors="coerce").fillna(0) == 0).sum()))
    print(
        "transfer options:",
        int((pd.to_numeric(cached["options_df"]["transfer_count"], errors="coerce").fillna(0) > 0).sum()),
    )
    print("feasible-commute Stadtteile:", len(ranking_df))
    print("district heatmap Stadtteile:", len(district_heatmap_gdf))
    print("direct target routes:", direct_target_segments["route_id"].nunique() if not direct_target_segments.empty else 0)
    print("transfer feeder routes:", transfer_segments["route_id"].nunique() if not transfer_segments.empty else 0)
    print("earlier feeder routes (2+ transfers):", earlier_feeder_segments["route_id"].nunique() if not earlier_feeder_segments.empty else 0)

    print("\nTop feasible commute Stadtteile")
    print(
        ranking_df.head(args.top_n)[
            [
                "stdt_id",
                "district_rank",
                "best_travel_time_min",
                "best_transfer_count",
                "departures_to_work_per_hour",
                "useful_line_count_total",
                "median_rent_per_m2",
                "ranking_score",
                "preference_reason",
            ]
        ].to_string(index=False)
    )
    print("\nLowest-scoring district Stadtteile on the all-Stadtteil heatmap")
    print(
        district_heatmap_gdf.sort_values("heatmap_score").head(min(args.top_n, 10))[
            ["stdt_id", "heatmap_score", "commute_status", "median_rent_per_m2", "best_travel_time_min", "departures_to_work_per_hour"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
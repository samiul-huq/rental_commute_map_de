from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
GTFS_DIR = INPUT_DIR / "gtfs"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"

GTFS_FILES = [
    "agency.txt",
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "stop_times.txt",
]

CALENDAR_WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def ensure_directories() -> None:
    for path in [INTERMEDIATE_DIR, PROCESSED_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def inspect_gtfs_files() -> pd.DataFrame:
    rows = []
    print("\n=== GTFS Inventory ===")
    for name in GTFS_FILES:
        path = GTFS_DIR / name
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            row_count = sum(1 for _ in reader)
        print(f"{name}: rows={row_count:,} columns={columns}")
        rows.append({"file_name": name, "row_count": row_count, "columns_json": json.dumps(columns)})

    inventory_df = pd.DataFrame(rows)
    inventory_df.to_csv(INTERMEDIATE_DIR / "gtfs_table_inventory.csv", index=False, encoding="utf-8")
    return inventory_df


def load_stdt_geography() -> gpd.GeoDataFrame:
    rent_geojson = PROCESSED_DIR / "rent_by_stdt.geojson"
    if not rent_geojson.exists():
        raise FileNotFoundError(f"Could not find required stage-1 output: {rent_geojson}")
    gdf = gpd.read_file(rent_geojson)
    source = rent_geojson

    gdf["stdt_id"] = gdf["stdt_id"].astype("string") if "stdt_id" in gdf.columns else gdf["osm_id"].astype("string")
    if "stdt_name" not in gdf.columns:
        gdf["stdt_name"] = gdf["name_de"].fillna(gdf["name"]).astype("string")
    print("\n=== Stadtteil Geography ===")
    print("source:", source.relative_to(ROOT))
    print("shape:", gdf.shape)
    print("crs:", gdf.crs)
    print("stadtteil key column: stdt_id")
    print("invalid geometries:", int((~gdf.geometry.is_valid).sum()))
    return gdf


def load_rent_outputs() -> pd.DataFrame:
    rent_df = pd.read_csv(PROCESSED_DIR / "rent_by_stdt.csv", dtype={"stdt_id": "string"})
    keep_cols = [
        "stdt_id",
        "stdt_name",
        "municipality_name",
        "grid_cell_count",
        "mean_rent_per_m2",
        "median_rent_per_m2",
        "std_rent_per_m2",
        "q25_rent_per_m2",
        "q75_rent_per_m2",
    ]
    rent_df = rent_df[[col for col in keep_cols if col in rent_df.columns]].copy()
    print("\n=== Rent Outputs ===")
    print("rent_by_stdt shape:", rent_df.shape)
    return rent_df


def expand_service_days(calendar_df: pd.DataFrame, calendar_dates_df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.Timestamp, pd.Timestamp]:
    calendar_df = calendar_df.copy()
    calendar_dates_df = calendar_dates_df.copy()
    calendar_df["start_date"] = pd.to_datetime(calendar_df["start_date"].astype(str), format="%Y%m%d")
    calendar_df["end_date"] = pd.to_datetime(calendar_df["end_date"].astype(str), format="%Y%m%d")
    calendar_dates_df["date"] = pd.to_datetime(calendar_dates_df["date"].astype(str), format="%Y%m%d")

    overall_start = min(calendar_df["start_date"].min(), calendar_dates_df["date"].min())
    overall_end = max(calendar_df["end_date"].max(), calendar_dates_df["date"].max())

    dates = pd.DataFrame({"date": pd.date_range(overall_start, overall_end, freq="D")})
    dates["weekday_name"] = dates["date"].dt.day_name().str.lower()

    calendar_long = calendar_df.melt(
        id_vars=["service_id", "start_date", "end_date"],
        value_vars=CALENDAR_WEEKDAYS,
        var_name="weekday_name",
        value_name="runs",
    )
    calendar_long = calendar_long.loc[calendar_long["runs"] == 1].copy()

    base_service_dates = calendar_long.merge(dates, on="weekday_name", how="inner")
    base_service_dates = base_service_dates.loc[
        base_service_dates["date"].between(base_service_dates["start_date"], base_service_dates["end_date"])
    ][["service_id", "date"]]

    additions = calendar_dates_df.loc[calendar_dates_df["exception_type"] == 1, ["service_id", "date"]]
    removals = calendar_dates_df.loc[calendar_dates_df["exception_type"] == 2, ["service_id", "date"]]

    service_dates = pd.concat([base_service_dates, additions], ignore_index=True).drop_duplicates()
    if not removals.empty:
        service_dates = service_dates.merge(removals.assign(remove_flag=1), on=["service_id", "date"], how="left")
        service_dates = service_dates.loc[service_dates["remove_flag"].isna(), ["service_id", "date"]]

    service_days = (
        service_dates.groupby("service_id")
        .size()
        .rename("service_days_count")
        .reset_index()
        .sort_values("service_days_count", ascending=False)
    )
    service_days.to_csv(INTERMEDIATE_DIR / "gtfs_service_days_by_service_id.csv", index=False, encoding="utf-8")

    covered_service_days = int(service_dates["date"].nunique())
    print("\n=== Service Day Handling ===")
    print("overall feed window:", overall_start.date(), "to", overall_end.date())
    print("covered service days:", covered_service_days)
    print("service_ids with activity:", service_days["service_id"].nunique())
    return service_days, covered_service_days, overall_start, overall_end


def classify_route_type(route_type_value: object) -> str:
    try:
        route_type = int(float(route_type_value))
    except (TypeError, ValueError):
        return "other"

    if route_type == 2 or 100 <= route_type < 200 or 400 <= route_type < 500 or 1500 <= route_type < 1700:
        return "rail"

    if route_type in {0, 1, 3, 4, 5, 6, 7, 11, 12} or 200 <= route_type < 400 or 700 <= route_type < 1500:
        return "local"

    return "other"


def build_stop_stdt_assignment(stdt_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    stops_df = pd.read_csv(
        GTFS_DIR / "stops.txt",
        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
        dtype={"stop_id": "string", "stop_name": "string", "parent_station": "string"},
    )

    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
    stops_df["location_type"] = pd.to_numeric(stops_df["location_type"], errors="coerce")

    # First-pass practical choice: count platform/stop points that are boardable or unspecified.
    stops_df = stops_df.loc[
        stops_df["stop_lat"].notna()
        & stops_df["stop_lon"].notna()
        & (stops_df["location_type"].isna() | stops_df["location_type"].eq(0))
    ].copy()

    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs="EPSG:4326",
    )
    if stops_gdf.crs != stdt_gdf.crs:
        stops_gdf = stops_gdf.to_crs(stdt_gdf.crs)

    stops_with_stdt = gpd.sjoin(
        stops_gdf,
        stdt_gdf[["stdt_id", "stdt_name", "municipality_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    assigned = stops_with_stdt.loc[stops_with_stdt["stdt_id"].notna()].copy()
    assigned_local = assigned.to_crs(3035)
    assigned_local["area_m2"] = assigned_local.geometry.area
    assigned_local = (
        assigned_local.sort_values(["stop_id", "area_m2"], ascending=[True, True])
        .drop_duplicates(subset=["stop_id"], keep="first")
        .copy()
    )
    assigned = assigned_local.to_crs(stdt_gdf.crs)
    assigned["stdt_id"] = assigned["stdt_id"].astype("string")
    assigned[["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station", "stdt_id", "stdt_name", "municipality_name"]].to_csv(
        INTERMEDIATE_DIR / "gtfs_stops_with_stdt.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    print("\n=== Stop Assignment ===")
    print("stops with coordinates and boardable type:", len(stops_df))
    print("stops assigned to Stadtteile:", len(assigned))
    print("unique assigned stop_ids:", assigned["stop_id"].nunique())
    return assigned[["stop_id", "stdt_id"]].drop_duplicates()


def init_pair_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("CREATE TABLE stdt_trip_pairs (stdt_id TEXT NOT NULL, trip_id TEXT NOT NULL, PRIMARY KEY (stdt_id, trip_id));")
    conn.execute("CREATE TABLE stdt_route_pairs (stdt_id TEXT NOT NULL, route_id TEXT NOT NULL, PRIMARY KEY (stdt_id, route_id));")
    conn.execute("CREATE TABLE stdt_agency_pairs (stdt_id TEXT NOT NULL, agency_id TEXT NOT NULL, PRIMARY KEY (stdt_id, agency_id));")
    conn.execute(
        "CREATE TABLE stdt_mode_stop_pairs (stdt_id TEXT NOT NULL, stop_id TEXT NOT NULL, mode_group TEXT NOT NULL, PRIMARY KEY (stdt_id, stop_id, mode_group));"
    )
    return conn


def insert_unique_pairs(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame, columns: list[str]) -> None:
    if df.empty:
        return
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, map(tuple, df[columns].itertuples(index=False, name=None)))


def process_stop_times(
    stop_stdt_df: pd.DataFrame,
    trips_lookup_df: pd.DataFrame,
    covered_service_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stop_to_stdt = stop_stdt_df.set_index("stop_id")["stdt_id"]
    relevant_stop_ids = set(stop_to_stdt.index.tolist())

    pair_db_path = INTERMEDIATE_DIR / "gtfs_stdt_pairs.sqlite"
    conn = init_pair_db(pair_db_path)

    departures_totals: defaultdict[str, float] = defaultdict(float)
    departures_by_mode: defaultdict[tuple[str, str], float] = defaultdict(float)
    duplicate_assignment_rows = 0

    chunk_size = 2_000_000
    print("\n=== stop_times Chunk Processing ===")
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            GTFS_DIR / "stop_times.txt",
            usecols=["trip_id", "stop_id", "departure_time"],
            dtype={"trip_id": "string", "stop_id": "string", "departure_time": "string"},
            chunksize=chunk_size,
        ),
        start=1,
    ):
        chunk = chunk.loc[chunk["stop_id"].isin(relevant_stop_ids)].copy()
        if chunk.empty:
            print(f"chunk {chunk_index}: no relevant stop_ids")
            continue

        chunk["stdt_id"] = chunk["stop_id"].map(stop_to_stdt)
        duplicate_assignment_rows += int(chunk.duplicated(subset=["stop_id", "trip_id", "departure_time", "stdt_id"]).sum())

        chunk = chunk.merge(trips_lookup_df, on="trip_id", how="left")
        chunk = chunk.loc[chunk["stdt_id"].notna() & chunk["service_days_count"].fillna(0).gt(0)].copy()
        if chunk.empty:
            print(f"chunk {chunk_index}: no active service rows after trip lookup")
            continue

        has_departure = chunk["departure_time"].notna() & chunk["departure_time"].ne("")
        departures_chunk = chunk.loc[has_departure].groupby("stdt_id")["service_days_count"].sum()
        for stdt_id, value in departures_chunk.items():
            departures_totals[str(stdt_id)] += float(value)

        departures_mode_chunk = (
            chunk.loc[has_departure & chunk["mode_group"].isin(["rail", "local"])]
            .groupby(["stdt_id", "mode_group"])["service_days_count"]
            .sum()
        )
        for (stdt_id, mode_group), value in departures_mode_chunk.items():
            departures_by_mode[(str(stdt_id), str(mode_group))] += float(value)

        trip_pairs = chunk[["stdt_id", "trip_id"]].drop_duplicates()
        route_pairs = chunk[["stdt_id", "route_id"]].dropna().drop_duplicates()
        agency_pairs = chunk[["stdt_id", "agency_id"]].dropna().drop_duplicates()
        mode_stop_pairs = (
            chunk.loc[chunk["mode_group"].isin(["rail", "local"]), ["stdt_id", "stop_id", "mode_group"]]
            .drop_duplicates()
        )

        insert_unique_pairs(conn, "stdt_trip_pairs", trip_pairs, ["stdt_id", "trip_id"])
        insert_unique_pairs(conn, "stdt_route_pairs", route_pairs, ["stdt_id", "route_id"])
        insert_unique_pairs(conn, "stdt_agency_pairs", agency_pairs, ["stdt_id", "agency_id"])
        insert_unique_pairs(conn, "stdt_mode_stop_pairs", mode_stop_pairs, ["stdt_id", "stop_id", "mode_group"])
        conn.commit()
        print(
            f"chunk {chunk_index}: rows={len(chunk):,} trip_pairs={len(trip_pairs):,} "
            f"route_pairs={len(route_pairs):,} agency_pairs={len(agency_pairs):,}"
        )

    stop_count_df = stop_stdt_df.groupby("stdt_id")["stop_id"].nunique().rename("stop_count").reset_index()

    trip_count_df = pd.read_sql_query(
        "SELECT stdt_id, COUNT(*) AS unique_trips_count FROM stdt_trip_pairs GROUP BY stdt_id",
        conn,
    )
    route_count_df = pd.read_sql_query(
        "SELECT stdt_id, COUNT(*) AS unique_routes_count FROM stdt_route_pairs GROUP BY stdt_id",
        conn,
    )
    agency_count_df = pd.read_sql_query(
        "SELECT stdt_id, COUNT(*) AS agencies_count FROM stdt_agency_pairs GROUP BY stdt_id",
        conn,
    )
    mode_stop_count_df = pd.read_sql_query(
        "SELECT stdt_id, mode_group, COUNT(*) AS stop_count_mode FROM stdt_mode_stop_pairs GROUP BY stdt_id, mode_group",
        conn,
    )
    conn.close()

    departures_df = pd.DataFrame(
        [{"stdt_id": stdt_id, "departures_total_feed_window": value} for stdt_id, value in departures_totals.items()]
    )
    if departures_df.empty:
        departures_df = pd.DataFrame(columns=["stdt_id", "departures_total_feed_window"])

    mode_departures_df = pd.DataFrame(
        [
            {"stdt_id": stdt_id, "mode_group": mode_group, "departures_total_feed_window_mode": value}
            for (stdt_id, mode_group), value in departures_by_mode.items()
        ]
    )

    metrics_df = stop_count_df.merge(trip_count_df, on="stdt_id", how="outer")
    metrics_df = metrics_df.merge(route_count_df, on="stdt_id", how="outer")
    metrics_df = metrics_df.merge(agency_count_df, on="stdt_id", how="outer")
    metrics_df = metrics_df.merge(departures_df, on="stdt_id", how="outer")
    metrics_df = metrics_df.fillna(
        {
            "stop_count": 0,
            "unique_trips_count": 0,
            "unique_routes_count": 0,
            "agencies_count": 0,
            "departures_total_feed_window": 0,
        }
    )

    metrics_df["covered_service_days"] = covered_service_days
    metrics_df["departures_per_day_avg"] = metrics_df["departures_total_feed_window"] / covered_service_days
    metrics_df["departures_total_7d"] = metrics_df["departures_per_day_avg"] * 7
    metrics_df["departures_per_stop_per_day"] = np.where(
        metrics_df["stop_count"] > 0,
        metrics_df["departures_per_day_avg"] / metrics_df["stop_count"],
        np.nan,
    )

    if not mode_stop_count_df.empty:
        rail_stop = mode_stop_count_df.loc[mode_stop_count_df["mode_group"] == "rail", ["stdt_id", "stop_count_mode"]].rename(
            columns={"stop_count_mode": "rail_stop_count"}
        )
        local_stop = mode_stop_count_df.loc[
            mode_stop_count_df["mode_group"] == "local", ["stdt_id", "stop_count_mode"]
        ].rename(columns={"stop_count_mode": "local_transit_stop_count"})
        metrics_df = metrics_df.merge(rail_stop, on="stdt_id", how="left").merge(local_stop, on="stdt_id", how="left")
    else:
        metrics_df["rail_stop_count"] = np.nan
        metrics_df["local_transit_stop_count"] = np.nan

    if not mode_departures_df.empty:
        rail_dep = mode_departures_df.loc[
            mode_departures_df["mode_group"] == "rail", ["stdt_id", "departures_total_feed_window_mode"]
        ].rename(columns={"departures_total_feed_window_mode": "rail_departures_total_feed_window"})
        local_dep = mode_departures_df.loc[
            mode_departures_df["mode_group"] == "local", ["stdt_id", "departures_total_feed_window_mode"]
        ].rename(columns={"departures_total_feed_window_mode": "local_transit_departures_total_feed_window"})
        metrics_df = metrics_df.merge(rail_dep, on="stdt_id", how="left").merge(local_dep, on="stdt_id", how="left")
    else:
        metrics_df["rail_departures_total_feed_window"] = np.nan
        metrics_df["local_transit_departures_total_feed_window"] = np.nan

    metrics_df["rail_departures_per_day"] = metrics_df["rail_departures_total_feed_window"].fillna(0) / covered_service_days
    metrics_df["local_transit_departures_per_day"] = (
        metrics_df["local_transit_departures_total_feed_window"].fillna(0) / covered_service_days
    )

    metrics_df = metrics_df.drop(
        columns=["departures_total_feed_window", "rail_departures_total_feed_window", "local_transit_departures_total_feed_window"]
    )

    diagnostics_df = pd.DataFrame(
        {
            "metric": ["duplicate_stop_time_assignments_seen"],
            "value": [duplicate_assignment_rows],
        }
    )
    diagnostics_df.to_csv(INTERMEDIATE_DIR / "gtfs_processing_diagnostics_stdt.csv", index=False, encoding="utf-8")
    return metrics_df, diagnostics_df


def save_plots(merged_gdf: gpd.GeoDataFrame, merged_df: pd.DataFrame) -> None:
    def save_choropleth(column: str, title: str, output_name: str, cmap: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 14))
        merged_gdf.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            linewidth=0.05,
            edgecolor="#555555",
            legend=True,
            missing_kwds={
                "color": "#e3e3e3",
                "edgecolor": "#9a9a9a",
                "hatch": "///",
                "label": "No transit data",
            },
        )
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    save_choropleth(
        "departures_per_day_avg",
        "Germany Stadtteil Departures per Day (Average)",
        "germany_stdt_departures_per_day_stdt.png",
        "YlGnBu",
    )
    save_choropleth(
        "stop_count",
        "Germany Stadtteil GTFS Stop Count",
        "germany_stdt_stop_count_stdt.png",
        "Blues",
    )

    scatter_specs = [
        ("departures_per_day_avg", "Departures per Day (Average)", "rent_vs_departures_scatter_stdt.png"),
        ("stop_count", "Stop Count", "rent_vs_stop_count_scatter_stdt.png"),
    ]

    for metric, xlabel, output_name in scatter_specs:
        plot_df = merged_df.loc[merged_df["median_rent_per_m2"].notna() & merged_df[metric].notna()].copy()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(plot_df[metric], plot_df["median_rent_per_m2"], s=10, alpha=0.35, color="#1f77b4")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Median rent per m²")
        ax.set_title(f"Median Rent per m² vs {xlabel}")
        if plot_df[metric].max() / max(plot_df[metric].min() or 1, 1) > 100:
            ax.set_xscale("log")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")
        plt.close(fig)


def write_readme(
    inventory_df: pd.DataFrame,
    covered_service_days: int,
    overall_start: pd.Timestamp,
    overall_end: pd.Timestamp,
    assigned_stop_count: int,
    metrics_df: pd.DataFrame,
) -> None:
    text = dedent(
        f"""
        # GTFS to Stadtteil Integration

        ## Inputs
        - GTFS folder: `input/gtfs`
        - Stadtteil geography: `data/processed/rent_by_stdt.geojson`

        ## Feed coverage
        - Feed window detected: `{overall_start.date()}` to `{overall_end.date()}`

        ## GTFS files
        {inventory_df[['file_name', 'row_count']].to_string(index=False)}

        ## Service-day handling
        - `calendar.txt` and `calendar_dates.txt` are expanded across the detected feed window.
        - Additions (`exception_type = 1`) are appended and removals (`exception_type = 2`) are subtracted.
        - Covered service days in the feed window: `{covered_service_days}`.
        - `departures_per_day_avg` is total weighted departures divided by covered service days.
        - `departures_total_7d` is `departures_per_day_avg * 7`.

        ## Spatial assignment
        - GTFS stops are converted to points from `stop_lat` and `stop_lon`.
        - Stops are spatially joined into the Stadtteil polygons used by the rent outputs.
        - Only boardable or unspecified stop points (`location_type` null or `0`) are counted.
        - Stops assigned to Stadtteil polygons: `{assigned_stop_count}`.

        ## Transit metrics computed
        - `stop_count`
        - `unique_routes_count`
        - `unique_trips_count`
        - `departures_total_7d`
        - `departures_per_day_avg`
        - `departures_per_stop_per_day`
        - `agencies_count`
        - `rail_stop_count`
        - `local_transit_stop_count`
        - `rail_departures_per_day`
        - `local_transit_departures_per_day`

        ## Mode grouping
        - `rail`: GTFS `route_type` values for standard rail plus rail-like extended codes.
        - `local`: bus, tram, subway, ferry, funicular, and broad local-transit extended codes.

        ## Output files
        - `data/processed/rent_transit_by_stdt.csv`
        - `data/processed/rent_transit_by_stdt.geojson`
        - `data/processed/rent_transit_by_stdt.gpkg`
        """
    ).strip()
    (ROOT / "README_gtfs_stdt_integration.md").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    ensure_directories()

    inventory_df = inspect_gtfs_files()
    stdt_gdf = load_stdt_geography()
    rent_df = load_rent_outputs()

    calendar_df = pd.read_csv(GTFS_DIR / "calendar.txt", dtype={"service_id": "string"})
    calendar_dates_df = pd.read_csv(GTFS_DIR / "calendar_dates.txt", dtype={"service_id": "string"})
    service_days_df, covered_service_days, overall_start, overall_end = expand_service_days(calendar_df, calendar_dates_df)

    routes_df = pd.read_csv(
        GTFS_DIR / "routes.txt",
        usecols=["route_id", "agency_id", "route_type"],
        dtype={"route_id": "string", "agency_id": "string"},
    )
    routes_df["mode_group"] = routes_df["route_type"].map(classify_route_type)

    trips_df = pd.read_csv(
        GTFS_DIR / "trips.txt",
        usecols=["route_id", "service_id", "trip_id"],
        dtype={"route_id": "string", "service_id": "string", "trip_id": "string"},
    )
    trips_lookup_df = trips_df.merge(routes_df, on="route_id", how="left")
    trips_lookup_df = trips_lookup_df.merge(service_days_df, on="service_id", how="left")
    trips_lookup_df["service_days_count"] = trips_lookup_df["service_days_count"].fillna(0)
    trips_lookup_df = trips_lookup_df[["trip_id", "route_id", "agency_id", "mode_group", "service_days_count"]]
    trips_lookup_df.to_csv(INTERMEDIATE_DIR / "gtfs_trip_lookup_sample.csv", index=False, encoding="utf-8")

    stop_stdt_df = build_stop_stdt_assignment(stdt_gdf)
    metrics_df, processing_diag_df = process_stop_times(stop_stdt_df, trips_lookup_df, covered_service_days)

    metrics_summary = metrics_df.describe(include="all").transpose()

    base_geo_columns = [col for col in ["stdt_id", "stdt_name", "municipality_name", "population", "geometry"] if col in stdt_gdf.columns]
    base_stdt_gdf = stdt_gdf[base_geo_columns].copy()
    rent_transit_df = rent_df.merge(metrics_df, on="stdt_id", how="left")
    rent_geo_df = rent_df.drop(columns=["stdt_name", "municipality_name"], errors="ignore")
    merged_gdf = base_stdt_gdf.merge(rent_geo_df, on="stdt_id", how="left").merge(metrics_df, on="stdt_id", how="left")

    rent_transit_df.to_csv(PROCESSED_DIR / "rent_transit_by_stdt.csv", index=False, encoding="utf-8")
    merged_gdf.to_file(PROCESSED_DIR / "rent_transit_by_stdt.geojson", driver="GeoJSON")
    merged_gdf.to_file(PROCESSED_DIR / "rent_transit_by_stdt.gpkg", driver="GPKG")

    save_plots(merged_gdf, rent_transit_df)

    stdt_count = len(stdt_gdf)
    rent_stdt_count = int(rent_df["stdt_id"].nunique())
    transit_stdt_count = int(metrics_df.loc[metrics_df["stop_count"].fillna(0) > 0, "stdt_id"].nunique())
    both_count = int(
        merged_gdf.loc[merged_gdf["median_rent_per_m2"].notna() & merged_gdf["stop_count"].fillna(0).gt(0), "stdt_id"].nunique()
    )
    neither_count = int(
        merged_gdf.loc[merged_gdf["median_rent_per_m2"].isna() & merged_gdf["stop_count"].fillna(0).eq(0), "stdt_id"].nunique()
    )

    print("\n=== Diagnostics ===")
    print("number of Stadtteil polygons:", stdt_count)
    print("number of Stadtteile with rental data:", rent_stdt_count)
    print("number of Stadtteile with transit data:", transit_stdt_count)
    print("number of Stadtteile with both:", both_count)
    print("number of Stadtteile with neither:", neither_count)
    print("duplicate stop assignments seen in chunk processing:", int(processing_diag_df.iloc[0]["value"]))

    print("\nTop 20 Stadtteile by departures_per_day_avg")
    print(
        metrics_df.nlargest(20, "departures_per_day_avg")[["stdt_id", "departures_per_day_avg", "stop_count"]].to_string(index=False)
    )
    print("\nTop 20 Stadtteile by stop_count")
    print(metrics_df.nlargest(20, "stop_count")[["stdt_id", "stop_count", "departures_per_day_avg"]].to_string(index=False))
    print("\nTop 20 Stadtteile by departures_per_stop_per_day")
    print(
        metrics_df.nlargest(20, "departures_per_stop_per_day")[["stdt_id", "departures_per_stop_per_day", "stop_count"]].to_string(index=False)
    )

    print("\nTransit metric summary statistics")
    print(
        metrics_df[
            [
                "stop_count",
                "unique_routes_count",
                "unique_trips_count",
                "departures_total_7d",
                "departures_per_day_avg",
                "departures_per_stop_per_day",
                "agencies_count",
                "rail_stop_count",
                "local_transit_stop_count",
                "rail_departures_per_day",
                "local_transit_departures_per_day",
            ]
        ]
        .describe()
        .to_string()
    )

    zero_stop_stdt = int(metrics_df["stop_count"].fillna(0).eq(0).sum())
    high_departure_threshold = float(metrics_df["departures_per_day_avg"].quantile(0.999))
    suspicious_high = metrics_df.loc[metrics_df["departures_per_day_avg"] > high_departure_threshold]

    print("\nSuspicious value checks")
    print("zero-stop Stadtteile:", zero_stop_stdt)
    print("99.9th percentile departures_per_day_avg:", high_departure_threshold)
    print("Stadtteile above that threshold:", len(suspicious_high))

    write_readme(
        inventory_df=inventory_df,
        covered_service_days=covered_service_days,
        overall_start=overall_start,
        overall_end=overall_end,
        assigned_stop_count=stop_stdt_df["stop_id"].nunique(),
        metrics_df=metrics_df,
    )

    print("\n=== Final Summary ===")
    print("GTFS tables used:", ", ".join(inventory_df["file_name"].tolist()))
    print("stops assigned to Stadtteile:", stop_stdt_df["stop_id"].nunique())
    print("Stadtteile with transit metrics:", transit_stdt_count)
    print("output files written:")
    for path in [
        PROCESSED_DIR / "rent_transit_by_stdt.csv",
        PROCESSED_DIR / "rent_transit_by_stdt.geojson",
        PROCESSED_DIR / "rent_transit_by_stdt.gpkg",
        FIGURES_DIR / "germany_stdt_departures_per_day_stdt.png",
        FIGURES_DIR / "germany_stdt_stop_count_stdt.png",
        FIGURES_DIR / "rent_vs_departures_scatter_stdt.png",
        FIGURES_DIR / "rent_vs_stop_count_scatter_stdt.png",
        ROOT / "README_gtfs_stdt_integration.md",
    ]:
        print(" -", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

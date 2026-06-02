"""
Collect historical NASA FIRMS fire hotspot data and aggregate daily fire features.

Input:
- FIRMS_MAP_KEY from .env

Output:
- data/raw/firms/firms_historical_raw_2018_2022.csv
- data/processed/firms_daily_features_2018_2022.csv

This script fetches NASA FIRMS VIIRS SNPP Standard Processing data one day at a time.
Daily fetching is slower but more stable for historical FIRMS requests.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import MFU_LATITUDE, MFU_LONGITUDE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "firms" / "firms_historical_raw_2018_2022.csv"
)

DAILY_OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "firms_daily_features_2018_2022.csv"
)

START_DATE = "2018-01-01"
END_DATE = "2022-12-31"

FIRMS_SOURCE = "VIIRS_SNPP_SP"

# Stable bounding box around MFU / Chiang Rai nearby region.
# Format: west,south,east,north
BBOX = "99.4,19.6,100.4,20.5"

REQUEST_TIMEOUT_SECONDS = 25
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 1


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("utf-8", errors="replace").decode("utf-8"))


def ensure_output_folders() -> None:
    RAW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DAILY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_firms_map_key() -> str:
    load_dotenv()

    map_key = os.getenv("FIRMS_MAP_KEY") or os.getenv("NASA_FIRMS_MAP_KEY")

    if not map_key:
        raise EnvironmentError(
            "FIRMS_MAP_KEY not found in .env.\n"
            "Add this line to .env:\n"
            "FIRMS_MAP_KEY=your_nasa_firms_map_key_here"
        )

    return map_key.strip()


def build_firms_url(map_key: str, date_str: str) -> str:
    return (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{map_key}/{FIRMS_SOURCE}/{BBOX}/1/{date_str}"
    )


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_km = 6371.0

    diff_lat = math.radians(lat2 - lat1)
    diff_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(diff_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(diff_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


def normalize_confidence(value: Any) -> float:
    if pd.isna(value):
        return 0.0

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"l", "low"}:
            return 30.0
        if normalized in {"n", "nominal"}:
            return 60.0
        if normalized in {"h", "high"}:
            return 90.0

        try:
            return float(normalized)
        except ValueError:
            return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_firms_csv(text: str) -> pd.DataFrame:
    clean_text = text.strip()

    if not clean_text:
        return pd.DataFrame()

    if clean_text.lower().startswith("<!doctype") or clean_text.lower().startswith("<html"):
        raise ValueError("NASA FIRMS returned HTML instead of CSV.")

    if "invalid" in clean_text.lower() and "map" in clean_text.lower():
        raise ValueError("NASA FIRMS MAP_KEY appears invalid.")

    df = pd.read_csv(StringIO(clean_text))

    if df.empty:
        return pd.DataFrame()

    required_columns = ["latitude", "longitude", "acq_date"]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required FIRMS columns: {missing_columns}")

    return df


def calculate_daily_features(date_str: str, df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if df.empty:
        daily_row = {
            "date": date_str,
            "fire_count": 0,
            "fire_avg_confidence": 0.0,
            "fire_avg_brightness": 0.0,
            "fire_avg_frp": 0.0,
            "fire_min_distance_km": 0.0,
            "fire_pressure": 0.0,
        }

        return daily_row, pd.DataFrame()

    df = df.copy()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    if df.empty:
        daily_row = {
            "date": date_str,
            "fire_count": 0,
            "fire_avg_confidence": 0.0,
            "fire_avg_brightness": 0.0,
            "fire_avg_frp": 0.0,
            "fire_min_distance_km": 0.0,
            "fire_pressure": 0.0,
        }

        return daily_row, pd.DataFrame()

    df["date"] = date_str
    df["distance_km"] = df.apply(
        lambda row: haversine_distance_km(
            MFU_LATITUDE,
            MFU_LONGITUDE,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    if "confidence" in df.columns:
        df["confidence_numeric"] = df["confidence"].apply(normalize_confidence)
    else:
        df["confidence_numeric"] = 0.0

    if "bright_ti4" in df.columns:
        df["brightness_numeric"] = pd.to_numeric(df["bright_ti4"], errors="coerce").fillna(0)
    elif "brightness" in df.columns:
        df["brightness_numeric"] = pd.to_numeric(df["brightness"], errors="coerce").fillna(0)
    else:
        df["brightness_numeric"] = 0.0

    if "frp" in df.columns:
        df["frp_numeric"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0)
    else:
        df["frp_numeric"] = 0.0

    df["fire_pressure_point"] = df.apply(
        lambda row: float(row["brightness_numeric"]) / ((float(row["distance_km"]) + 1) ** 2),
        axis=1,
    )

    daily_row = {
        "date": date_str,
        "fire_count": int(len(df)),
        "fire_avg_confidence": round(float(df["confidence_numeric"].mean()), 4),
        "fire_avg_brightness": round(float(df["brightness_numeric"].mean()), 4),
        "fire_avg_frp": round(float(df["frp_numeric"].mean()), 4),
        "fire_min_distance_km": round(float(df["distance_km"].min()), 4),
        "fire_pressure": round(float(df["fire_pressure_point"].sum()), 4),
    }

    return daily_row, df


def fetch_one_day(map_key: str, date_str: str) -> tuple[dict[str, Any], pd.DataFrame]:
    url = build_firms_url(map_key, date_str)

    retries = MAX_RETRIES

    while retries > 0:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

            if response.status_code == 200:
                df = parse_firms_csv(response.text)
                daily_row, raw_df = calculate_daily_features(date_str, df)
                return daily_row, raw_df

            safe_print(f"  API status {response.status_code} on {date_str}. Retrying...")
            retries -= 1
            time.sleep(2)

        except Exception as exc:
            safe_print(f"  Request error on {date_str}: {exc}. Retrying...")
            retries -= 1
            time.sleep(2)

    safe_print(f"  Skipping {date_str} after failed retries.")

    fallback_row = {
        "date": date_str,
        "fire_count": 0,
        "fire_avg_confidence": 0.0,
        "fire_avg_brightness": 0.0,
        "fire_avg_frp": 0.0,
        "fire_min_distance_km": 0.0,
        "fire_pressure": 0.0,
    }

    return fallback_row, pd.DataFrame()


def collect_historical_firms() -> tuple[pd.DataFrame, pd.DataFrame]:
    map_key = get_firms_map_key()

    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

    total_days = (end_date - start_date).days + 1

    safe_print("=" * 80)
    safe_print("NASA FIRMS historical daily collection")
    safe_print("=" * 80)
    safe_print(f"Source: {FIRMS_SOURCE}")
    safe_print(f"Date range: {START_DATE} to {END_DATE}")
    safe_print(f"Total days: {total_days}")
    safe_print(f"Bounding box: {BBOX}")
    safe_print("")

    daily_rows: list[dict[str, Any]] = []
    raw_frames: list[pd.DataFrame] = []

    current_date = start_date
    day_index = 1

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")

        safe_print(f"[{day_index}/{total_days}] Fetching {date_str}")

        daily_row, raw_df = fetch_one_day(map_key, date_str)
        daily_rows.append(daily_row)

        if not raw_df.empty:
            raw_frames.append(raw_df)
            safe_print(
                f"  Fires: {daily_row['fire_count']} | "
                f"Pressure: {daily_row['fire_pressure']}"
            )
        else:
            safe_print("  Fires: 0 | Pressure: 0.0")

        time.sleep(REQUEST_DELAY_SECONDS)

        current_date += timedelta(days=1)
        day_index += 1

    daily_df = pd.DataFrame(daily_rows)

    if raw_frames:
        raw_df = pd.concat(raw_frames, ignore_index=True)
        raw_df = raw_df.drop_duplicates().reset_index(drop=True)
    else:
        raw_df = pd.DataFrame()

    return raw_df, daily_df


def save_outputs(raw_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    ensure_output_folders()

    raw_df.to_csv(RAW_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    daily_df.to_csv(DAILY_OUTPUT_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    raw_df, daily_df = collect_historical_firms()
    save_outputs(raw_df, daily_df)

    safe_print("")
    safe_print("=" * 80)
    safe_print("NASA FIRMS historical processing complete")
    safe_print("=" * 80)
    safe_print(f"Raw output: {RAW_OUTPUT_FILE}")
    safe_print(f"Daily features output: {DAILY_OUTPUT_FILE}")
    safe_print(f"Raw fire hotspot rows: {len(raw_df)}")
    safe_print(f"Daily feature rows: {len(daily_df)}")
    safe_print(f"Days with fire_count > 0: {(daily_df['fire_count'] > 0).sum()}")


if __name__ == "__main__":
    main()
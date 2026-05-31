"""
Collect recent fire hotspot data from NASA FIRMS Area API.

This script fetches recent VIIRS fire detections around the MFU/Chiang Rai area,
saves the raw CSV response, calculates distance from MFU, filters nearby fires,
and appends a daily fire summary row for later PM2.5 feature engineering.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    FIRMS_RAW_DIR,
    FIRE_LOOKBACK_DAYS,
    FIRE_RADIUS_KM,
    MFU_LATITUDE,
    MFU_LONGITUDE,
    NASA_FIRMS_BASE_URL,
    NASA_FIRMS_MAP_KEY,
    RAW_FIRMS_FILE,
    ensure_project_directories,
)


FIRMS_SOURCE = "VIIRS_SNPP_NRT"

# Bounding box around Northern Thailand / nearby cross-border fire sources.
# Format required by FIRMS Area API:
# west,south,east,north
AREA_COORDINATES = "97.0,18.0,102.5,22.5"


def fetch_firms_recent_csv() -> str:
    """
    Fetch recent fire hotspot data from NASA FIRMS Area API.

    Returns:
        CSV text returned by NASA FIRMS.

    Raises:
        ValueError: If NASA FIRMS map key is missing.
        requests.RequestException: If the API request fails.
    """
    if not NASA_FIRMS_MAP_KEY:
        raise ValueError(
            "NASA_FIRMS_MAP_KEY is missing. Please add it to your .env file."
        )

    url = (
        f"{NASA_FIRMS_BASE_URL}/area/csv/"
        f"{NASA_FIRMS_MAP_KEY}/"
        f"{FIRMS_SOURCE}/"
        f"{AREA_COORDINATES}/"
        f"{FIRE_LOOKBACK_DAYS}"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    return response.text


def save_raw_csv(csv_text: str) -> Path:
    """
    Save the raw FIRMS CSV response as a timestamped file.

    Args:
        csv_text: Raw CSV text from FIRMS.

    Returns:
        Path to the saved raw CSV file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = FIRMS_RAW_DIR / f"firms_recent_{timestamp}.csv"

    output_path.write_text(csv_text, encoding="utf-8")

    return output_path


def parse_firms_csv(csv_text: str) -> pd.DataFrame:
    """
    Parse FIRMS CSV text into a DataFrame.

    Args:
        csv_text: Raw CSV text from FIRMS.

    Returns:
        FIRMS fire detection DataFrame.
    """
    if not csv_text.strip():
        return pd.DataFrame()

    df = pd.read_csv(StringIO(csv_text))

    if df.empty:
        return df

    return df


def calculate_haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great-circle distance between two coordinates.

    Args:
        lat1: First latitude.
        lon1: First longitude.
        lat2: Second latitude.
        lon2: Second longitude.

    Returns:
        Distance in kilometers.
    """
    earth_radius_km = 6371.0

    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius_km * c


def add_distance_from_mfu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance from MFU to each fire hotspot row.

    Args:
        df: FIRMS fire detection DataFrame.

    Returns:
        DataFrame with distance_km column.
    """
    if df.empty:
        return df

    required_columns = {"latitude", "longitude"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"FIRMS data missing required columns: {required_columns - set(df.columns)}"
        )

    df = df.copy()

    df["distance_km"] = df.apply(
        lambda row: calculate_haversine_distance_km(
            MFU_LATITUDE,
            MFU_LONGITUDE,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    return df


def filter_nearby_fires(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep fire detections within the configured radius from MFU.

    Args:
        df: FIRMS fire detection DataFrame.

    Returns:
        Nearby fire detections.
    """
    if df.empty:
        return df

    return df[df["distance_km"] <= FIRE_RADIUS_KM].copy()


def calculate_average_confidence(nearby_df: pd.DataFrame) -> float:
    """
    Calculate average fire confidence.

    VIIRS confidence can be categorical:
    - l / low
    - n / nominal
    - h / high

    This function maps those categories into numeric scores. If numeric confidence
    values are provided by another FIRMS source, those values are also supported.

    Args:
        nearby_df: Nearby fire detection DataFrame.

    Returns:
        Average confidence score.
    """
    confidence_column = "confidence"

    if nearby_df.empty or confidence_column not in nearby_df.columns:
        return 0.0

    confidence_mapping = {
        "l": 30,
        "low": 30,
        "n": 60,
        "nominal": 60,
        "h": 90,
        "high": 90,
    }

    confidence_values = nearby_df[confidence_column].astype(str).str.lower().str.strip()

    mapped_confidence = confidence_values.map(confidence_mapping)
    numeric_confidence = pd.to_numeric(nearby_df[confidence_column], errors="coerce")

    combined_confidence = mapped_confidence.fillna(numeric_confidence)

    average_confidence = combined_confidence.mean()

    if pd.isna(average_confidence):
        return 0.0

    return float(average_confidence)


def calculate_average_brightness(nearby_df: pd.DataFrame) -> float:
    """
    Calculate average brightness from available FIRMS brightness columns.

    VIIRS commonly uses bright_ti4, while other sources may provide brightness.

    Args:
        nearby_df: Nearby fire detection DataFrame.

    Returns:
        Average brightness value.
    """
    if nearby_df.empty:
        return 0.0

    possible_brightness_columns = ["bright_ti4", "brightness", "bright_t31"]

    for column in possible_brightness_columns:
        if column in nearby_df.columns:
            average_brightness = pd.to_numeric(
                nearby_df[column],
                errors="coerce",
            ).mean()

            if pd.isna(average_brightness):
                return 0.0

            return float(average_brightness)

    return 0.0


def calculate_fire_pressure(nearby_df: pd.DataFrame) -> float:
    """
    Calculate fire pressure based on distance.

    Nearby fires contribute more pressure than distant fires.

    Formula:
        fire_pressure = sum(1 / (distance_km + 1))

    Args:
        nearby_df: Nearby fire detection DataFrame.

    Returns:
        Fire pressure score.
    """
    if nearby_df.empty:
        return 0.0

    fire_pressure_components = 1 / (nearby_df["distance_km"] + 1)

    return float(fire_pressure_components.sum())


def build_fire_summary(df: pd.DataFrame, nearby_df: pd.DataFrame) -> dict[str, Any]:
    """
    Build one summary row for fire features.

    Args:
        df: All FIRMS detections in the configured area.
        nearby_df: Detections within FIRE_RADIUS_KM from MFU.

    Returns:
        Summary row dictionary.
    """
    collected_at = datetime.now().isoformat(timespec="seconds")

    if nearby_df.empty:
        return {
            "collected_at": collected_at,
            "source": FIRMS_SOURCE,
            "lookback_days": FIRE_LOOKBACK_DAYS,
            "radius_km": FIRE_RADIUS_KM,
            "area_fire_count": len(df),
            "fire_count": 0,
            "fire_avg_confidence": 0.0,
            "fire_avg_brightness": 0.0,
            "fire_min_distance_km": None,
            "fire_pressure": 0.0,
        }

    avg_confidence = calculate_average_confidence(nearby_df)
    avg_brightness = calculate_average_brightness(nearby_df)
    min_distance = float(nearby_df["distance_km"].min())
    fire_pressure = calculate_fire_pressure(nearby_df)

    return {
        "collected_at": collected_at,
        "source": FIRMS_SOURCE,
        "lookback_days": FIRE_LOOKBACK_DAYS,
        "radius_km": FIRE_RADIUS_KM,
        "area_fire_count": len(df),
        "fire_count": len(nearby_df),
        "fire_avg_confidence": avg_confidence,
        "fire_avg_brightness": avg_brightness,
        "fire_min_distance_km": min_distance,
        "fire_pressure": fire_pressure,
    }


def append_summary_to_csv(row: dict[str, Any], output_path: Path) -> None:
    """
    Append one fire summary row to CSV.

    Args:
        row: Fire summary row.
        output_path: Target CSV path.
    """
    df = pd.DataFrame([row])

    if output_path.exists():
        df.to_csv(output_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    """
    Run the NASA FIRMS recent fire data collection workflow.
    """
    ensure_project_directories()

    print("Fetching recent NASA FIRMS fire hotspot data...")
    csv_text = fetch_firms_recent_csv()

    raw_csv_path = save_raw_csv(csv_text)
    print(f"Saved raw CSV: {raw_csv_path}")

    df = parse_firms_csv(csv_text)
    df = add_distance_from_mfu(df)
    nearby_df = filter_nearby_fires(df)

    summary_row = build_fire_summary(df, nearby_df)
    append_summary_to_csv(summary_row, RAW_FIRMS_FILE)

    print("Saved NASA FIRMS fire summary row:")
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print(f"CSV file: {RAW_FIRMS_FILE}")


if __name__ == "__main__":
    main()
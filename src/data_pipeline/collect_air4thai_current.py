"""
Collect current air quality data from the Air4Thai public API.

This script fetches the latest Air4Thai station data, saves the full raw JSON,
extracts the configured station record, and appends a normalized row to CSV.

The script is intentionally defensive because public API response structures
can change over time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    AIR4THAI_CURRENT_API_URL,
    AIR4THAI_RAW_DIR,
    AIR4THAI_STATION_ID,
    RAW_AIR4THAI_FILE,
    ensure_project_directories,
)


def fetch_air4thai_current_data() -> dict[str, Any]:
    """
    Fetch current air quality data from Air4Thai API.

    Returns:
        Dictionary response from the Air4Thai API.

    Raises:
        requests.RequestException: If the API request fails.
        ValueError: If the response is not valid JSON.
    """
    response = requests.get(AIR4THAI_CURRENT_API_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Air4Thai API response is not a JSON object.")

    return data


def save_raw_json(data: dict[str, Any]) -> Path:
    """
    Save the full raw API response as a timestamped JSON file.

    Args:
        data: Raw API response dictionary.

    Returns:
        Path to the saved JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = AIR4THAI_RAW_DIR / f"air4thai_current_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return output_path


def extract_station_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract station records from Air4Thai response.

    Air4Thai commonly returns station data under a "stations" key.
    This function still checks multiple possible structures to avoid failing
    silently if the API format changes.

    Args:
        data: Raw API response dictionary.

    Returns:
        List of station records.
    """
    possible_keys = ["stations", "Stations", "data", "Data"]

    for key in possible_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value

    for value in data.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value

    return []


def find_station_by_id(
    station_records: list[dict[str, Any]],
    station_id: str,
) -> dict[str, Any] | None:
    """
    Find one station record by station ID.

    Args:
        station_records: List of station records.
        station_id: Target Air4Thai station ID.

    Returns:
        Matching station record or None.
    """
    station_id_keys = ["stationID", "stationId", "station_id", "id"]

    for station in station_records:
        for key in station_id_keys:
            if str(station.get(key, "")).strip() == station_id:
                return station

    return None


def get_nested_value(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    """
    Safely get a nested dictionary value.

    Args:
        data: Source dictionary.
        path: List of nested keys.
        default: Value returned when the path does not exist.

    Returns:
        Nested value or default.
    """
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def normalize_station_record(station: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize one Air4Thai station record into a flat row.

    Args:
        station: Raw station dictionary.

    Returns:
        Flat dictionary ready for CSV storage.
    """
    aqi = station.get("AQILast") or station.get("aqiLast") or {}

    row = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "station_id": station.get("stationID") or station.get("stationId"),
        "name_th": station.get("nameTH"),
        "name_en": station.get("nameEN"),
        "area_th": station.get("areaTH"),
        "area_en": station.get("areaEN"),
        "latitude": station.get("lat"),
        "longitude": station.get("long"),
        "date": get_nested_value(aqi, ["date"]),
        "time": get_nested_value(aqi, ["time"]),
        "aqi": get_nested_value(aqi, ["AQI", "aqi"]),
        "aqi_param": get_nested_value(aqi, ["AQI", "param"]),
        "pm25": get_nested_value(aqi, ["PM25", "value"]),
        "pm10": get_nested_value(aqi, ["PM10", "value"]),
        "o3": get_nested_value(aqi, ["O3", "value"]),
        "co": get_nested_value(aqi, ["CO", "value"]),
        "no2": get_nested_value(aqi, ["NO2", "value"]),
        "so2": get_nested_value(aqi, ["SO2", "value"]),
    }

    return row


def append_row_to_csv(row: dict[str, Any], output_path: Path) -> None:
    """
    Append one normalized row to CSV.

    Args:
        row: Normalized station row.
        output_path: Target CSV path.
    """
    df = pd.DataFrame([row])

    if output_path.exists():
        df.to_csv(output_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    """
    Run the Air4Thai current data collection workflow.
    """
    ensure_project_directories()

    print("Fetching current Air4Thai data...")
    data = fetch_air4thai_current_data()

    raw_json_path = save_raw_json(data)
    print(f"Saved raw JSON: {raw_json_path}")

    station_records = extract_station_records(data)

    if not station_records:
        raise ValueError("No station records found in Air4Thai API response.")

    station = find_station_by_id(station_records, AIR4THAI_STATION_ID)

    if station is None:
        available_station_ids = [
            item.get("stationID") or item.get("stationId")
            for item in station_records
        ]

        raise ValueError(
            f"Station ID '{AIR4THAI_STATION_ID}' was not found. "
            f"Available station IDs include: {available_station_ids[:20]}"
        )

    row = normalize_station_record(station)
    append_row_to_csv(row, RAW_AIR4THAI_FILE)

    print("Saved normalized Air4Thai row:")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"CSV file: {RAW_AIR4THAI_FILE}")


if __name__ == "__main__":
    main()
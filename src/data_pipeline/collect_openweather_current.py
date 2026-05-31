"""
Collect current weather data from the OpenWeather Current Weather API.

This script fetches current weather for the configured MFU coordinates,
saves the full raw JSON response, extracts useful weather variables,
and appends one normalized row to CSV.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    MFU_LATITUDE,
    MFU_LONGITUDE,
    OPENWEATHER_API_KEY,
    OPENWEATHER_CURRENT_API_URL,
    RAW_WEATHER_FILE,
    WEATHER_RAW_DIR,
    ensure_project_directories,
)


def fetch_openweather_current_data() -> dict[str, Any]:
    """
    Fetch current weather data from OpenWeather for the configured coordinates.

    Returns:
        Dictionary response from the OpenWeather API.

    Raises:
        ValueError: If the OpenWeather API key is missing.
        requests.RequestException: If the API request fails.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is missing. Please add it to your .env file."
        )

    params = {
        "lat": MFU_LATITUDE,
        "lon": MFU_LONGITUDE,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(
        OPENWEATHER_CURRENT_API_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError("OpenWeather API response is not a JSON object.")

    return data


def save_raw_json(data: dict[str, Any]) -> Path:
    """
    Save the full raw OpenWeather response as a timestamped JSON file.

    Args:
        data: Raw API response dictionary.

    Returns:
        Path to the saved JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = WEATHER_RAW_DIR / f"openweather_current_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return output_path


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


def normalize_weather_record(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize OpenWeather current response into a flat row.

    Args:
        data: Raw OpenWeather API response.

    Returns:
        Flat dictionary ready for CSV storage.
    """
    weather_items = data.get("weather")

    weather_main = None
    weather_description = None

    if isinstance(weather_items, list) and weather_items:
        first_weather = weather_items[0]

        if isinstance(first_weather, dict):
            weather_main = first_weather.get("main")
            weather_description = first_weather.get("description")

    rain_1h = get_nested_value(data, ["rain", "1h"], 0)
    rain_3h = get_nested_value(data, ["rain", "3h"], 0)

    row = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "source_datetime_utc": datetime.utcfromtimestamp(
            data.get("dt", 0)
        ).isoformat(timespec="seconds"),
        "location_name": data.get("name"),
        "latitude": get_nested_value(data, ["coord", "lat"]),
        "longitude": get_nested_value(data, ["coord", "lon"]),
        "temperature": get_nested_value(data, ["main", "temp"]),
        "feels_like": get_nested_value(data, ["main", "feels_like"]),
        "humidity": get_nested_value(data, ["main", "humidity"]),
        "pressure": get_nested_value(data, ["main", "pressure"]),
        "wind_speed": get_nested_value(data, ["wind", "speed"]),
        "wind_direction": get_nested_value(data, ["wind", "deg"]),
        "wind_gust": get_nested_value(data, ["wind", "gust"]),
        "cloudiness": get_nested_value(data, ["clouds", "all"]),
        "precipitation_1h": rain_1h,
        "precipitation_3h": rain_3h,
        "visibility": data.get("visibility"),
        "weather_main": weather_main,
        "weather_description": weather_description,
    }

    return row


def append_row_to_csv(row: dict[str, Any], output_path: Path) -> None:
    """
    Append one normalized row to CSV.

    Args:
        row: Normalized weather row.
        output_path: Target CSV path.
    """
    df = pd.DataFrame([row])

    if output_path.exists():
        df.to_csv(output_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    """
    Run the OpenWeather current data collection workflow.
    """
    ensure_project_directories()

    print("Fetching current OpenWeather data...")
    data = fetch_openweather_current_data()

    raw_json_path = save_raw_json(data)
    print(f"Saved raw JSON: {raw_json_path}")

    row = normalize_weather_record(data)
    append_row_to_csv(row, RAW_WEATHER_FILE)

    print("Saved normalized OpenWeather row:")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"CSV file: {RAW_WEATHER_FILE}")


if __name__ == "__main__":
    main()
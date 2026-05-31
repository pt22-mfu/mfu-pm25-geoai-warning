from pathlib import Path
from datetime import datetime

import pandas as pd

from src.config import TARGET_COLUMN


BASE_DIR = Path(__file__).resolve().parents[2]

AIR4THAI_RAW_PATH = BASE_DIR / "data" / "raw" / "air4thai" / "air4thai_raw.csv"
WEATHER_RAW_PATH = BASE_DIR / "data" / "raw" / "weather" / "weather_raw.csv"
FIRMS_RAW_PATH = BASE_DIR / "data" / "raw" / "firms" / "firms_raw.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "current_feature_snapshot.csv"


def read_latest_row(csv_path: Path, source_name: str) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{source_name} raw CSV file not found: {csv_path}\n"
            f"Please run the related data collector first."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"{source_name} raw CSV file is empty: {csv_path}")

    return df.iloc[-1]


def get_value(row: pd.Series, column_name: str, default=None):
    if column_name not in row.index:
        return default

    value = row[column_name]

    if pd.isna(value):
        return default

    return value


def build_current_snapshot() -> pd.DataFrame:
    air4thai = read_latest_row(AIR4THAI_RAW_PATH, "Air4Thai")
    weather = read_latest_row(WEATHER_RAW_PATH, "OpenWeather")
    firms = read_latest_row(FIRMS_RAW_PATH, "NASA FIRMS")

    snapshot = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),

        "air4thai_collected_at": get_value(air4thai, "collected_at"),
        "weather_collected_at": get_value(weather, "collected_at"),
        "firms_collected_at": get_value(firms, "collected_at"),

        "station_id": get_value(air4thai, "station_id"),
        "station_name_en": get_value(air4thai, "name_en"),

        TARGET_COLUMN: get_value(air4thai, "pm25"),
        "aqi": get_value(air4thai, "aqi"),

        "temperature": get_value(weather, "temperature"),
        "humidity": get_value(weather, "humidity"),
        "pressure": get_value(weather, "pressure"),
        "wind_speed": get_value(weather, "wind_speed"),
        "wind_direction": get_value(weather, "wind_direction"),
        "precipitation": get_value(weather, "precipitation_1h", 0),

        "fire_count": get_value(firms, "fire_count", 0),
        "fire_avg_confidence": get_value(firms, "fire_avg_confidence", 0),
        "fire_avg_brightness": get_value(firms, "fire_avg_brightness", 0),
        "fire_min_distance_km": get_value(firms, "fire_min_distance_km"),
        "fire_pressure": get_value(firms, "fire_pressure", 0),
    }

    return pd.DataFrame([snapshot])


def save_current_snapshot(snapshot_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_df.to_csv(OUTPUT_PATH, index=False)


def main() -> None:
    print("Building current PM2.5 feature snapshot...")

    snapshot_df = build_current_snapshot()
    save_current_snapshot(snapshot_df)

    print("Current feature snapshot created successfully:")
    print(snapshot_df.to_string(index=False))
    print(f"\nSaved CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
"""
Prepare cleaned Chiang Rai PM2.5 and weather dataset.

Input:
- data/raw/pm25/CEI-2561-2565-final-data.csv

Output:
- data/processed/chiang_rai_pm25_weather_2018_2022_clean.csv

This script:
1. Reads the cleaned historical Chiang Rai dataset.
2. Standardizes column names.
3. Validates date range and row count.
4. Checks date continuity.
5. Checks missing values.
6. Creates PM2.5 lag features.
7. Saves the processed dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "pm25" / "CEI-2561-2565-final-data.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chiang_rai_pm25_weather_2018_2022_clean.csv"
)

EXPECTED_START_DATE = "2018-01-01"
EXPECTED_END_DATE = "2022-12-31"
EXPECTED_RAW_ROWS = 1826


COLUMN_RENAME_MAP = {
    "Date": "date",
    "Pressure_max": "pressure_max",
    "Pressure_min": "pressure_min",
    "Pressure_avg": "pressure_avg",
    "Temp_max": "temperature_max",
    "Temp_min": "temperature_min",
    "Temp_avg": "temperature_avg",
    "Humidity_max": "humidity_max",
    "Humidity_min": "humidity_min",
    "Humidity_avg": "humidity_avg",
    "Precipitation": "precipitation",
    "Sunshine": "sunshine",
    "Evaporation": "evaporation",
    "Wind_direct": "wind_direction",
    "Wind_speed": "wind_speed",
    "PM25": "pm25",
}


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("utf-8", errors="replace").decode("utf-8"))


def ensure_output_folder() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please place CEI-2561-2565-final-data.csv in data/raw/pm25/"
        )

    df = pd.read_csv(INPUT_FILE)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in COLUMN_RENAME_MAP.keys() if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df = df.rename(columns=COLUMN_RENAME_MAP)

    expected_columns = list(COLUMN_RENAME_MAP.values())
    df = df[expected_columns]

    return df


def clean_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        bad_rows = df[df["date"].isna()]
        raise ValueError(f"Invalid date values found:\n{bad_rows.head()}")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in df.columns if column != "date"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    safe_print("=" * 80)
    safe_print("Dataset validation")
    safe_print("=" * 80)

    row_count = len(df)
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")

    safe_print(f"Rows: {row_count}")
    safe_print(f"Start date: {start_date}")
    safe_print(f"End date: {end_date}")

    if row_count != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Unexpected row count. Expected {EXPECTED_RAW_ROWS}, got {row_count}"
        )

    if start_date != EXPECTED_START_DATE:
        raise ValueError(
            f"Unexpected start date. Expected {EXPECTED_START_DATE}, got {start_date}"
        )

    if end_date != EXPECTED_END_DATE:
        raise ValueError(
            f"Unexpected end date. Expected {EXPECTED_END_DATE}, got {end_date}"
        )

    duplicate_dates = df[df["date"].duplicated(keep=False)]

    if not duplicate_dates.empty:
        raise ValueError(f"Duplicate dates found:\n{duplicate_dates}")

    expected_dates = pd.date_range(EXPECTED_START_DATE, EXPECTED_END_DATE, freq="D")
    actual_dates = pd.DatetimeIndex(df["date"])

    missing_dates = expected_dates.difference(actual_dates)

    if len(missing_dates) > 0:
        raise ValueError(f"Missing dates found: {missing_dates.tolist()}")

    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(f"Missing values found:\n{missing_values}")

    safe_print("Date continuity: OK")
    safe_print("Duplicate dates: None")
    safe_print("Missing values: None")
    safe_print("Validation passed")


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag2"] = df["pm25"].shift(2)
    df["pm25_lag3"] = df["pm25"].shift(3)

    df = df.dropna(subset=["pm25_lag1", "pm25_lag2", "pm25_lag3"]).reset_index(
        drop=True
    )

    return df


def save_processed_dataset(df: pd.DataFrame) -> None:
    ensure_output_folder()
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    safe_print("Preparing PM2.5 and weather dataset...")

    df = load_dataset()
    df = standardize_columns(df)
    df = clean_date_column(df)
    df = convert_numeric_columns(df)

    validate_dataset(df)

    processed_df = add_lag_features(df)
    save_processed_dataset(processed_df)

    safe_print("")
    safe_print("=" * 80)
    safe_print("Processing complete")
    safe_print("=" * 80)
    safe_print(f"Input file: {INPUT_FILE}")
    safe_print(f"Output file: {OUTPUT_FILE}")
    safe_print(f"Processed rows: {len(processed_df)}")
    safe_print(
        f"Processed date range: "
        f"{processed_df['date'].min().strftime('%Y-%m-%d')} "
        f"to {processed_df['date'].max().strftime('%Y-%m-%d')}"
    )
    safe_print("")
    safe_print("Created lag features:")
    safe_print("- pm25_lag1")
    safe_print("- pm25_lag2")
    safe_print("- pm25_lag3")


if __name__ == "__main__":
    main()
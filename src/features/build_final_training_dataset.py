from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PM25_WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "chiang_rai_pm25_weather_2018_2022_clean.csv"
FIRMS_DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "firms_daily_features_2018_2022.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "final" / "pm25_training_dataset_2018_2022.csv"


def validate_file_exists(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Required input file not found: {file_path}")


def normalize_date_column(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError(f"'date' column is missing in {file_name}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        invalid_count = df["date"].isna().sum()
        raise ValueError(f"{file_name} contains {invalid_count} invalid date values")

    return df


def validate_unique_dates(df: pd.DataFrame, file_name: str) -> None:
    duplicate_count = df["date"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"{file_name} contains {duplicate_count} duplicate date rows")


def create_v7_style_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag2"] = df["pm25"].shift(2)
    df["pm25_lag3"] = df["pm25"].shift(3)
    df["pm25_3day_avg"] = df[["pm25_lag1", "pm25_lag2", "pm25_lag3"]].mean(axis=1)

    df["fire_pressure_lag1"] = df["fire_pressure"].shift(1)
    df["fire_pressure_lag2"] = df["fire_pressure"].shift(2)
    df["fire_pressure_lag3"] = df["fire_pressure"].shift(3)
    df["fire_pressure_3day_avg"] = df[
        ["fire_pressure_lag1", "fire_pressure_lag2", "fire_pressure_lag3"]
    ].mean(axis=1)

    df["month"] = df["date"].dt.month
    df["is_burning_season"] = df["month"].isin([1, 2, 3, 4]).astype(int)

    return df


def main() -> None:
    print("=" * 80)
    print("Building compact v7-style PM2.5 training dataset")
    print("=" * 80)

    validate_file_exists(PM25_WEATHER_PATH)
    validate_file_exists(FIRMS_DAILY_PATH)

    pm25_weather_df = pd.read_csv(PM25_WEATHER_PATH)
    firms_df = pd.read_csv(FIRMS_DAILY_PATH)

    pm25_weather_df = normalize_date_column(pm25_weather_df, "PM2.5/weather dataset")
    firms_df = normalize_date_column(firms_df, "FIRMS daily features dataset")

    validate_unique_dates(pm25_weather_df, "PM2.5/weather dataset")
    validate_unique_dates(firms_df, "FIRMS daily features dataset")

    merged_df = pm25_weather_df.merge(
        firms_df,
        on="date",
        how="left",
        validate="one_to_one",
    )

    fire_columns = [
        "fire_count",
        "fire_avg_confidence",
        "fire_avg_brightness",
        "fire_avg_frp",
        "fire_min_distance_km",
        "fire_pressure",
    ]

    for column in fire_columns:
        if column in merged_df.columns:
            merged_df[column] = merged_df[column].fillna(0)

    merged_df = create_v7_style_features(merged_df)

    selected_columns = [
        "date",
        "pressure_avg",
        "temperature_avg",
        "humidity_avg",
        "precipitation",
        "sunshine",
        "wind_direction",
        "wind_speed",
        "pm25_lag1",
        "pm25_lag2",
        "pm25_lag3",
        "pm25_3day_avg",
        "fire_count",
        "fire_pressure",
        "fire_pressure_lag1",
        "fire_pressure_lag2",
        "fire_pressure_3day_avg",
        "month",
        "is_burning_season",
        "pm25",
    ]

    missing_columns = [
        column for column in selected_columns if column not in merged_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing selected columns: {missing_columns}")

    final_df = merged_df[selected_columns].copy()

    before_drop = len(final_df)
    final_df = final_df.dropna().reset_index(drop=True)
    after_drop = len(final_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nCompact final dataset created successfully")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {final_df.shape}")
    print(f"Feature count: {len(selected_columns) - 2}")
    print(f"Rows dropped due to lag features: {before_drop - after_drop}")
    print(f"Date range: {final_df['date'].min().date()} to {final_df['date'].max().date()}")
    print(f"Missing values: {final_df.isna().sum().sum()}")
    print(f"Duplicate dates: {final_df['date'].duplicated().sum()}")

    print("\nColumns:")
    for column in final_df.columns:
        print(f"- {column}")

    print("\nPreview:")
    print(final_df.head())


if __name__ == "__main__":
    main()
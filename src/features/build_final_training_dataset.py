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

    df["date"] = df["date"].dt.date
    return df


def validate_unique_dates(df: pd.DataFrame, file_name: str) -> None:
    duplicate_count = df["date"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"{file_name} contains {duplicate_count} duplicate date rows")


def create_fire_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    if "fire_pressure" not in df.columns:
        raise ValueError("'fire_pressure' column is missing from merged dataset")

    df["fire_pressure_lag1"] = df["fire_pressure"].shift(1)
    df["fire_pressure_lag2"] = df["fire_pressure"].shift(2)
    df["fire_pressure_lag3"] = df["fire_pressure"].shift(3)

    return df


def main() -> None:
    print("=" * 80)
    print("Building final PM2.5 training dataset")
    print("=" * 80)

    validate_file_exists(PM25_WEATHER_PATH)
    validate_file_exists(FIRMS_DAILY_PATH)

    pm25_weather_df = pd.read_csv(PM25_WEATHER_PATH)
    firms_df = pd.read_csv(FIRMS_DAILY_PATH)

    pm25_weather_df = normalize_date_column(pm25_weather_df, "PM2.5/weather dataset")
    firms_df = normalize_date_column(firms_df, "FIRMS daily features dataset")

    validate_unique_dates(pm25_weather_df, "PM2.5/weather dataset")
    validate_unique_dates(firms_df, "FIRMS daily features dataset")

    print(f"PM2.5/weather shape: {pm25_weather_df.shape}")
    print(f"FIRMS daily shape: {firms_df.shape}")

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
        "fire_min_distance_km",
        "fire_pressure",
    ]

    for column in fire_columns:
        if column in merged_df.columns:
            merged_df[column] = merged_df[column].fillna(0)

    merged_df = create_fire_lag_features(merged_df)

    before_drop = len(merged_df)
    merged_df = merged_df.dropna().reset_index(drop=True)
    after_drop = len(merged_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal dataset created successfully")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {merged_df.shape}")
    print(f"Rows dropped due to lag features: {before_drop - after_drop}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Missing values: {merged_df.isna().sum().sum()}")
    print(f"Duplicate dates: {merged_df['date'].duplicated().sum()}")

    print("\nColumns:")
    for column in merged_df.columns:
        print(f"- {column}")

    print("\nPreview:")
    print(merged_df.head())


if __name__ == "__main__":
    main()
"""Train and compare weather-only and fire-integrated PM2.5 models.

This experiment is a controlled ablation study for the SP2 report. For each
algorithm, the weather-only and fire-integrated variants use the same dataset,
chronological split, target transformation, and hyperparameters. The only
difference is whether NASA FIRMS-derived features are included.

Default input:
    data/final/pm25_training_dataset_2018_2022.csv

Default outputs:
    reports/ablation/ablation_metrics.csv
    reports/ablation/test_predictions.csv
    reports/ablation/experiment_config.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "final" / "pm25_training_dataset_2018_2022.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "ablation"

TARGET_COLUMN = "pm25"
DATE_COLUMN = "date"
DEFAULT_TEST_YEAR = 2022
RANDOM_STATE = 42
BURNING_SEASON_MONTHS = (1, 2, 3, 4)
HIGH_PM25_THRESHOLD = 50.0


# A strong weather-only baseline retains the PM2.5 lag and seasonality features.
# The controlled fire-integrated variant adds only NASA FIRMS-derived variables.
WEATHER_FEATURES = [
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
    "month",
    "is_burning_season",
]

FIRE_FEATURES = [
    "fire_count",
    "fire_pressure",
    "fire_pressure_lag1",
    "fire_pressure_lag2",
    "fire_pressure_3day_avg",
]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    output_directory: str
    test_year: int
    training_rule: str
    test_rule: str
    target: str
    target_transformation: str
    burning_season_months: tuple[int, ...]
    high_pm25_threshold: float
    random_state: int
    weather_features: list[str]
    fire_features: list[str]
    environment: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare weather-only and NASA fire-integrated PM2.5 models using "
            "a chronological holdout."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Training dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Experiment output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=DEFAULT_TEST_YEAR,
        help=f"Calendar year used as the holdout test set (default: {DEFAULT_TEST_YEAR})",
    )
    return parser.parse_args()


def load_and_validate_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    required_columns = {
        DATE_COLUMN,
        TARGET_COLUMN,
        *WEATHER_FEATURES,
        *FIRE_FEATURES,
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if df[DATE_COLUMN].isna().any():
        invalid_count = int(df[DATE_COLUMN].isna().sum())
        raise ValueError(f"Dataset contains {invalid_count} invalid date values")

    if df[DATE_COLUMN].duplicated().any():
        duplicate_count = int(df[DATE_COLUMN].duplicated().sum())
        raise ValueError(f"Dataset contains {duplicate_count} duplicate dates")

    numeric_columns = [TARGET_COLUMN, *WEATHER_FEATURES, *FIRE_FEATURES]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    missing_value_count = int(df[numeric_columns].isna().sum().sum())
    if missing_value_count:
        raise ValueError(
            f"Required numeric fields contain {missing_value_count} missing or invalid values"
        )

    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError("PM2.5 target contains negative values")

    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[DATE_COLUMN].dt.year < test_year].copy()
    test_df = df[df[DATE_COLUMN].dt.year == test_year].copy()

    if train_df.empty:
        raise ValueError(f"No training rows exist before test year {test_year}")
    if test_df.empty:
        raise ValueError(f"No test rows exist for test year {test_year}")
    if train_df[DATE_COLUMN].max() >= test_df[DATE_COLUMN].min():
        raise ValueError("Chronological split validation failed")

    return train_df, test_df


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "lightgbm": lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgboost": lambda: XGBRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    if len(actual) == 0:
        raise ValueError("Cannot calculate metrics for an empty evaluation subset")

    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def evaluation_scopes(test_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "full_test_year": pd.Series(True, index=test_df.index),
        "burning_season": test_df[DATE_COLUMN].dt.month.isin(BURNING_SEASON_MONTHS),
        "high_pm25_days": test_df[TARGET_COLUMN] > HIGH_PM25_THRESHOLD,
    }


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_train_log = np.log1p(train_df[TARGET_COLUMN].to_numpy())
    target_test = test_df[TARGET_COLUMN].to_numpy()

    feature_sets = {
        "weather_only": WEATHER_FEATURES,
        "weather_plus_fire": [*WEATHER_FEATURES, *FIRE_FEATURES],
    }
    scopes = evaluation_scopes(test_df)

    metric_rows: list[dict[str, object]] = []
    prediction_output = test_df[[DATE_COLUMN, TARGET_COLUMN, "month"]].copy()

    for algorithm, factory in model_factories().items():
        for variant, feature_columns in feature_sets.items():
            model = factory()
            model.fit(train_df[feature_columns], target_train_log)

            predicted_log = model.predict(test_df[feature_columns])
            predicted = np.clip(np.expm1(predicted_log), a_min=0, a_max=None)
            prediction_column = f"{algorithm}_{variant}"
            prediction_output[prediction_column] = predicted

            for scope_name, scope_mask in scopes.items():
                mask = scope_mask.to_numpy(dtype=bool)
                metrics = calculate_metrics(target_test[mask], predicted[mask])
                metric_rows.append(
                    {
                        "algorithm": algorithm,
                        "variant": variant,
                        "scope": scope_name,
                        "feature_count": len(feature_columns),
                        "sample_count": int(mask.sum()),
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["scope", "algorithm", "variant"]
    )
    return metrics_df.reset_index(drop=True), prediction_output


def build_ablation_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the change produced by adding fire features to each algorithm."""
    weather = metrics_df[metrics_df["variant"] == "weather_only"].set_index(
        ["algorithm", "scope"]
    )
    fire = metrics_df[metrics_df["variant"] == "weather_plus_fire"].set_index(
        ["algorithm", "scope"]
    )

    summary = pd.DataFrame(index=weather.index)
    summary["sample_count"] = fire["sample_count"]
    summary["r2_weather_only"] = weather["r2"]
    summary["r2_weather_plus_fire"] = fire["r2"]
    summary["r2_improvement"] = fire["r2"] - weather["r2"]
    summary["mae_weather_only"] = weather["mae"]
    summary["mae_weather_plus_fire"] = fire["mae"]
    summary["mae_reduction"] = weather["mae"] - fire["mae"]
    summary["rmse_weather_only"] = weather["rmse"]
    summary["rmse_weather_plus_fire"] = fire["rmse"]
    summary["rmse_reduction"] = weather["rmse"] - fire["rmse"]
    return summary.reset_index().sort_values(["scope", "algorithm"])


def get_environment_versions() -> dict[str, str]:
    package_names = ["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost"]
    versions = {"python": platform.python_version()}
    for package_name in package_names:
        versions[package_name] = importlib.metadata.version(package_name)
    return versions


def save_outputs(
    metrics_df: pd.DataFrame,
    ablation_summary_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "ablation_metrics.csv", index=False)
    ablation_summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)


def print_summary(
    dataset: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    ablation_summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("=" * 88)
    print("SP2 CONTROLLED ABLATION STUDY")
    print("=" * 88)
    print(f"Dataset rows: {len(dataset)}")
    print(
        f"Dataset period: {dataset[DATE_COLUMN].min().date()} to "
        f"{dataset[DATE_COLUMN].max().date()}"
    )
    print(
        f"Training: {len(train_df)} rows, {train_df[DATE_COLUMN].min().date()} to "
        f"{train_df[DATE_COLUMN].max().date()}"
    )
    print(
        f"Testing:  {len(test_df)} rows, {test_df[DATE_COLUMN].min().date()} to "
        f"{test_df[DATE_COLUMN].max().date()}"
    )
    print()
    display_columns = [
        "scope",
        "algorithm",
        "variant",
        "sample_count",
        "r2",
        "mae",
        "rmse",
    ]
    printable = metrics_df[display_columns].copy()
    printable[["r2", "mae", "rmse"]] = printable[["r2", "mae", "rmse"]].round(4)
    print(printable.to_string(index=False))
    print()
    print("Effect of adding NASA fire features (positive values indicate improvement):")
    delta_columns = [
        "scope",
        "algorithm",
        "r2_improvement",
        "mae_reduction",
        "rmse_reduction",
    ]
    deltas = ablation_summary_df[delta_columns].copy()
    deltas[["r2_improvement", "mae_reduction", "rmse_reduction"]] = deltas[
        ["r2_improvement", "mae_reduction", "rmse_reduction"]
    ].round(4)
    print(deltas.to_string(index=False))
    print()
    print(f"Saved experiment outputs to: {output_dir}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()

    dataset = load_and_validate_dataset(dataset_path)
    train_df, test_df = chronological_split(dataset, args.test_year)
    metrics_df, predictions_df = run_experiment(train_df, test_df)
    ablation_summary_df = build_ablation_summary(metrics_df)

    config = ExperimentConfig(
        dataset=str(dataset_path),
        output_directory=str(output_dir),
        test_year=args.test_year,
        training_rule=f"date year < {args.test_year}",
        test_rule=f"date year == {args.test_year}",
        target=TARGET_COLUMN,
        target_transformation="log1p for training; expm1 for evaluation",
        burning_season_months=BURNING_SEASON_MONTHS,
        high_pm25_threshold=HIGH_PM25_THRESHOLD,
        random_state=RANDOM_STATE,
        weather_features=WEATHER_FEATURES,
        fire_features=FIRE_FEATURES,
        environment=get_environment_versions(),
    )
    save_outputs(
        metrics_df,
        ablation_summary_df,
        predictions_df,
        config,
        output_dir,
    )
    print_summary(
        dataset,
        train_df,
        test_df,
        metrics_df,
        ablation_summary_df,
        output_dir,
    )


if __name__ == "__main__":
    main()

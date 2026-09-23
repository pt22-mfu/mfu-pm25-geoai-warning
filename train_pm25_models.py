"""Train and compare weather-only and fire-integrated PM2.5 models.

This experiment is a controlled ablation study for the SP2 report. For each
algorithm, the weather-only and fire-integrated variants use the same dataset,
chronological split, target transformation, and hyperparameters. The only
difference is whether NASA FIRMS-derived features are included.

Default input:
    data/final/pm25_training_dataset_2018_2022.csv

Default outputs:
    reports/ablation/ablation_metrics.csv
    reports/ablation/ablation_summary.csv
    reports/ablation/test_predictions.csv
    reports/ablation/experiment_config.json
    reports/ablation/ablation_mae_comparison.png
    reports/ablation/ablation_rmse_comparison.png
    reports/ablation/high_pm25_predictions.png
"""

from __future__ import annotations

import argparse
import hashlib
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
    dataset_sha256: str
    dataset_rows: int
    dataset_period: str
    output_directory: str
    test_year: int
    training_rows: int
    testing_rows: int
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
    package_names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "matplotlib",
    ]
    versions = {"python": platform.python_version()}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not installed"
    return versions


def portable_path(path: Path) -> str:
    """Use a repository-relative path when the path is inside the project."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Return a stable fingerprint so the exact experiment dataset is known."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Chart generation requires matplotlib. Install it with: "
            "python -m pip install matplotlib"
        ) from exc
    return plt


def save_comparison_chart(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Save a report-ready weather-only versus fire-feature comparison."""
    plt = import_pyplot()
    scope_order = ["full_test_year", "burning_season", "high_pm25_days"]
    scope_labels = ["Full test year", "Burning season", "High-PM2.5 days"]
    algorithms = ["lightgbm", "xgboost"]
    colors = {"weather_only": "#9CA3AF", "weather_plus_fire": "#E4572E"}
    labels = {"weather_only": "Weather only", "weather_plus_fire": "Weather + fire"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    positions = np.arange(len(scope_order))
    width = 0.36

    for axis, algorithm in zip(axes, algorithms):
        algorithm_rows = metrics_df[metrics_df["algorithm"] == algorithm]
        for offset, variant in zip((-width / 2, width / 2), colors):
            variant_rows = algorithm_rows[
                algorithm_rows["variant"] == variant
            ].set_index("scope")
            values = [variant_rows.loc[scope, metric] for scope in scope_order]
            bars = axis.bar(
                positions + offset,
                values,
                width,
                label=labels[variant],
                color=colors[variant],
            )
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

        axis.set_title(algorithm.upper())
        axis.set_xticks(positions, scope_labels, rotation=15, ha="right")
        axis.set_ylabel(f"{metric.upper()} (µg/m³; lower is better)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"PM2.5 Ablation Study: {metric.upper()} Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_high_pm25_chart(predictions_df: pd.DataFrame, output_path: Path) -> None:
    """Visualize predictions on the high-pollution evaluation subset."""
    plt = import_pyplot()
    high_days = predictions_df[
        predictions_df[TARGET_COLUMN] > HIGH_PM25_THRESHOLD
    ].sort_values(DATE_COLUMN)
    x = np.arange(len(high_days))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for axis, algorithm in zip(axes, ("lightgbm", "xgboost")):
        axis.plot(
            x,
            high_days[TARGET_COLUMN],
            marker="o",
            linewidth=2,
            color="#111827",
            label="Actual PM2.5",
        )
        axis.plot(
            x,
            high_days[f"{algorithm}_weather_only"],
            marker="s",
            linewidth=1.5,
            color="#9CA3AF",
            label="Weather only",
        )
        axis.plot(
            x,
            high_days[f"{algorithm}_weather_plus_fire"],
            marker="^",
            linewidth=1.5,
            color="#E4572E",
            label="Weather + fire",
        )
        axis.set_title(algorithm.upper())
        axis.set_ylabel("PM2.5 (µg/m³)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=3)

    axes[-1].set_xticks(
        x,
        high_days[DATE_COLUMN].dt.strftime("%d %b").tolist(),
        rotation=55,
        ha="right",
    )
    axes[-1].set_xlabel("2022 high-PM2.5 test days (actual PM2.5 > 50 µg/m³)")
    fig.suptitle(
        "Predictions During High-Pollution Conditions",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    save_comparison_chart(
        metrics_df,
        "mae",
        output_dir / "ablation_mae_comparison.png",
    )
    save_comparison_chart(
        metrics_df,
        "rmse",
        output_dir / "ablation_rmse_comparison.png",
    )
    save_high_pm25_chart(
        predictions_df,
        output_dir / "high_pm25_predictions.png",
    )


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
        dataset=portable_path(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        dataset_rows=len(dataset),
        dataset_period=(
            f"{dataset[DATE_COLUMN].min().date()} to "
            f"{dataset[DATE_COLUMN].max().date()}"
        ),
        output_directory=portable_path(output_dir),
        test_year=args.test_year,
        training_rows=len(train_df),
        testing_rows=len(test_df),
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

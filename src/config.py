"""
Project configuration constants for the MFU PM2.5 GeoAI Warning System.

This file centralizes paths, station settings, API settings, model settings,
and feature definitions so the rest of the project can reuse the same values.
"""

from pathlib import Path
from dotenv import load_dotenv
import os


# =========================
# Base Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

AIR4THAI_RAW_DIR = RAW_DATA_DIR / "air4thai"
WEATHER_RAW_DIR = RAW_DATA_DIR / "weather"
FIRMS_RAW_DIR = RAW_DATA_DIR / "firms"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# =========================
# Environment Variables
# =========================

load_dotenv(PROJECT_ROOT / ".env")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NASA_FIRMS_MAP_KEY = os.getenv("NASA_FIRMS_MAP_KEY")


# =========================
# Location Settings
# =========================

PROJECT_LOCATION_NAME = "Mae Fah Luang University"
PROJECT_PROVINCE = "Chiang Rai"
PROJECT_COUNTRY = "Thailand"

MFU_LATITUDE = 20.0443
MFU_LONGITUDE = 99.8924

# Air4Thai station near Chiang Rai / MFU area.
# This can be adjusted later after we verify the best station for the final dataset.
AIR4THAI_STATION_ID = "73t"


# =========================
# API Settings
# =========================

AIR4THAI_CURRENT_API_URL = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"

OPENWEATHER_CURRENT_API_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_API_URL = "https://api.openweathermap.org/data/2.5/forecast"

NASA_FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"


# =========================
# Dataset Scope
# =========================

START_YEAR = 2020
END_YEAR = 2025

TARGET_COLUMN = "pm25"


# =========================
# Fire Feature Settings
# =========================

FIRE_RADIUS_KM = 100

# Days used for fire pressure aggregation.
FIRE_LOOKBACK_DAYS = 3


# =========================
# Model Settings
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_CANDIDATES = [
    "random_forest",
    "svr",
    "xgboost",
    "lightgbm",
]

CHAMPION_MODEL_CANDIDATES = [
    "xgboost",
    "lightgbm",
]


# =========================
# Feature Columns
# =========================

BASE_WEATHER_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "precipitation",
]

LAG_FEATURES = [
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
]

FIRE_FEATURES = [
    "fire_count",
    "fire_avg_confidence",
    "fire_avg_brightness",
    "fire_pressure",
    "fire_pressure_lag1",
    "fire_pressure_lag2",
    "fire_pressure_lag3",
]

FEATURE_COLUMNS = BASE_WEATHER_FEATURES + LAG_FEATURES + FIRE_FEATURES


# =========================
# Output File Paths
# =========================

RAW_AIR4THAI_FILE = AIR4THAI_RAW_DIR / "air4thai_raw.csv"
RAW_WEATHER_FILE = WEATHER_RAW_DIR / "weather_raw.csv"
RAW_FIRMS_FILE = FIRMS_RAW_DIR / "firms_raw.csv"

PROCESSED_AIR4THAI_FILE = PROCESSED_DATA_DIR / "air4thai_processed.csv"
PROCESSED_WEATHER_FILE = PROCESSED_DATA_DIR / "weather_processed.csv"
PROCESSED_FIRMS_FILE = PROCESSED_DATA_DIR / "firms_processed.csv"

FINAL_DATASET_FILE = FINAL_DATA_DIR / "mfu_pm25_final_dataset.csv"

MODEL_COMPARISON_REPORT = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_pm25_model.pkl"


# =========================
# Utility
# =========================

def ensure_project_directories() -> None:
    """
    Create required project directories if they do not already exist.
    This is safe to run multiple times.
    """
    directories = [
        AIR4THAI_RAW_DIR,
        WEATHER_RAW_DIR,
        FIRMS_RAW_DIR,
        PROCESSED_DATA_DIR,
        FINAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
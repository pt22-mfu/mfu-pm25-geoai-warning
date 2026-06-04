import os
import math
import json
import base64
import textwrap
from io import StringIO
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import google.generativeai as genai
except Exception:
    genai = None


# =============================================================================
# MFU PM2.5 GEOAI WARNING DASHBOARD
# Senior Project SP2 Dashboard
# Theme: MFU Brand Identity
# =============================================================================

st.set_page_config(
    page_title="MFU PM2.5 GeoAI Warning",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# THEME
# =============================================================================

# Old-money professional dashboard theme
NAVY_BLUE = "#0f172a"
ROYAL_BLUE = "#1e3a8a"
BLUE_SOFT = "#eff6ff"
BG_LIGHT = "#f4f7f9"
SURFACE_WHITE = "#ffffff"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"
BORDER = "#dbe3ea"
BORDER_STRONG = "#cbd5e1"
SLATE = "#64748b"

# Keep the original variable names as aliases so the prediction logic remains stable.
MFU_RED = ROYAL_BLUE
MFU_GOLD = "#64748b"
WARM_WHITE = BG_LIGHT

# Risk colors. Red is intentionally reserved for unhealthy or urgent states only.
GOOD = "#15803d"
MODERATE = "#b7791f"
UNHEALTHY = "#dc2626"
HAZARDOUS = "#7e22ce"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(30, 58, 138, 0.055), transparent 30%),
        radial-gradient(circle at top right, rgba(15, 23, 42, 0.045), transparent 24%),
        linear-gradient(180deg, {BG_LIGHT} 0%, #eef3f8 100%);
    color: {TEXT_DARK};
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2.6rem;
    max-width: 1500px;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY_BLUE} 0%, #172033 100%);
}}

[data-testid="stSidebar"] * {{
    color: #ffffff !important;
}}

[data-testid="stMetric"] {{
    background: {SURFACE_WHITE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}}

[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.80rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

[data-testid="stMetricValue"] {{
    color: {NAVY_BLUE} !important;
    font-weight: 900 !important;
    letter-spacing: -0.04em;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 7px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.66rem 1rem;
    font-weight: 800;
    color: {TEXT_MUTED};
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, {NAVY_BLUE}, {ROYAL_BLUE});
    color: white !important;
    box-shadow: 0 7px 16px rgba(30, 58, 138, 0.18);
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] p {{
    color: white !important;
}}

/* Force Streamlit widget labels (like selectbox) to remain dark and readable */
div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label {{
    color: {TEXT_DARK} !important;
    font-weight: 750 !important;
}}

div.stButton > button:first-child {{
    background: linear-gradient(135deg, {NAVY_BLUE}, {ROYAL_BLUE});
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.70rem 1.1rem;
    font-weight: 900;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.16);
}}

div.stButton > button:first-child:hover {{
    filter: brightness(1.06);
    transform: translateY(-1px);
}}

.glass-card, .professional-card {{
    background: {SURFACE_WHITE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
    color: {TEXT_DARK};
}}

.hero-card {{
    border-left: 6px solid {ROYAL_BLUE};
}}

.gold-card, .blue-card {{
    border-top: 5px solid {ROYAL_BLUE};
}}

.red-card {{
    border-top: 5px solid {UNHEALTHY};
}}

.model-card {{
    background: {SURFACE_WHITE};
    border: 1px solid {BORDER};
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
}}

.model-card.champion {{
    border-top: 5px solid {ROYAL_BLUE};
    background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
}}

.model-card.contender {{
    border-top: 5px solid #334155;
}}

.model-card.baseline {{
    border-top: 5px solid #94a3b8;
}}

.header-wrap {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 22px;
    padding: 18px 20px;
    border: 1px solid {BORDER};
    border-radius: 18px;
    background: rgba(255,255,255,0.76);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
}}

.logo-img {{
    width: 74px;
    height: auto;
    filter: drop-shadow(0px 4px 8px rgba(15,23,42,0.12));
}}

.header-title {{
    font-size: 32px;
    font-weight: 900;
    color: {NAVY_BLUE};
    margin: 0;
    letter-spacing: -0.75px;
}}

.header-subtitle {{
    font-size: 15px;
    color: {TEXT_MUTED};
    font-weight: 700;
    margin: 5px 0 0 0;
}}

.section-title {{
    color: {NAVY_BLUE};
    font-size: 1.25rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 0.5rem 0 0.85rem 0;
}}

.small-muted {{
    color: {TEXT_MUTED};
    font-weight: 700;
    font-size: 0.88rem;
}}

.status-pill {{
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
}}

.warning-box {{
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 14px;
    padding: 16px;
}}

.success-box {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 16px;
}}

.info-box {{
    background: {BLUE_SOFT};
    border: 1px solid #bfdbfe;
    border-radius: 14px;
    padding: 16px;
}}

.footer-note {{
    margin-top: 34px;
    padding: 16px 0 4px 0;
    border-top: 1px solid {BORDER};
    text-align: center;
    color: {TEXT_MUTED};
    font-size: 13px;
    font-weight: 700;
}}



/* Hide Streamlit's default top toolbar/header to remove the black top bar. */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {{
    display: none !important;
    height: 0 !important;
}}

.weather-mini-card {{
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #dbe3ea;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
}}

.weather-mini-label {{
    color: #475569;
    font-size: 0.78rem;
    font-weight: 900;
    letter-spacing: 0.055em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

.weather-mini-value {{
    color: #0f172a;
    font-size: 1.45rem;
    font-weight: 900;
    letter-spacing: -0.035em;
}}

.weather-mini-subtext {{
    color: #64748b;
    font-size: 0.84rem;
    font-weight: 700;
    margin-top: 4px;
}}

div[data-testid="stAlert"] {{
    border-radius: 14px;
}}

a {{
    color: {ROYAL_BLUE};
}}

.clean-factor-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-top: 16px;
}}

.clean-factor-card {{
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #dbe3ea;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
}}

.clean-label {{
    color: #475569;
    font-size: 0.76rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 7px;
}}

.clean-value {{
    color: #0f172a;
    font-size: 1.42rem;
    font-weight: 900;
    letter-spacing: -0.035em;
    line-height: 1.08;
}}

.clean-subtext {{
    color: #64748b;
    font-size: 0.83rem;
    font-weight: 700;
    margin-top: 5px;
    line-height: 1.35;
}}

.model-light-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    overflow: hidden;
    border: 1px solid #dbe3ea;
    border-radius: 16px;
    background: white;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
}}

.model-light-table th {{
    background: #f8fafc;
    color: #334155;
    text-align: left;
    font-size: 0.78rem;
    letter-spacing: 0.055em;
    text-transform: uppercase;
    padding: 14px 16px;
    border-bottom: 1px solid #e2e8f0;
}}

.model-light-table td {{
    color: #0f172a;
    font-size: 0.93rem;
    font-weight: 750;
    padding: 13px 16px;
    border-bottom: 1px solid #edf2f7;
}}

.model-light-table tr:last-child td {{
    border-bottom: none;
}}

.reading-card {{
    background: #ffffff;
    border: 1px solid #dbe3ea;
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}}

.reading-row {{
    padding: 12px 0;
    border-bottom: 1px solid #edf2f7;
}}

.reading-row:last-child {{
    border-bottom: none;
}}

.advisory-shell {{
    background: #ffffff;
    border: 1px solid #dbe3ea;
    border-radius: 18px;
    padding: 24px 28px;
    line-height: 1.75;
    color: #0f172a;
    font-size: 1rem;
    font-weight: 650;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
}}

.setup-card {{
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 16px;
    padding: 18px 20px;
    color: #7c2d12;
    font-weight: 750;
    line-height: 1.55;
}}

@media (max-width: 1100px) {{
    .clean-factor-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

</style>
""",
    unsafe_allow_html=True,
)

FOOTER_HTML = f"""
<div class="footer-note">
    <span style="color:{ROYAL_BLUE}; font-weight:900;">The Outliers</span>
    <span> · CPE Senior Project</span>
    <span style="margin:0 10px; color:{BORDER_STRONG};">|</span>
    <span style="color:{NAVY_BLUE}; font-weight:900;">Advisor - Aj. Khwunta Kirimasthong</span>
</div>
"""

# =============================================================================
# PATHS / CONFIG
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")

MFU_LAT = 20.045
MFU_LON = 99.895
AIR4THAI_STATION_ID = "73t"
NASA_BBOX = "99.4,19.6,100.4,20.5"

FEATURE_COLS_18 = [
    "Pressure_avg",
    "Temp_avg",
    "Humidity_avg",
    "Precipitation",
    "Sunshine",
    "Wind_direct",
    "Wind_speed",
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_3Day_Avg",
    "Fire_Count",
    "Fire_Pressure",
    "Fire_Pressure_Lag1",
    "Fire_Pressure_Lag2",
    "Fire_Pressure_3Day_Avg",
    "Month",
    "Is_Burning_Season",
]

FEATURE_COLS_7 = [
    "Pressure_avg",
    "Temp_avg",
    "Humidity_avg",
    "Precipitation",
    "Sunshine",
    "Wind_direct",
    "Wind_speed",
]

MODEL_METRICS = pd.DataFrame(
    [
        {
            "model": "LightGBM Fire-Integrated",
            "role": "New SP2 Champion",
            "features": 18,
            "r2": 0.8590,
            "mae": 3.2050,
            "rmse": 4.7760,
        },
        {
            "model": "XGBoost Fire-Integrated",
            "role": "Strong Contender",
            "features": 18,
            "r2": 0.8503,
            "mae": 3.1928,
            "rmse": 4.9207,
        },
        {
            "model": "SVR Weather Baseline",
            "role": "Non-linear Baseline",
            "features": 7,
            "r2": 0.2273,
            "mae": 7.1350,
            "rmse": 11.1791,
        },
        {
            "model": "MLR Weather Baseline",
            "role": "Linear Baseline",
            "features": 7,
            "r2": -0.3255,
            "mae": 9.3503,
            "rmse": 14.6420,
        },
    ]
)


# =============================================================================
# SECRETS
# =============================================================================

def get_secret(name: str, fallback: str = "") -> str:
    try:
        return st.secrets.get(name, fallback)
    except Exception:
        return fallback


OPENWEATHER_API_KEY = get_secret("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_KEY")
NASA_KEY = get_secret("NASA_KEY", "YOUR_NASA_KEY")
GISTDA_KEY = get_secret("GISTDA_KEY", "YOUR_GISTDA_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "")


def render_html(markup: str):
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def safe_load_model(filename: str):
    paths = [
        os.path.join(MODEL_DIR, filename),
        os.path.join(SCRIPT_DIR, filename),
        filename,
    ]

    for path in paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as exc:
                st.sidebar.warning(f"Could not load {filename}: {exc}")

    return None


@st.cache_resource
def load_models() -> dict:
    return {
        "lgbm_fire": safe_load_model("lgbm_pm25_model.pkl"),
        "xgb_fire": safe_load_model("pm25_model_v7.pkl"),
        "svr": safe_load_model("svr_pm25_model.pkl"),
        "mlr": safe_load_model("mlr_pm25_model.pkl"),
    }


def get_base64_img(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as file:
            data = file.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return ""


def get_risk_label(value: float):
    if value <= 25:
        return "Good", GOOD, "🟢"
    if value <= 50:
        return "Moderate", MODERATE, "🟡"
    if value <= 100:
        return "Unhealthy", UNHEALTHY, "🟠"
    return "Hazardous", HAZARDOUS, "🔴"


def health_message(value: float) -> str:
    if value <= 25:
        return "Air quality is generally acceptable for outdoor activities."
    if value <= 50:
        return "Sensitive groups should reduce long outdoor exposure."
    if value <= 100:
        return "Outdoor activity should be reduced. Masks are recommended."
    return "Avoid outdoor activity. Consider indoor air filtration and medical advice if symptoms occur."


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def apply_plot_style(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT_DARK, size=13),
        title_font=dict(family="Inter", color=TEXT_DARK, size=19),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(140, 21, 21, 0.10)", linecolor=BORDER)
    return fig


def normalize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "Date",
        "PM25": "PM25",
        "pm25": "PM25",
        "PM2.5": "PM25",
        "pressure_avg": "Pressure_avg",
        "temperature_avg": "Temp_avg",
        "temp_avg": "Temp_avg",
        "humidity_avg": "Humidity_avg",
        "precipitation": "Precipitation",
        "sunshine": "Sunshine",
        "wind_direction": "Wind_direct",
        "wind_direct": "Wind_direct",
        "wind_speed": "Wind_speed",
        "fire_count": "Fire_Count",
        "fire_pressure": "Fire_Pressure",
        "fire_pressure_3day_avg": "Fire_Pressure_3Day_Avg",
        "is_burning_season": "Is_Burning_Season",
        "month": "Month",
    }

    normalized = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"])
    return normalized


@st.cache_data(ttl=900)
def fetch_weather_and_forecast():
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_KEY":
        return None, pd.DataFrame()

    try:
        current_url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?lat={MFU_LAT}&lon={MFU_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        forecast_url = (
            "https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={MFU_LAT}&lon={MFU_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        )

        current_weather = requests.get(current_url, timeout=8).json()
        forecast_weather = requests.get(forecast_url, timeout=8).json()

        try:
            air4thai_url = (
                "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
                f"?stationID={AIR4THAI_STATION_ID}"
            )
            air4thai = requests.get(air4thai_url, timeout=6).json()
            latest_pm25 = float(air4thai["AQILast"]["PM25"]["value"])
            pm25_source = f"Air4Thai station {AIR4THAI_STATION_ID}"
        except Exception:
            pollution_url = (
                "https://api.openweathermap.org/data/2.5/air_pollution"
                f"?lat={MFU_LAT}&lon={MFU_LON}&appid={OPENWEATHER_API_KEY}"
            )
            pollution = requests.get(pollution_url, timeout=8).json()
            latest_pm25 = float(pollution["list"][0]["components"]["pm2_5"])
            pm25_source = "OpenWeather air pollution API"

        current = {
            "temp": float(current_weather["main"]["temp"]),
            "humidity": float(current_weather["main"]["humidity"]),
            "pressure": float(current_weather["main"]["pressure"]),
            "wind_speed": float(current_weather["wind"]["speed"]),
            "wind_direction": float(current_weather["wind"].get("deg", 0)),
            "desc": current_weather["weather"][0]["description"].title(),
            "pm25_current": latest_pm25,
            "pm25_source": pm25_source,
            "fetch_time": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        }

        forecast_rows = []
        for item in forecast_weather.get("list", []):
            forecast_rows.append(
                {
                    "datetime": datetime.fromtimestamp(item["dt"]),
                    "Pressure_avg": float(item["main"]["pressure"]),
                    "Temp_avg": float(item["main"]["temp"]),
                    "Humidity_avg": float(item["main"]["humidity"]),
                    "Precipitation": float(item.get("rain", {}).get("3h", 0)),
                    "Sunshine": 5.0,
                    "Wind_direct": float(item["wind"].get("deg", 0)),
                    "Wind_speed": float(item["wind"]["speed"]),
                    "pm25_lag1": latest_pm25,
                }
            )

        return current, pd.DataFrame(forecast_rows)

    except Exception as exc:
        st.sidebar.warning(f"Weather API failed: {exc}")
        return None, pd.DataFrame()


def fetch_nasa_fire_for_date(date_str: str):
    if NASA_KEY == "YOUR_NASA_KEY":
        return pd.DataFrame()

    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{NASA_KEY}/VIIRS_SNPP_NRT/{NASA_BBOX}/1/{date_str}"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()

        df = pd.read_csv(StringIO(response.text))
        if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
            return pd.DataFrame()

        df["distance_km"] = df.apply(
            lambda row: haversine(MFU_LAT, MFU_LON, row["latitude"], row["longitude"]),
            axis=1,
        )
        df["fire_pressure"] = df["bright_ti4"] / ((df["distance_km"] + 1) ** 2)
        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def fetch_recent_fire_features():
    today = datetime.now()
    frames = []
    pressure_values = []
    count_values = []

    for offset in range(3):
        date_str = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        daily = fetch_nasa_fire_for_date(date_str)
        if not daily.empty:
            daily["source_date"] = date_str
            frames.append(daily)
            count_values.append(len(daily))
            pressure_values.append(float(daily["fire_pressure"].sum()))
        else:
            count_values.append(0)
            pressure_values.append(0.0)

    hotspots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fire_count = int(count_values[0])
    fire_pressure = round(pressure_values[0], 4)
    fire_pressure_lag1 = round(pressure_values[1], 4)
    fire_pressure_lag2 = round(pressure_values[2], 4)
    fire_pressure_3day_avg = round(float(np.mean(pressure_values)), 4)

    return {
        "hotspots": hotspots,
        "fire_count": fire_count,
        "fire_pressure": fire_pressure,
        "fire_pressure_lag1": fire_pressure_lag1,
        "fire_pressure_lag2": fire_pressure_lag2,
        "fire_pressure_3day_avg": fire_pressure_3day_avg,
    }


@st.cache_data(ttl=600)
def load_historical_data() -> pd.DataFrame:
    candidates = [
        os.path.join(SCRIPT_DIR, "data", "final", "pm25_training_dataset_2018_2022.csv"),
        os.path.join(SCRIPT_DIR, "data", "processed", "chiang_rai_pm25_weather_2018_2022_clean.csv"),
        os.path.join(SCRIPT_DIR, "data", "processed", "final_training_data.csv"),
        os.path.join(SCRIPT_DIR, "final_training_data.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return normalize_history_columns(df)
            except Exception:
                continue

    return pd.DataFrame()


def build_fire_input(base_df: pd.DataFrame, fire_features: dict) -> pd.DataFrame:
    model_input = base_df.copy()

    model_input["pm25_lag2"] = model_input["pm25_lag1"]
    model_input["pm25_lag3"] = model_input["pm25_lag1"]
    model_input["pm25_3Day_Avg"] = model_input["pm25_lag1"]

    model_input["Fire_Count"] = fire_features["fire_count"]
    model_input["Fire_Pressure"] = fire_features["fire_pressure"]
    model_input["Fire_Pressure_Lag1"] = fire_features["fire_pressure_lag1"]
    model_input["Fire_Pressure_Lag2"] = fire_features["fire_pressure_lag2"]
    model_input["Fire_Pressure_3Day_Avg"] = fire_features["fire_pressure_3day_avg"]

    model_input["Month"] = datetime.now().month
    model_input["Is_Burning_Season"] = 1 if datetime.now().month in [2, 3, 4, 5] else 0

    return model_input[FEATURE_COLS_18]


def predict_log_model(model, input_df: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(input_df))

    raw_predictions = model.predict(input_df)
    return np.expm1(raw_predictions).clip(min=0)


def predict_raw_model(model, input_df: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(input_df))

    return np.asarray(model.predict(input_df)).clip(min=0)


def make_gistda_map_html(hotspots: pd.DataFrame, current_pred: float, status_text: str, status_icon: str, wind_direction: float = 0):
    if GISTDA_KEY == "YOUR_GISTDA_KEY":
        return None

    hotspot_rows = []
    if not hotspots.empty:
        for _, row in hotspots.head(250).iterrows():
            distance = float(row.get("distance_km", 0))
            if distance <= 100:
                hotspot_rows.append(
                    {
                        "lat": float(row["latitude"]),
                        "lon": float(row["longitude"]),
                        "distance": round(distance, 1),
                    }
                )

    hotspots_json = json.dumps(hotspot_rows)

    final_detail = (
        f"<div style='font-family:Inter,Arial,sans-serif; color:#0f172a; min-width:230px;'>"
        f"<div style='font-size:15px; font-weight:800; margin-bottom:6px;'>MFU Prediction Target</div>"
        f"<div><b>Area:</b> Mae Fah Luang University</div>"
        f"<div><b>Predicted PM2.5:</b> {current_pred:.1f} &micro;g/m&sup3;</div>"
        f"<div><b>Status:</b> {status_icon} {status_text}</div>"
        f"<div><b>Model:</b> LightGBM Fire-Integrated</div>"
        f"<div style='margin-top:6px; color:#475569;'>The blue box marks the localized area represented by this prediction.</div>"
        f"</div>"
    )

    wind_detail = (
        f"<div style='font-family:Inter,Arial,sans-serif; color:#0f172a; min-width:220px;'>"
        f"<div style='font-size:15px; font-weight:800; margin-bottom:6px;'>Wind Direction Indicator</div>"
        f"<div><b>Direction:</b> {wind_direction:.0f}&deg;</div>"
        f"<div style='margin-top:6px; color:#475569;'>Arrow direction visualizes the current wind vector used as one of the model inputs.</div>"
        f"</div>"
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            html, body {{ margin: 0; width: 100%; height: 100%; font-family: Inter, Arial, sans-serif; }}
            #map {{
                width: 100%;
                height: 590px;
                border-radius: 16px;
                background: #f4f7f9;
                border: 1px solid #dbe3ea;
                overflow: hidden;
            }}
        </style>
        <script src="https://api.sphere.gistda.or.th/map/?key={GISTDA_KEY}"></script>
    </head>
    <body>
        <div id="map"></div>
        <script>
            function loadSphereMap() {{
                if (!window.sphere) return;

                const map = new window.sphere.Map({{
                    placeholder: document.getElementById("map"),
                    zoom: 10,
                    center: {{ lon: {MFU_LON}, lat: {MFU_LAT} }}
                }});

                map.Event.bind(window.sphere.EventName.Ready, function () {{
                    const mfu = {{ lon: {MFU_LON}, lat: {MFU_LAT} }};

                    map.Overlays.add(new window.sphere.Polygon([
                        {{ lon: 99.855, lat: 20.015 }},
                        {{ lon: 99.935, lat: 20.015 }},
                        {{ lon: 99.935, lat: 20.075 }},
                        {{ lon: 99.855, lat: 20.075 }}
                    ], {{
                        title: "MFU localized prediction area",
                        detail: "This boundary represents the campus-focused area for safe interpretation of the model output.",
                        lineColor: '#1e3a8a',
                        lineWidth: 3,
                        fillColor: 'rgba(30,58,138,0.10)'
                    }}));

                    map.Overlays.add(new window.sphere.Circle(mfu, 25000, {{
                        title: "25 km near-campus monitoring radius",
                        detail: "Near-field smoke and weather monitoring radius.",
                        lineColor: '#334155',
                        lineWidth: 2,
                        fillColor: 'rgba(51,65,85,0.05)'
                    }}));

                    map.Overlays.add(new window.sphere.Circle(mfu, 100000, {{
                        title: "100 km fire monitoring radius",
                        detail: "NASA FIRMS hotspots within this radius are visualized as fire markers.",
                        lineColor: '#dc2626',
                        lineWidth: 2,
                        lineDash: [6, 6],
                        fillColor: 'rgba(220,38,38,0.035)'
                    }}));

                    map.Overlays.add(new window.sphere.Marker(mfu, {{
                        title: "Mae Fah Luang University",
                        detail: `{final_detail}`,
                        icon: {{
                            html: '<div style="font-size: 30px; filter: drop-shadow(0 2px 3px rgba(15,23,42,0.35));">🎓</div>',
                            offset: {{ x: 15, y: 15 }}
                        }}
                    }}));

                    map.Overlays.add(new window.sphere.Marker({{ lon: {MFU_LON + 0.025}, lat: {MFU_LAT + 0.018} }}, {{
                        title: "Current wind direction",
                        detail: `{wind_detail}`,
                        icon: {{
                            html: '<div style="font-size: 31px; transform: rotate({wind_direction}deg); filter: drop-shadow(0 2px 3px rgba(15,23,42,0.35));">⬇️</div>',
                            offset: {{ x: 15, y: 15 }}
                        }}
                    }}));

                    const hotspots = {hotspots_json};
                    hotspots.forEach(pt => {{
                        map.Overlays.add(new window.sphere.Marker({{ lon: pt.lon, lat: pt.lat }}, {{
                            title: "NASA FIRMS fire hotspot",
                            detail: "Distance from MFU: " + pt.distance + " km<br>Source: NASA FIRMS VIIRS",
                            icon: {{
                                html: '<div style="font-size: 18px; filter: drop-shadow(0 1px 2px rgba(15,23,42,0.35));">🔥</div>',
                                offset: {{ x: 9, y: 9 }}
                            }}
                        }}));
                    }});
                }});
            }}

            window.onload = () => setTimeout(loadSphereMap, 900);
        </script>
    </body>
    </html>
    """
    return html


def generate_llm_warning(language: str, current_pred: float, max_pred: float, status_text: str, current_data: dict, fire_features: dict):
    if not GEMINI_API_KEY:
        return "SETUP_REQUIRED::GEMINI_API_KEY is missing. Add GEMINI_API_KEY to .streamlit/secrets.toml before generating the advisory."

    if genai is None:
        return "SETUP_REQUIRED::The google-generativeai package is not installed in this virtual environment. Install it with: python -m pip install google-generativeai"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        prompt = f"""
You are an environmental health advisory assistant for Mae Fah Luang University (MFU) in Chiang Rai, Thailand.

Generate the response strictly in {language}.
Audience: university students, lecturers, staff, and visitors.
Tone: calm, practical, readable, and not like a government order.

Use only this data:
- Current predicted PM2.5: {current_pred:.1f} µg/m³
- Maximum predicted PM2.5 in the 5-day forecast: {max_pred:.1f} µg/m³
- Risk status: {status_text}
- Temperature: {current_data.get("temp", 0):.1f} °C
- Humidity: {current_data.get("humidity", 0):.1f}%
- Wind speed: {current_data.get("wind_speed", 0):.1f} m/s
- Wind direction: {current_data.get("wind_direction", 0):.0f} degrees
- Active NASA FIRMS hotspots today: {fire_features["fire_count"]}
- Fire pressure index: {fire_features["fire_pressure"]:.2f}

Output format:
1. Situational Analysis
   - Explain the current campus air-quality situation in simple terms.
   - Mention wind and fire hotspot context only when useful.
   - If the situation is normal, say it clearly and avoid over-warning.

2. Recommended Actions
   - Give practical actions for students and staff.
   - If air quality is good, keep the advice light and reassuring.
   - If air quality is moderate or worse, recommend reasonable outdoor activity and mask guidance.

Rules:
- Do not invent numbers or locations.
- Do not claim official government authority.
- Do not exaggerate.
- Keep it concise and easy to read.
"""
        # --- 🛡️ FALLBACK MECHANISM ---
        # List models in order of priority (fastest/cheapest first, followed by more capable/older stable ones)
        fallback_models = [
            "gemini-3.1-flash-lite", 
            "gemini-3.5-flash",
            "gemini-3-flash",
            "gemini-2.5-flash"
        ]

        last_error = None
        for model_name in fallback_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response.text:
                    return response.text
            except Exception as e:
                # If current model fails, record the error and continue to the next model in the list
                last_error = e
                continue

        # If ALL models in the list fail, only then return the error message
        return f"AI_ERROR::AI generation failed across all fallback models. Last error: {last_error}"

    except Exception as exc:
        return f"AI_ERROR::System error during AI generation setup: {exc}"



# =============================================================================
# LOAD DATA / MODELS
# =============================================================================

models = load_models()
current_data, forecast_df = fetch_weather_and_forecast()
fire_features = fetch_recent_fire_features()
history_df = load_historical_data()

if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""


# =============================================================================
# HEADER / SIDEBAR
# =============================================================================

# Using the official MFU logo URL provided by PT
mfu_logo_url = "https://archives.mfu.ac.th/wp-content/uploads/2019/06/Mae-Fah-Luang-University-2.png"
logo_html = f'<img src="{mfu_logo_url}" class="logo-img" style="max-height: 65px; width: auto; margin-right: 10px;">'

st.markdown(
    f"""
<div class="header-wrap">
    {logo_html}
    <div>
        <h1 class="header-title">MFU PM2.5 GeoAI Warning Dashboard</h1>
        <p class="header-subtitle">Chiang Rai localized prediction · NASA FIRMS fire monitoring · GISTDA spatial visualization · AI campus advisory</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## MFU PM2.5 GeoAI")
    st.markdown("**System readiness**")

    st.write(f"LightGBM Champion: {'Ready' if models['lgbm_fire'] else 'Offline'}")
    st.write(f"XGBoost Contender: {'Ready' if models['xgb_fire'] else 'Offline'}")
    st.write(f"SVR Baseline: {'Ready' if models['svr'] else 'Offline'}")
    st.write(f"MLR Baseline: {'Ready' if models['mlr'] else 'Offline'}")
    st.divider()

    st.markdown("**Live data layers**")
    st.write("Air4Thai / OpenWeather")
    st.write("NASA FIRMS VIIRS")
    st.write("GISTDA Sphere Map")
    st.write("Gemini AI Advisory")
    st.divider()

    st.markdown("**Presentation theme**")
    st.write("Navy · White · Slate")


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

pipeline_ready = current_data is not None and not forecast_df.empty and models["lgbm_fire"] is not None

if pipeline_ready:
    base_input = forecast_df[
        [
            "Pressure_avg",
            "Temp_avg",
            "Humidity_avg",
            "Precipitation",
            "Sunshine",
            "Wind_direct",
            "Wind_speed",
            "pm25_lag1",
        ]
    ].copy()

    fire_input = build_fire_input(base_input, fire_features)

    forecast_df["lgbm_pm25"] = predict_log_model(models["lgbm_fire"], fire_input)
    forecast_df["xgb_pm25"] = predict_log_model(models["xgb_fire"], fire_input)

    current_7 = base_input[FEATURE_COLS_7]
    forecast_df["svr_pm25"] = predict_raw_model(models["svr"], current_7)
    forecast_df["mlr_pm25"] = predict_raw_model(models["mlr"], current_7)

    forecast_df["predicted_pm25"] = forecast_df["lgbm_pm25"].rolling(2, min_periods=1).mean()

    current_pred = float(forecast_df.iloc[0]["predicted_pm25"])
    max_pred = float(forecast_df["predicted_pm25"].max())
    avg_pred = float(forecast_df["predicted_pm25"].mean())
    status_text, status_color, status_icon = get_risk_label(current_pred)
else:
    current_pred = 0.0
    max_pred = 0.0
    avg_pred = 0.0
    status_text, status_color, status_icon = "Offline", TEXT_MUTED, "⚪"


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🗺️ Prediction & Map",
        "🤖 AI Advisory(Gemini)",
        "📊 Charts & Graphs",
        "🔬 Model Overview",
    ]
)


# =============================================================================
# TAB 1: PREDICTION & MAP
# =============================================================================

with tab1:
    if not pipeline_ready:
        st.error(
            "Prediction pipeline is not ready. Check API keys and required model file: models/lgbm_pm25_model.pkl"
        )
    else:
        # --- 1. TOP ROW: Prediction Card & Fire Alert Card ---
        top_left, top_right = st.columns([1.65, 1], gap="large")

        with top_left:
            # Main Prediction Card (Fixed Height for Alignment)
            render_html(f"""
<div class="professional-card" style="border-left: 6px solid {ROYAL_BLUE}; margin-bottom: 24px; height: 165px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
    <div style="font-size: 0.8rem; font-weight: 800; color: {TEXT_MUTED}; letter-spacing: 0.05em; text-transform: uppercase;">
        📍 Mae Fah Luang University
    </div>
    <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 12px;">
        <div style="font-size: 1.2rem; font-weight: 800; color: {TEXT_DARK};">Current PM2.5:</div>
        <div style="font-size: 3rem; font-weight: 900; color: {status_color}; letter-spacing: -0.04em; line-height: 1;">
            {current_pred:.1f} <span style="font-size: 1.2rem; font-weight: 700;">µg/m³</span>
        </div>
    </div>
    <div style="margin-top: 12px; font-size: 0.95rem; color: {TEXT_DARK};">
        Model: <strong>LightGBM (85.90% Fire-Integrated)</strong> &nbsp;|&nbsp; Status: <strong style="color: {status_color};">{status_text}</strong>
    </div>
</div>
            """)

        with top_right:
            # Regional Fire Alert Card (Swapped to Top, Fixed Height)
            fire_count = fire_features.get("fire_count", 0)
            render_html(f"""
<div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.05); height: 165px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
    <div style="color: #dc2626; font-weight: 900; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; gap: 8px; text-transform: uppercase; letter-spacing: 0.05em;">
        🔥 Regional Fire Alert
    </div>
    <div style="margin-top: 10px; font-size: 1.05rem; color: #7f1d1d; font-weight: 700;">
        Active Hotspots: <span style="font-size: 2.2rem; font-weight: 900; color: #b91c1c; margin-left: 6px; line-height: 1;">{fire_count}</span>
    </div>
    <div style="font-size: 0.8rem; color: #991b1b; margin-top: 10px; font-weight: 700;">
        Integrated into LightGBM Predictions
    </div>
</div>
            """)


        # --- 2. MIDDLE ROW: Full-Stretch Map ---
        st.markdown('<div class="section-title" style="margin-top: 6px; font-size: 1.15rem;">📍 GISTDA map: Prediction area & Nearby fire activity</div>', unsafe_allow_html=True)
        map_html = make_gistda_map_html(
            fire_features["hotspots"],
            current_pred,
            status_text,
            status_icon,
            current_data.get("wind_direction", 0),
        )
        if map_html:
            components.html(map_html, height=520, scrolling=False)
        else:
            st.warning("GISTDA_KEY is missing. Map cannot be displayed.")


        # --- 3. BOTTOM ROW: Charts vs Weather & Forecast Summaries ---
        st.markdown("<br>", unsafe_allow_html=True)
        bot_left, bot_right = st.columns([1.65, 1], gap="large")

        with bot_left:
            # Forecast Line Chart
            st.markdown('<div class="section-title" style="margin-top: 0px; font-size: 1.15rem;">📈 5-Day PM2.5 Forecast Trend</div>', unsafe_allow_html=True)
            fig_forecast = px.line(
                forecast_df,
                x="datetime",
                y="predicted_pm25",
                labels={"datetime": "Date/time", "predicted_pm25": "PM2.5 (µg/m³)"},
            )
            fig_forecast.update_traces(line=dict(color=ROYAL_BLUE, width=3), mode="lines+markers", marker=dict(size=5))
            fig_forecast.add_hline(y=50, line_dash="dash", line_color=UNHEALTHY, annotation_text="Unhealthy threshold (50)")
            fig_forecast = apply_plot_style(fig_forecast, height=320)
            fig_forecast.update_layout(margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_forecast, use_container_width=True, theme=None)

            # PM2.5 Level Guide
            render_html("""
<div style="margin-top: 24px;">
    <div style="font-weight: 800; color: #0f172a; margin-bottom: 12px; font-size: 1rem;">📊 PM2.5 Level Guide (µg/m³)</div>
    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 120px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #166534;">0 - 25</div>
            <div style="font-size: 1rem; font-weight: 900; color: #15803d;">Good</div>
        </div>
        <div style="flex: 1; min-width: 120px; background: #fefce8; border: 1px solid #fef08a; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #854d0e;">26 - 50</div>
            <div style="font-size: 1rem; font-weight: 900; color: #b7791f;">Moderate</div>
        </div>
        <div style="flex: 1; min-width: 120px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #991b1b;">51 - 100</div>
            <div style="font-size: 1rem; font-weight: 900; color: #dc2626;">Unhealthy</div>
        </div>
        <div style="flex: 1; min-width: 120px; background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #6b21a8;">100+</div>
            <div style="font-size: 1rem; font-weight: 900; color: #7e22ce;">Hazardous</div>
        </div>
    </div>
</div>
            """)

        with bot_right:
            # 1. Weather Card (Swapped to Bottom)
            weather_desc = current_data.get("desc", "Unknown")
            weather_emoji = "⛅"
            if "sun" in weather_desc.lower() or "clear" in weather_desc.lower(): weather_emoji = "☀️"
            elif "rain" in weather_desc.lower() or "drizzle" in weather_desc.lower(): weather_emoji = "🌧️"
            elif "cloud" in weather_desc.lower(): weather_emoji = "☁️"
            elif "storm" in weather_desc.lower() or "thunder" in weather_desc.lower(): weather_emoji = "⛈️"

            render_html(f"""
<div class="professional-card" style="text-align: center; margin-bottom: 24px;">
    <div style="font-size: 0.75rem; font-weight: 800; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px;">
        🕒 LAST UPDATED: {current_data.get("fetch_time", "-").split(',')[0]}
    </div>
    <div style="font-size: 3rem; font-weight: 900; color: {NAVY_BLUE}; line-height: 1;">
        {current_data.get("temp", 0):.1f}°C
    </div>
    <div style="font-size: 1.1rem; font-weight: 700; color: {TEXT_DARK}; margin-top: 12px;">
        {weather_desc} {weather_emoji}
    </div>
    <div style="display: flex; justify-content: center; gap: 24px; margin-top: 20px; padding-top: 16px; border-top: 1px solid {BORDER};">
        <div style="font-size: 0.9rem; font-weight: 700; color: {TEXT_MUTED};">💨 {current_data.get("wind_speed", 0):.1f} m/s</div>
        <div style="font-size: 0.9rem; font-weight: 700; color: {TEXT_MUTED};">💧 {current_data.get("humidity", 0):.0f}%</div>
    </div>
</div>
            """)

            # 2. Forecast Summary (Daily Max)
            forecast_df['Date_Str'] = forecast_df['datetime'].dt.strftime('%d %A')
            daily_summary = forecast_df.groupby('Date_Str', sort=False)['predicted_pm25'].max().reset_index().head(5)
            
            summary_html = ""
            for _, row in daily_summary.iterrows():
                summary_html += f"""
<div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid {BORDER};">
    <span style="color: {TEXT_MUTED}; font-weight: 700; font-size: 0.95rem;">{row['Date_Str']}</span>
    <span style="color: {NAVY_BLUE}; font-weight: 900; font-size: 0.95rem;">PM2.5: {row['predicted_pm25']:.1f} µg/m³</span>
</div>
"""
                
            render_html(f"""
<div style="background: {SURFACE_WHITE}; padding: 20px; border-radius: 16px; margin-bottom: 18px;">
    <div style="font-size: 1.25rem; font-weight: 900; color: {NAVY_BLUE}; margin-bottom: 12px;">Forecast Summary</div>
{summary_html}
</div>
            """)

            # 3. MFU Medical Center Button
            render_html("""
<a href="https://hospital.mfu.ac.th/" target="_blank" style="display: block; width: 100%; text-align: center; background: #ffffff; border: 1px solid #dbe3ea; padding: 14px; border-radius: 12px; color: #9f1239; font-weight: 800; font-size: 1rem; text-decoration: none; box-shadow: 0 4px 6px rgba(15, 23, 42, 0.04); transition: all 0.2s ease;">
    🏥 MFU Medical Center
</a>
            """)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# =============================================================================
# TAB 2: LLM WARNING
# =============================================================================

with tab2:
    render_html(f"""
    <div class="professional-card blue-card">
        <h2 style="margin:0; color:{NAVY_BLUE}; letter-spacing:-0.035em;">AI campus advisory</h2>
        <p style="margin:10px 0 0 0; color:{TEXT_MUTED}; font-weight:750; line-height:1.55;">
            Generates a calm situational analysis for MFU students and staff using the live prediction, weather, wind, and fire monitoring data.
        </p>
    </div>
    """)

    if not pipeline_ready:
        st.error("Prediction data is required before generating an advisory.")
    else:
        left, right = st.columns([0.82, 1.75], gap="large")

        with left:
            render_html(f"""
            <div class="professional-card" style="margin-top:18px;">
                <div class="clean-label">Current PM2.5</div>
                <div class="clean-value" style="font-size:2.4rem; color:{status_color};">{current_pred:.1f} µg/m³</div>
                <div class="clean-subtext">{status_icon} {status_text} · max forecast {max_pred:.1f} µg/m³</div>
                <hr style="border:0; border-top:1px solid {BORDER}; margin:18px 0;">
                <div class="clean-label">Monitoring context</div>
                <div style="color:{TEXT_DARK}; font-weight:850; line-height:1.75;">
                    Weather: {current_data.get("desc", "Unknown")}<br>
                    Wind: {current_data.get("wind_speed", 0):.1f} m/s at {current_data.get("wind_direction", 0):.0f}°<br>
                    NASA hotspots: {fire_features["fire_count"]}<br>
                    Model: gemini-3.1-flash-lite
                </div>
            </div>
            """)

            # --- START OF UI FIX ---
            st.markdown("""
            <div style="background: #e2e8f0; padding: 6px 12px; border-radius: 6px; display: inline-block; font-weight: 750; color: #0f172a; font-size: 0.85rem; margin-bottom: 4px; margin-top: 18px; border: 1px solid #cbd5e1;">
                🗣️ Output language
            </div>
            """, unsafe_allow_html=True)

            language = st.selectbox(
                "Hidden Label",
                ["English", "Thai", "Burmese", "Chinese"],
                index=1,
                label_visibility="collapsed"
            )
            # --- END OF UI FIX ---

            if genai is None:
                render_html("""
                <div class="setup-card">
                    Gemini advisory is not ready in this virtual environment because <code>google-generativeai</code> is not installed.<br>
                    Install command:<br><code>python -m pip install google-generativeai</code>
                </div>
                """)

            if st.button("Generate campus advisory"):
                with st.spinner(f"Generating advisory in {language}..."):
                    st.session_state.ai_report = generate_llm_warning(
                        language=language,
                        current_pred=current_pred,
                        max_pred=max_pred,
                        status_text=status_text,
                        current_data=current_data,
                        fire_features=fire_features,
                    )

        with right:
            st.markdown('<div class="section-title" style="margin-top:18px;">Advisory output</div>', unsafe_allow_html=True)
            if st.session_state.ai_report:
                if st.session_state.ai_report.startswith("SETUP_REQUIRED::"):
                    message = st.session_state.ai_report.replace("SETUP_REQUIRED::", "")
                    render_html(f"""
                    <div class="setup-card">
                        <strong>Setup required</strong><br>{message}
                    </div>
                    """)
                elif st.session_state.ai_report.startswith("AI_ERROR::"):
                    message = st.session_state.ai_report.replace("AI_ERROR::", "")
                    render_html(f"""
                    <div class="setup-card">
                        <strong>AI advisory could not be generated.</strong><br>{message}
                    </div>
                    """)
                else:
                    safe_report = st.session_state.ai_report.replace("\n", "<br>")
                    render_html(f"""
                    <div class="advisory-shell">
                        {safe_report}
                    </div>
                    """)
            else:
                render_html("""
                <div class="advisory-shell" style="color:#64748b; font-weight:750;">
                    Choose a language and click <strong>Generate campus advisory</strong>. The output will appear here as a readable situational analysis followed by practical recommendations.
                </div>
                """)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# =============================================================================
# TAB 3: CHARTS & GRAPHS
# =============================================================================

with tab3:
    st.markdown(
        f"""
<div class="professional-card blue-card">
    <h2 style="margin-top:0;color:{NAVY_BLUE};">Historical trends and fire-pressure analysis</h2>
    <p style="color:{TEXT_MUTED};font-weight:700;line-height:1.55;">
        Historical PM2.5 patterns, fire activity, weather relationships, and seasonal behavior for the Chiang Rai-focused dataset.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if history_df.empty:
        st.warning("Historical dataset not found. Expected data/final/pm25_training_dataset_2018_2022.csv")
    else:
        history_df = history_df.sort_values("Date") if "Date" in history_df.columns else history_df

        c1, c2 = st.columns(2, gap="large")

        with c1:
            if {"Date", "PM25"}.issubset(history_df.columns):
                fig_hist = px.line(
                    history_df,
                    x="Date",
                    y="PM25",
                    title="Historical PM2.5 trend",
                    labels={"PM25": "PM2.5 (µg/m³)", "Date": "Date"},
                )
                fig_hist.update_traces(line=dict(color=ROYAL_BLUE, width=3))
                fig_hist = apply_plot_style(fig_hist, height=390)
                st.plotly_chart(fig_hist, use_container_width=True, theme=None)

        with c2:
            if {"Temp_avg", "PM25", "Humidity_avg"}.issubset(history_df.columns):
                fig_scatter = px.scatter(
                    history_df,
                    x="Temp_avg",
                    y="PM25",
                    color="Humidity_avg",
                    title="Temperature vs PM2.5 correlation",
                    labels={
                        "Temp_avg": "Temperature (°C)",
                        "PM25": "PM2.5 (µg/m³)",
                        "Humidity_avg": "Humidity (%)",
                    },
                    color_continuous_scale="Blues",
                )
                fig_scatter = apply_plot_style(fig_scatter, height=390)
                st.plotly_chart(fig_scatter, use_container_width=True, theme=None)

        c3, c4 = st.columns(2, gap="large")

        with c3:
            if {"Date", "Fire_Count"}.issubset(history_df.columns):
                fig_fire_count = px.bar(
                    history_df,
                    x="Date",
                    y="Fire_Count",
                    title="NASA FIRMS fire hotspot count",
                    labels={"Fire_Count": "Fire count", "Date": "Date"},
                )
                fig_fire_count.update_traces(marker_color=NAVY_BLUE)
                fig_fire_count = apply_plot_style(fig_fire_count, height=380)
                st.plotly_chart(fig_fire_count, use_container_width=True, theme=None)

        with c4:
            if {"Date", "Fire_Pressure"}.issubset(history_df.columns):
                fig_fire_pressure = px.line(
                    history_df,
                    x="Date",
                    y="Fire_Pressure",
                    title="Fire pressure trend",
                    labels={"Fire_Pressure": "Fire pressure index", "Date": "Date"},
                )
                fig_fire_pressure.update_traces(line=dict(color=UNHEALTHY, width=2.5))
                fig_fire_pressure = apply_plot_style(fig_fire_pressure, height=380)
                st.plotly_chart(fig_fire_pressure, use_container_width=True, theme=None)

        if {"Date", "PM25", "Fire_Pressure"}.issubset(history_df.columns):
            st.markdown('<div class="section-title">PM2.5 and fire pressure combined view</div>', unsafe_allow_html=True)
            fig_combined = go.Figure()
            fig_combined.add_trace(
                go.Scatter(
                    x=history_df["Date"],
                    y=history_df["PM25"],
                    name="PM2.5",
                    line=dict(color=ROYAL_BLUE, width=3),
                )
            )
            fig_combined.add_trace(
                go.Scatter(
                    x=history_df["Date"],
                    y=history_df["Fire_Pressure"],
                    name="Fire pressure",
                    yaxis="y2",
                    line=dict(color=UNHEALTHY, width=2),
                )
            )
            fig_combined.update_layout(
                title="PM2.5 vs fire pressure",
                yaxis=dict(title="PM2.5 (µg/m³)"),
                yaxis2=dict(title="Fire pressure", overlaying="y", side="right"),
            )
            fig_combined = apply_plot_style(fig_combined, height=430)
            st.plotly_chart(fig_combined, use_container_width=True, theme=None)

        if {"Month", "PM25"}.issubset(history_df.columns):
            monthly = (
                history_df.groupby("Month", as_index=False)["PM25"]
                .mean()
                .sort_values("Month")
            )
            fig_month = px.bar(
                monthly,
                x="Month",
                y="PM25",
                title="Average PM2.5 by month",
                labels={"Month": "Month", "PM25": "Average PM2.5 (µg/m³)"},
            )
            fig_month.update_traces(marker_color=ROYAL_BLUE)
            fig_month = apply_plot_style(fig_month, height=380)
            st.plotly_chart(fig_month, use_container_width=True, theme=None)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# =============================================================================
# TAB 4: MODEL OVERVIEW
# =============================================================================

with tab4:
    st.markdown('<div class="section-title" style="font-size: 1.5rem; margin-bottom: 6px;">🥊 4-Model Defense Showdown</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted" style="margin-top:-10px; margin-bottom:24px;">Scientific justification for selecting LightGBM with Fire Factors over standard ML approaches.</p>', unsafe_allow_html=True)

    if pipeline_ready:
        current_fire_input = fire_input.iloc[[0]]
        current_weather_7 = base_input[FEATURE_COLS_7].iloc[[0]]

        live_lgbm = predict_log_model(models["lgbm_fire"], current_fire_input)[0]
        live_xgb = predict_log_model(models["xgb_fire"], current_fire_input)[0]
        live_svr = predict_raw_model(models["svr"], current_weather_7)[0]
        live_mlr = predict_raw_model(models["mlr"], current_weather_7)[0]

        # 1. Top Section: 2x2 Grid for 4 Models Showdown Cards
        r1_c1, r1_c2 = st.columns(2, gap="large")
        r2_c1, r2_c2 = st.columns(2, gap="large")

        with r1_c1:
            render_html(f"""
<div class="professional-card" style="background: #f0fdf4; border: 1px solid #bbf7d0; min-height: 155px; margin-bottom: 20px;">
    <div style="font-size: 0.75rem; font-weight: 900; color: #166534; background: #dcfce7; display: inline-block; padding: 4px 10px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">
        🥇 THE ULTIMATE CHAMPION
    </div>
    <div style="font-size: 1.35rem; font-weight: 900; color: #14532d; margin-top: 12px;">LightGBM (Fire Factors)</div>
    <div style="font-size: 2.6rem; font-weight: 900; color: #166534; margin-top: 10px; letter-spacing: -0.04em;">
        {live_lgbm:.1f} <span style="font-size: 1.1rem; font-weight: 700;">μg/m³</span>
    </div>
</div>
            """)

        with r1_c2:
            render_html(f"""
<div class="professional-card" style="border-top: 5px solid {ROYAL_BLUE}; min-height: 155px; margin-bottom: 20px;">
    <div style="font-size: 0.75rem; font-weight: 900; color: {ROYAL_BLUE}; background: #eff6ff; display: inline-block; padding: 4px 10px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">
        🔥 STRONG CONTENDER
    </div>
    <div style="font-size: 1.35rem; font-weight: 900; color: {NAVY_BLUE}; margin-top: 12px;">XGBoost (Fire Factors)</div>
    <div style="font-size: 2.6rem; font-weight: 900; color: {ROYAL_BLUE}; margin-top: 10px; letter-spacing: -0.04em;">
        {live_xgb:.1f} <span style="font-size: 1.1rem; font-weight: 700;">μg/m³</span>
    </div>
</div>
            """)

        with r2_c1:
            render_html(f"""
<div class="professional-card" style="border-top: 5px solid {SLATE}; min-height: 155px; margin-bottom: 20px;">
    <div style="font-size: 0.75rem; font-weight: 900; color: {SLATE}; background: #f8fafc; display: inline-block; padding: 4px 10px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">
        ⛅ WEATHER ONLY BASELINE
    </div>
    <div style="font-size: 1.35rem; font-weight: 900; color: {NAVY_BLUE}; margin-top: 12px;">Support Vector Reg.</div>
    <div style="font-size: 2.6rem; font-weight: 900; color: {SLATE}; margin-top: 10px; letter-spacing: -0.04em;">
        {live_svr:.1f} <span style="font-size: 1.1rem; font-weight: 700;">μg/m³</span>
    </div>
</div>
            """)

        with r2_c2:
            render_html(f"""
<div class="professional-card" style="border-top: 5px solid {BORDER_STRONG}; min-height: 155px; margin-bottom: 20px;">
    <div style="font-size: 0.75rem; font-weight: 900; color: #475569; background: #f1f5f9; display: inline-block; padding: 4px 10px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">
        📉 LINEAR BASELINE
    </div>
    <div style="font-size: 1.35rem; font-weight: 900; color: {NAVY_BLUE}; margin-top: 12px;">Multiple Linear Reg.</div>
    <div style="font-size: 2.6rem; font-weight: 900; color: #475569; margin-top: 10px; letter-spacing: -0.04em;">
        {live_mlr:.1f} <span style="font-size: 1.1rem; font-weight: 700;">μg/m³</span>
    </div>
</div>
            """)

        # 2. Bottom Section: 2x2 Macro Layout (Left: Accuracy Chart + Text, Right: Error Chart + Text)
        st.markdown("<br>", unsafe_allow_html=True)
        bot_col1, bot_col2 = st.columns(2, gap="large")

        with bot_col1:
            # Chart: Accuracy Comparison (R²)
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=["LightGBM", "XGBoost", "SVR", "Linear Reg"],
                y=[85.90, 85.03, 22.73, -32.55],
                text=["85.9%", "85.0%", "22.7%", "-32.6%"],
                textposition="outside",
                marker_color=["#15803d", "#1e3a8a", "#475569", "#94a3b8"],
                cliponaxis=False
            ))
            fig_r2.update_layout(
                title="Accuracy Comparison (R²)",
                xaxis_title="Model",
                yaxis_title="Accuracy %",
                yaxis=dict(range=[-45, 100])
            )
            fig_r2 = apply_plot_style(fig_r2, height=380)
            fig_r2.update_layout(margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_r2, use_container_width=True, theme=None)

            # Text Table: Accuracy Metrics
            render_html(f"""
<div style="background: {SURFACE_WHITE}; padding: 20px; border-radius: 16px; border: 1px solid {BORDER}; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04); margin-top: 12px;">
<div style="font-size: 0.82rem; font-weight: 800; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 14px;">
    📊 Accuracy Metrics Summary (R²)
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: 0.95rem;">
    <span style="font-weight: 700; color: #166534;">LightGBM (Fire)</span>
    <span style="font-weight: 900; color: #166534;">85.90%</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: 0.95rem;">
    <span style="font-weight: 700; color: {ROYAL_BLUE};">XGBoost (Fire)</span>
    <span style="font-weight: 900; color: {ROYAL_BLUE};">85.03%</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: 0.95rem; color: {TEXT_MUTED};">
    <span>SVR Baseline</span>
    <span style="font-weight: 700;">22.73%</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; font-size: 0.95rem; color: {TEXT_MUTED};">
    <span>Multiple Linear Reg</span>
    <span style="font-weight: 700;">-32.55%</span>
</div>
</div>
            """)

        with bot_col2:
            # Chart: Error Rate Comparison (MAE)
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Bar(
                x=["LightGBM", "XGBoost", "SVR", "Linear Reg"],
                y=[3.21, 3.19, 7.14, 9.35],
                text=["3.21", "3.19", "7.14", "9.35"],
                textposition="outside",
                marker_color=["#15803d", "#1e3a8a", "#475569", "#94a3b8"],
                cliponaxis=False
            ))
            fig_mae.update_layout(
                title="Error Rate Comparison (MAE)",
                xaxis_title="Model",
                yaxis_title="MAE Value",
                yaxis=dict(range=[0, 11])
            )
            fig_mae = apply_plot_style(fig_mae, height=380)
            fig_mae.update_layout(margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_mae, use_container_width=True, theme=None)

            # Text Table: Error Metrics
            render_html(f"""
<div style="background: {SURFACE_WHITE}; padding: 20px; border-radius: 16px; border: 1px solid {BORDER}; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04); margin-top: 12px;">
<div style="font-size: 0.82rem; font-weight: 800; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 14px;">
    📉 Error Rate Metrics Summary (MAE)
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: 0.95rem;">
    <span style="font-weight: 700; color: #166534;">LightGBM (Fire)</span>
    <span style="font-weight: 900; color: #166534;">3.21</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: 0.95rem;">
    <span style="font-weight: 700; color: {ROYAL_BLUE};">XGBoost (Fire)</span>
    <span style="font-weight: 900; color: {ROYAL_BLUE};">3.19</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: 0.95rem; color: {TEXT_MUTED};">
    <span>SVR Baseline</span>
    <span style="font-weight: 700;">7.14</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 8px 0; font-size: 0.95rem; color: {TEXT_MUTED};">
    <span>Multiple Linear Reg</span>
    <span style="font-weight: 700;">9.35</span>
</div>
</div>
            """)

        # 3. Full Stretch Scientific Justification Paragraph at the very bottom
        st.markdown("<br>", unsafe_allow_html=True)
        render_html(f"""
<div style="background: #f0fdf4; border-left: 5px solid #0d9488; padding: 20px; border-radius: 14px; box-shadow: 0 6px 18px rgba(13, 148, 136, 0.04);">
<div style="font-size: 1.05rem; color: #115e59; font-weight: 800; margin-bottom: 6px;">🎯 Methodological Insights & Defense Justification</div>
<div style="font-size: 0.92rem; color: #134e4a; font-weight: 650; line-height: 1.6;">
    By tracking performance from the baseline MLR up to our Champion LightGBM, we scientifically prove that:
    1) MFU's complex localized weather interactions require non-linear ensemble architectures (Boosting tree frameworks significantly outperform linear and support vector alternatives).
    2) The integration of NASA FIRMS real-time hotspot spatial data is absolutely critical to cross the 80% accuracy threshold, closing the variance gap that weather data alone cannot resolve.
</div>
</div>
        """)

    else:
        st.warning("Live model comparison is unavailable until the prediction pipeline is ready.")

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

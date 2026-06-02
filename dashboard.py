import os
import math
import json
import base64
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

# =============================================================================
# FIXED: Using the new official Google GenAI SDK
# =============================================================================
try:
    from google import genai
except ImportError:
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

MFU_RED = "#8C1515"
MFU_GOLD = "#C8963C"
WARM_WHITE = "#f8f6f3"
SURFACE_WHITE = "#ffffff"
TEXT_DARK = "#1f2933"
TEXT_MUTED = "#667085"
BORDER = "#eadfda"
GOOD = "#15803d"
MODERATE = "#b7791f"
UNHEALTHY = "#c2410c"
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
        radial-gradient(circle at top left, rgba(140, 21, 21, 0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(200, 150, 60, 0.12), transparent 22%),
        linear-gradient(180deg, {WARM_WHITE} 0%, #fbfaf8 100%);
    color: {TEXT_DARK};
}}

.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {MFU_RED} 0%, #661010 100%);
}}

[data-testid="stSidebar"] * {{
    color: #ffffff !important;
}}

[data-testid="stMetric"] {{
    background: {SURFACE_WHITE};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 22px rgba(72, 45, 31, 0.08);
}}

[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-weight: 700 !important;
}}

[data-testid="stMetricValue"] {{
    color: {MFU_RED} !important;
    font-weight: 900 !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 6px 16px rgba(72, 45, 31, 0.05);
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 12px;
    padding: 0.7rem 1rem;
    font-weight: 800;
    color: {TEXT_MUTED};
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {MFU_RED}, #a51f1f);
    color: white !important;
}}

.stTabs [aria-selected="true"] p {{
    color: white !important;
}}

div.stButton > button:first-child {{
    background: linear-gradient(135deg, {MFU_RED}, #a51f1f);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.72rem 1.1rem;
    font-weight: 900;
    box-shadow: 0 10px 22px rgba(140, 21, 21, 0.18);
}}

div.stButton > button:first-child:hover {{
    background: linear-gradient(135deg, #751111, {MFU_RED});
    filter: brightness(1.03);
}}

.glass-card {{
    background: {SURFACE_WHITE};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 26px rgba(72, 45, 31, 0.08);
    color: {TEXT_DARK};
}}

.hero-card {{
    border-left: 7px solid {MFU_RED};
}}

.gold-card {{
    border-top: 6px solid {MFU_GOLD};
}}

.red-card {{
    border-top: 6px solid {MFU_RED};
}}

.model-card {{
    background: {SURFACE_WHITE};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 20px rgba(72, 45, 31, 0.07);
}}

.model-card.champion {{
    border-top: 6px solid {MFU_RED};
    background: linear-gradient(180deg, #fffafa 0%, #ffffff 100%);
}}

.model-card.contender {{
    border-top: 6px solid {MFU_GOLD};
}}

.model-card.baseline {{
    border-top: 6px solid #9ca3af;
}}

.header-wrap {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 24px;
    padding-bottom: 20px;
    border-bottom: 2px solid {BORDER};
}}

.logo-img {{
    width: 78px;
    height: auto;
    filter: drop-shadow(0px 4px 8px rgba(0,0,0,0.10));
}}

.header-title {{
    font-size: 36px;
    font-weight: 900;
    color: {MFU_RED};
    margin: 0;
    letter-spacing: -0.7px;
}}

.header-subtitle {{
    font-size: 15px;
    color: {TEXT_MUTED};
    font-weight: 700;
    margin: 4px 0 0 0;
}}

.small-muted {{
    color: {TEXT_MUTED};
    font-weight: 600;
    font-size: 0.92rem;
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
    border-radius: 16px;
    padding: 18px;
}}

.success-box {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 18px;
}}

div[data-testid="stAlert"] {{
    border-radius: 14px;
}}

a {{
    color: {MFU_RED};
}}
</style>
""",
    unsafe_allow_html=True,
)

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
    "Pressure_avg", "Temp_avg", "Humidity_avg", "Precipitation",
    "Sunshine", "Wind_direct", "Wind_speed", "pm25_lag1",
    "pm25_lag2", "pm25_lag3", "pm25_3Day_Avg", "Fire_Count",
    "Fire_Pressure", "Fire_Pressure_Lag1", "Fire_Pressure_Lag2",
    "Fire_Pressure_3Day_Avg", "Month", "Is_Burning_Season",
]

FEATURE_COLS_7 = [
    "Pressure_avg", "Temp_avg", "Humidity_avg", "Precipitation",
    "Sunshine", "Wind_direct", "Wind_speed",
]

MODEL_METRICS = pd.DataFrame(
    [
        {"model": "LightGBM Fire-Integrated", "role": "New SP2 Champion", "features": 18, "r2": 0.8590, "mae": 3.2050, "rmse": 4.7760},
        {"model": "XGBoost Fire-Integrated", "role": "Strong Contender", "features": 18, "r2": 0.8503, "mae": 3.1928, "rmse": 4.9207},
        {"model": "SVR Weather Baseline", "role": "Non-linear Baseline", "features": 7, "r2": 0.2273, "mae": 7.1350, "rmse": 11.1791},
        {"model": "MLR Weather Baseline", "role": "Linear Baseline", "features": 7, "r2": -0.3255, "mae": 9.3503, "rmse": 14.6420},
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

# =============================================================================
# HELPERS
# =============================================================================

def safe_load_model(filename: str):
    paths = [os.path.join(MODEL_DIR, filename), os.path.join(SCRIPT_DIR, filename), filename]
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
            return f"data:image/png;base64,{base64.b64encode(file.read()).decode()}"
    return ""

def get_risk_label(value: float):
    if value <= 25: return "Good", GOOD, "🟢"
    if value <= 50: return "Moderate", MODERATE, "🟡"
    if value <= 100: return "Unhealthy", UNHEALTHY, "🟠"
    return "Hazardous", HAZARDOUS, "🔴"

def health_message(value: float) -> str:
    if value <= 25: return "Air quality is generally acceptable for outdoor activities."
    if value <= 50: return "Sensitive groups should reduce long outdoor exposure."
    if value <= 100: return "Outdoor activity should be reduced. Masks are recommended."
    return "Avoid outdoor activity. Consider indoor air filtration and medical advice if symptoms occur."

def haversine(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
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
        "date": "Date", "PM25": "PM25", "pm25": "PM25", "PM2.5": "PM25",
        "pressure_avg": "Pressure_avg", "temperature_avg": "Temp_avg", "temp_avg": "Temp_avg",
        "humidity_avg": "Humidity_avg", "precipitation": "Precipitation", "sunshine": "Sunshine",
        "wind_direction": "Wind_direct", "wind_direct": "Wind_direct", "wind_speed": "Wind_speed",
        "fire_count": "Fire_Count", "fire_pressure": "Fire_Pressure", "fire_pressure_3day_avg": "Fire_Pressure_3Day_Avg",
        "is_burning_season": "Is_Burning_Season", "month": "Month",
    }
    normalized = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"])
    return normalized

@st.cache_data(ttl=900)
def fetch_weather_and_forecast():
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_KEY": return None, pd.DataFrame()
    try:
        cw = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={MFU_LAT}&lon={MFU_LON}&appid={OPENWEATHER_API_KEY}&units=metric", timeout=8).json()
        fw = requests.get(f"https://api.openweathermap.org/data/2.5/forecast?lat={MFU_LAT}&lon={MFU_LON}&appid={OPENWEATHER_API_KEY}&units=metric", timeout=8).json()
        try:
            a4t = requests.get(f"http://air4thai.pcd.go.th/services/getNewAQI_JSON.php?stationID={AIR4THAI_STATION_ID}", timeout=6).json()
            latest_pm25 = float(a4t["AQILast"]["PM25"]["value"])
            pm25_source = f"Air4Thai station {AIR4THAI_STATION_ID}"
        except:
            pol = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution?lat={MFU_LAT}&lon={MFU_LON}&appid={OPENWEATHER_API_KEY}", timeout=8).json()
            latest_pm25 = float(pol["list"][0]["components"]["pm2_5"])
            pm25_source = "OpenWeather air pollution API"

        current = {
            "temp": float(cw["main"]["temp"]), "humidity": float(cw["main"]["humidity"]),
            "pressure": float(cw["main"]["pressure"]), "wind_speed": float(cw["wind"]["speed"]),
            "wind_direction": float(cw["wind"].get("deg", 0)), "desc": cw["weather"][0]["description"].title(),
            "pm25_current": latest_pm25, "pm25_source": pm25_source,
            "fetch_time": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        }
        
        rows = []
        for item in fw.get("list", []):
            rows.append({
                "datetime": datetime.fromtimestamp(item["dt"]), "Pressure_avg": float(item["main"]["pressure"]),
                "Temp_avg": float(item["main"]["temp"]), "Humidity_avg": float(item["main"]["humidity"]),
                "Precipitation": float(item.get("rain", {}).get("3h", 0)), "Sunshine": 5.0,
                "Wind_direct": float(item["wind"].get("deg", 0)), "Wind_speed": float(item["wind"]["speed"]),
                "pm25_lag1": latest_pm25,
            })
        return current, pd.DataFrame(rows)
    except Exception as exc:
        st.sidebar.warning(f"Weather API failed: {exc}")
        return None, pd.DataFrame()

def fetch_nasa_fire_for_date(date_str: str):
    if NASA_KEY == "YOUR_NASA_KEY": return pd.DataFrame()
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/VIIRS_SNPP_NRT/{NASA_BBOX}/1/{date_str}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return pd.DataFrame()
        df = pd.read_csv(StringIO(response.text))
        if df.empty or "latitude" not in df.columns: return pd.DataFrame()
        df["distance_km"] = df.apply(lambda row: haversine(MFU_LAT, MFU_LON, row["latitude"], row["longitude"]), axis=1)
        df["fire_pressure"] = df["bright_ti4"] / ((df["distance_km"] + 1) ** 2)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_recent_fire_features():
    today = datetime.now()
    frames, pressures, counts = [], [], []
    for offset in range(3):
        date_str = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        daily = fetch_nasa_fire_for_date(date_str)
        if not daily.empty:
            daily["source_date"] = date_str
            frames.append(daily); counts.append(len(daily)); pressures.append(float(daily["fire_pressure"].sum()))
        else:
            counts.append(0); pressures.append(0.0)

    return {
        "hotspots": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        "fire_count": int(counts[0]),
        "fire_pressure": round(pressures[0], 4),
        "fire_pressure_lag1": round(pressures[1], 4),
        "fire_pressure_lag2": round(pressures[2], 4),
        "fire_pressure_3day_avg": round(float(np.mean(pressures)), 4),
    }

@st.cache_data(ttl=600)
def load_historical_data() -> pd.DataFrame:
    candidates = [
        os.path.join(SCRIPT_DIR, "data", "final", "pm25_training_dataset_2018_2022.csv"),
        os.path.join(SCRIPT_DIR, "data", "processed", "chiang_rai_pm25_weather_2018_2022_clean.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try: return normalize_history_columns(pd.read_csv(path))
            except: continue
    return pd.DataFrame()

def build_fire_input(base_df: pd.DataFrame, fire: dict) -> pd.DataFrame:
    df = base_df.copy()
    df["pm25_lag2"] = df["pm25_lag1"]
    df["pm25_lag3"] = df["pm25_lag1"]
    df["pm25_3Day_Avg"] = df["pm25_lag1"]
    df["Fire_Count"] = fire["fire_count"]
    df["Fire_Pressure"] = fire["fire_pressure"]
    df["Fire_Pressure_Lag1"] = fire["fire_pressure_lag1"]
    df["Fire_Pressure_Lag2"] = fire["fire_pressure_lag2"]
    df["Fire_Pressure_3Day_Avg"] = fire["fire_pressure_3day_avg"]
    df["Month"] = datetime.now().month
    df["Is_Burning_Season"] = 1 if datetime.now().month in [2, 3, 4, 5] else 0
    return df[FEATURE_COLS_18]

def predict_log_model(model, df: pd.DataFrame) -> np.ndarray:
    if model is None: return np.zeros(len(df))
    return np.expm1(model.predict(df)).clip(min=0)

def predict_raw_model(model, df: pd.DataFrame) -> np.ndarray:
    if model is None: return np.zeros(len(df))
    return np.asarray(model.predict(df)).clip(min=0)

def make_gistda_map_html(hotspots: pd.DataFrame, current_pred: float, status_text: str, status_icon: str):
    if GISTDA_KEY == "YOUR_GISTDA_KEY": return None
    hotspot_rows = []
    if not hotspots.empty:
        for _, row in hotspots.head(250).iterrows():
            dist = float(row.get("distance_km", 0))
            if dist <= 100: hotspot_rows.append({"lat": float(row["latitude"]), "lon": float(row["longitude"]), "distance": round(dist, 1)})
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>html, body {{ margin: 0; width: 100%; height: 100%; }} #map {{ width: 100%; height: 560px; border-radius: 18px; border: 1px solid #eadfda; }}</style>
        <script src="https://api.sphere.gistda.or.th/map/?key={GISTDA_KEY}"></script>
    </head>
    <body>
        <div id="map"></div>
        <script>
            function loadSphereMap() {{
                if (!window.sphere) return;
                const map = new window.sphere.Map({{ placeholder: document.getElementById("map"), zoom: 9, center: {{ lon: {MFU_LON}, lat: {MFU_LAT} }} }});
                map.Event.bind(window.sphere.EventName.Ready, function () {{
                    const mfu = {{ lon: {MFU_LON}, lat: {MFU_LAT} }};
                    map.Overlays.add(new window.sphere.Marker(mfu, {{
                        title: "Mae Fah Luang University",
                        detail: 'Prediction: {current_pred:.1f} µg/m³<br>Status: {status_text}',
                        icon: {{ html: '<div style="font-size: 30px;">🎓</div>', offset: {{ x: 15, y: 15 }} }}
                    }}));
                    map.Overlays.add(new window.sphere.Circle(mfu, 25000, {{ lineColor: '#C8963C', lineWidth: 2, fillColor: 'rgba(200,150,60,0.08)' }}));
                    map.Overlays.add(new window.sphere.Circle(mfu, 50000, {{ lineColor: '#8C1515', lineWidth: 2, fillColor: 'rgba(140,21,21,0.06)' }}));
                    const hotspots = {json.dumps(hotspot_rows)};
                    hotspots.forEach(pt => {{
                        map.Overlays.add(new window.sphere.Marker({{ lon: pt.lon, lat: pt.lat }}, {{
                            title: "NASA FIRMS fire hotspot", detail: "Distance: " + pt.distance + " km",
                            icon: {{ html: '<div style="font-size: 17px;">🔥</div>', offset: {{ x: 8, y: 8 }} }}
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

# =============================================================================
# FIXED: Using Google GenAI SDK (No more warnings)
# =============================================================================
def generate_llm_warning(language: str, current_pred: float, max_pred: float, status_text: str, current_data: dict, fire_features: dict):
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is missing. Please add it to .streamlit/secrets.toml."

    if genai is None:
        return "google-genai package is not installed. Run: pip install google-genai"

    try:
        # Standard configuration for the new SDK
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
You are an environmental health advisory assistant for Mae Fah Luang University in Chiang Rai, Thailand.

Generate the response strictly in {language}.

Use this data:
- Current predicted PM2.5: {current_pred:.1f} µg/m³
- Maximum predicted PM2.5 in the 5-day forecast: {max_pred:.1f} µg/m³
- Risk status: {status_text}
- Temperature: {current_data.get("temp", 0):.1f} °C
- Humidity: {current_data.get("humidity", 0):.1f}%
- Wind speed: {current_data.get("wind_speed", 0):.1f} m/s
- Active NASA FIRMS hotspots today: {fire_features["fire_count"]}
- Fire pressure index: {fire_features["fire_pressure"]:.2f}

Write a concise advisory for students, staff, and visitors.
Structure:
1. Situation summary
2. Health risk level
3. Recommended actions
4. Outdoor activity advice
5. When to seek medical help

Rules:
- Do not exaggerate beyond the provided data.
- Do not claim official government authority.
- Use calm and practical language.
"""
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as exc:
        return f"AI generation failed: {exc}"


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

mfu_logo = get_base64_img(os.path.join(SCRIPT_DIR, "mfu_logo.png"))
logo_html = f'<img src="{mfu_logo}" class="logo-img">' if mfu_logo else '<div style="font-size:46px;">🎓</div>'

st.markdown(
    f"""
<div class="header-wrap">
    {logo_html}
    <div>
        <h1 class="header-title">Mae Fah Luang University PM2.5 GeoAI Warning System</h1>
        <p class="header-subtitle">Chiang Rai-focused PM2.5 prediction, fire hotspot monitoring, and multilingual health warning dashboard</p>
    </div>
</div>
""", unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## MFU PM2.5 GeoAI")
    st.markdown("**System status**")
    st.write(f"LightGBM Champion: {'Ready' if models['lgbm_fire'] else 'Offline'}")
    st.write(f"XGBoost Contender: {'Ready' if models['xgb_fire'] else 'Offline'}")
    st.write(f"SVR Baseline: {'Ready' if models['svr'] else 'Offline'}")
    st.write(f"MLR Baseline: {'Ready' if models['mlr'] else 'Offline'}")
    st.divider()
    st.markdown("**Data sources**")
    st.write("Air4Thai / OpenWeather\nNASA FIRMS VIIRS\nGISTDA Sphere Map\nGemini AI Advisory")
    st.divider()
    st.markdown("**MFU brand theme**")
    st.write(f"Primary: {MFU_RED}\nAccent: {MFU_GOLD}")

# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

pipeline_ready = current_data is not None and not forecast_df.empty and models["lgbm_fire"] is not None

if pipeline_ready:
    base_input = forecast_df[["Pressure_avg", "Temp_avg", "Humidity_avg", "Precipitation", "Sunshine", "Wind_direct", "Wind_speed", "pm25_lag1"]].copy()
    fire_input = build_fire_input(base_input, fire_features)

    forecast_df["lgbm_pm25"] = predict_log_model(models["lgbm_fire"], fire_input)
    forecast_df["xgb_pm25"] = predict_log_model(models["xgb_fire"], fire_input)
    
    current_7 = base_input[FEATURE_COLS_7]
    forecast_df["svr_pm25"] = predict_raw_model(models["svr"], current_7)
    forecast_df["mlr_pm25"] = predict_raw_model(models["mlr"], current_7)

    forecast_df["predicted_pm25"] = forecast_df["lgbm_pm25"].rolling(2, min_periods=1).mean()

    current_pred = float(forecast_df.iloc[0]["predicted_pm25"])
    max_pred = float(forecast_df["predicted_pm25"].max())
    status_text, status_color, status_icon = get_risk_label(current_pred)
else:
    current_pred = max_pred = 0.0
    status_text, status_color, status_icon = "Offline", TEXT_MUTED, "⚪"

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Prediction & Map", "🤖 LLM Warning", "📊 Charts & Graphs", "🔬 Model Overview"])

with tab1:
    if not pipeline_ready:
        st.error("Prediction pipeline is not ready. Check API keys and required model file: models/lgbm_pm25_model.pkl")
    else:
        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Current prediction", f"{current_pred:.1f} µg/m³", status_text)
        top2.metric("5-day maximum", f"{max_pred:.1f} µg/m³")
        top3.metric("Active hotspots", f"{fire_features['fire_count']}")
        top4.metric("Champion model", "LightGBM", "85.90% R²")

        left, right = st.columns([1.45, 1], gap="large")
        with left:
            st.markdown(f"""
<div class="glass-card hero-card">
    <p class="small-muted">📍 Mae Fah Luang University, Chiang Rai</p>
    <h2 style="margin: 8px 0 0 0; font-size: 28px; color:{TEXT_DARK};">Predicted PM2.5</h2>
    <div style="font-size: 58px; font-weight: 900; color:{status_color}; margin-top: 6px;">
        {current_pred:.1f} <span style="font-size: 22px;">µg/m³</span>
    </div>
    <p style="font-size: 17px; font-weight: 800; color:{TEXT_DARK}; margin-top: 6px;">
        {status_icon} Status: <span style="color:{status_color};">{status_text}</span>
    </p>
    <p class="small-muted">Active engine: LightGBM Fire-Integrated 18-feature model</p>
</div>
""", unsafe_allow_html=True)

            st.markdown("### 5-day PM2.5 forecast trend")
            fig_forecast = px.line(forecast_df, x="datetime", y="predicted_pm25", labels={"datetime": "Date/time", "predicted_pm25": "PM2.5 (µg/m³)"})
            fig_forecast.update_traces(line=dict(color=MFU_RED, width=4), mode="lines+markers")
            fig_forecast.add_hline(y=50, line_dash="dash", line_color="#dc2626", annotation_text="Unhealthy threshold")
            fig_forecast = apply_plot_style(fig_forecast, height=360)
            
            # FIXED: width="stretch" properly implemented
            st.plotly_chart(fig_forecast, width="stretch", theme=None)

        with right:
            st.markdown(f"""
<div class="glass-card gold-card">
    <h3 style="margin-top:0;color:{MFU_RED};">Weather & fire inputs</h3>
    <p><b>Latest PM2.5 source:</b> {current_data.get("pm25_source", "Unknown")}</p>
    <p><b>Current weather:</b> {current_data.get("desc", "Unknown")}</p>
    <p><b>Temperature:</b> {current_data.get("temp", 0):.1f} °C</p>
    <p><b>Humidity:</b> {current_data.get("humidity", 0):.1f}%</p>
    <p><b>Wind:</b> {current_data.get("wind_speed", 0):.1f} m/s, {current_data.get("wind_direction", 0):.0f}°</p>
    <hr style="border: none; border-top: 1px solid {BORDER};">
    <p><b>NASA FIRMS active hotspots:</b> {fire_features["fire_count"]}</p>
    <p><b>Fire pressure:</b> {fire_features["fire_pressure"]:.2f}</p>
    <p><b>3-day avg:</b> {fire_features["fire_pressure_3day_avg"]:.2f}</p>
    <p class="small-muted">Last updated: {current_data.get("fetch_time", "-")}</p>
</div>
<div class="warning-box" style="margin-top:18px;">
    <h4 style="color:{MFU_RED}; margin-top:0;">Health guidance</h4>
    <p style="color:{TEXT_DARK}; font-weight:700;">{health_message(current_pred)}</p>
    <a href="https://website01.mch.mfu.ac.th/mch-index.html" target="_blank" style="font-weight:900;">🏥 MFU Medical Center</a>
</div>
""", unsafe_allow_html=True)

        st.markdown("### Fire hotspot map around MFU")
        map_html = make_gistda_map_html(fire_features["hotspots"], current_pred, status_text, status_icon)
        if map_html:
            components.html(map_html, height=590, scrolling=False)
        else:
            st.warning("GISTDA_KEY is missing. Map cannot be displayed.")
            st.dataframe(fire_features["hotspots"].head(20) if not fire_features["hotspots"].empty else pd.DataFrame({"message": ["No hotspot data available"]}), width="stretch")

with tab2:
    st.markdown(f"""
<div class="glass-card red-card">
    <h2 style="margin-top:0;color:{MFU_RED};">Multilingual AI health advisory</h2>
    <p class="small-muted">Generates practical PM2.5 warning text using current prediction, weather, and NASA FIRMS fire features.</p>
</div>
""", unsafe_allow_html=True)

    if not pipeline_ready:
        st.error("Prediction data is required before generating an advisory.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            language = st.selectbox("Output language", ["English", "Thai", "Chinese", "Burmese"], index=1)
            st.metric("Current predicted PM2.5", f"{current_pred:.1f} µg/m³", status_text)
            st.metric("Active fire hotspots", f"{fire_features['fire_count']}")

            if st.button("Generate AI warning"):
                with st.spinner(f"Generating advisory in {language}..."):
                    st.session_state.ai_report = generate_llm_warning(
                        language, current_pred, max_pred, status_text, current_data, fire_features
                    )
        with c2:
            st.markdown("### AI advisory output")
            if st.session_state.ai_report:
                st.markdown(f'<div class="glass-card">{st.session_state.ai_report}</div>', unsafe_allow_html=True)
            else:
                st.info("Choose a language and click Generate AI warning.")

with tab3:
    st.markdown("## Historical charts and fire-pressure analysis")
    if history_df.empty:
        st.warning("Historical dataset not found.")
    else:
        history_df = history_df.sort_values("Date") if "Date" in history_df.columns else history_df
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if {"Date", "PM25"}.issubset(history_df.columns):
                fig_hist = px.line(history_df, x="Date", y="PM25", title="Historical PM2.5 trend")
                fig_hist.update_traces(line=dict(color=MFU_RED, width=3))
                st.plotly_chart(apply_plot_style(fig_hist, 390), width="stretch", theme=None)
        with c2:
            if {"Temp_avg", "PM25", "Humidity_avg"}.issubset(history_df.columns):
                fig_scatter = px.scatter(history_df, x="Temp_avg", y="PM25", color="Humidity_avg", title="Temperature vs PM2.5", color_continuous_scale="YlOrRd")
                st.plotly_chart(apply_plot_style(fig_scatter, 390), width="stretch", theme=None)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            if {"Date", "Fire_Count"}.issubset(history_df.columns):
                fig_fc = px.bar(history_df, x="Date", y="Fire_Count", title="NASA FIRMS hotspots")
                fig_fc.update_traces(marker_color=MFU_GOLD)
                st.plotly_chart(apply_plot_style(fig_fc, 380), width="stretch", theme=None)
        with c4:
            if {"Date", "Fire_Pressure"}.issubset(history_df.columns):
                fig_fp = px.line(history_df, x="Date", y="Fire_Pressure", title="Fire pressure trend")
                fig_fp.update_traces(line=dict(color=MFU_RED, width=3))
                st.plotly_chart(apply_plot_style(fig_fp, 380), width="stretch", theme=None)

with tab4:
    st.markdown("## Four-model overview and defense")
    st.markdown(f'<div class="glass-card"><h3 style="margin-top:0;color:{MFU_RED};">SP2 model story</h3><p style="font-weight:700;color:{TEXT_DARK};">SP1 used XGBoost as the weather-based champion. In SP2, the project keeps the approved prediction workflow but replaces the criticized dataset with Chiang Rai-focused data and adds NASA FIRMS fire features. With the new fire-integrated 18-feature design, LightGBM becomes the new champion while XGBoost remains the strong contender.</p></div>', unsafe_allow_html=True)
    
    if pipeline_ready:
        st.markdown("### Live prediction comparison")
        current_fire = fire_input.iloc[[0]]
        current_w7 = base_input[FEATURE_COLS_7].iloc[[0]]
        live_lgbm, live_xgb = predict_log_model(models["lgbm_fire"], current_fire)[0], predict_log_model(models["xgb_fire"], current_fire)[0]
        live_svr, live_mlr = predict_raw_model(models["svr"], current_w7)[0], predict_raw_model(models["mlr"], current_w7)[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="model-card champion"><p class="small-muted">🏆 New SP2 Champion</p><h3 style="color:{MFU_RED};">LightGBM</h3><h1 style="color:{MFU_RED};">{live_lgbm:.1f}</h1></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="model-card contender"><p class="small-muted">🔥 Strong Contender</p><h3 style="color:{MFU_GOLD};">XGBoost</h3><h1 style="color:{MFU_GOLD};">{live_xgb:.1f}</h1></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="model-card baseline"><p class="small-muted">📈 Non-linear</p><h3 style="color:{TEXT_DARK};">SVR</h3><h1 style="color:{TEXT_DARK};">{live_svr:.1f}</h1></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="model-card baseline"><p class="small-muted">📏 Linear Base</p><h3 style="color:{TEXT_DARK};">MLR</h3><h1 style="color:{TEXT_DARK};">{live_mlr:.1f}</h1></div>', unsafe_allow_html=True)

    st.markdown("### Performance comparison")
    display_metrics = MODEL_METRICS.copy()
    display_metrics["R² (%)"] = display_metrics["r2"] * 100

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_r2 = px.bar(display_metrics, x="model", y="R² (%)", color="role", title="R² comparison")
        fig_r2.update_layout(showlegend=False)
        st.plotly_chart(apply_plot_style(fig_r2, 420), width="stretch", theme=None)
    with c2:
        fig_mae = px.bar(display_metrics, x="model", y="mae", color="role", title="MAE comparison")
        fig_mae.update_layout(showlegend=False)
        st.plotly_chart(apply_plot_style(fig_mae, 420), width="stretch", theme=None)

    st.dataframe(display_metrics[["model", "role", "features", "r2", "mae", "rmse"]], width="stretch", hide_index=True)
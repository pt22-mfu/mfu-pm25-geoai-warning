import os
import math
import json
import base64
import textwrap
import html
from io import StringIO
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

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

.model-comparison-intro {{
    margin: 34px 0 30px 0;
    padding: 0 6px;
}}

.model-comparison-intro h2 {{
    margin: 0 0 10px 0;
    padding: 0;
    color: #111a2e;
    font-size: 1.55rem;
    font-weight: 750;
    line-height: 1.2;
    letter-spacing: -0.015em;
}}

.model-comparison-intro p {{
    margin: 0;
    padding: 0;
    color: #526176;
    font-size: 0.96rem;
    font-weight: 500;
    line-height: 1.55;
    max-width: 850px;
}}

.chart-analysis-intro {{
    margin: 34px 0 0 0;
    padding: 0 6px;
}}

.chart-analysis-intro h2 {{
    margin: 0 0 10px 0;
    padding: 0;
    color: #111a2e;
    font-size: 1.55rem;
    font-weight: 750;
    line-height: 1.2;
    letter-spacing: -0.015em;
}}

.chart-analysis-intro p {{
    margin: 0;
    padding: 0;
    color: #526176;
    font-size: 0.96rem;
    font-weight: 500;
    line-height: 1.55;
    max-width: 950px;
}}

.system-methodology-intro {{
    margin: 34px 0 0 0;
    padding: 0 6px;
}}

.system-methodology-intro h2 {{
    margin: 0 0 10px 0;
    padding: 0;
    color: #111a2e;
    font-size: 1.55rem;
    font-weight: 750;
    line-height: 1.2;
    letter-spacing: -0.015em;
}}

.system-methodology-intro p {{
    margin: 0;
    padding: 0;
    color: #526176;
    font-size: 0.96rem;
    font-weight: 500;
    line-height: 1.55;
    max-width: 950px;
}}

.system-section-heading {{
    margin: 28px 0 12px 0;
    padding: 0 4px;
}}

.system-section-heading h3 {{
    margin: 0 0 6px 0;
    padding: 0;
    color: #111a2e;
    font-size: 1.22rem;
    font-weight: 750;
    letter-spacing: -0.01em;
}}

.system-section-heading p {{
    margin: 0;
    padding: 0;
    color: #526176;
    font-size: 0.93rem;
    line-height: 1.55;
}}

.pipeline-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    margin: 16px 0 24px 0;
}}

@media (max-width: 900px) {{
    .pipeline-grid {{
        grid-template-columns: 1fr;
    }}
}}

.pipeline-card {{
    background: #ffffff;
    border: 1px solid #d7e0ea;
    border-radius: 14px;
    padding: 22px 22px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}}

.pipeline-card h4 {{
    margin: 0 0 14px 0;
    font-size: 1.05rem;
    font-weight: 750;
    color: #0f172a;
    border-bottom: 2px solid #1e3a8a;
    padding-bottom: 8px;
}}

.pipeline-step {{
    padding: 8px 0;
}}

.pipeline-step-header {{
    font-size: 0.86rem;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 3px;
}}

.pipeline-step-desc {{
    font-size: 0.87rem;
    color: #475569;
    line-height: 1.5;
}}

.pipeline-step-arrow {{
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1;
    margin: 4px 0;
}}

.methodology-table-container {{
    width: 100%;
    overflow-x: auto;
    margin: 14px 0 24px 0;
    border: 1px solid #d7e0ea;
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}}

.methodology-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    text-align: left;
}}

.methodology-table th {{
    background: #f8fafc;
    color: #334155;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 12px 16px;
    border-bottom: 1px solid #d7e0ea;
    white-space: nowrap;
}}

.methodology-table td {{
    color: #1e293b;
    padding: 11px 16px;
    border-bottom: 1px solid #edf2f7;
    line-height: 1.5;
    vertical-align: top;
}}

.methodology-table tr:last-child td {{
    border-bottom: none;
}}

.methodology-code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.82rem;
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 6px;
    border-radius: 5px;
    border: 1px solid #e2e8f0;
    display: inline-block;
    word-break: break-word;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpander"] details {{
    background: #ffffff !important;
    border: 1px solid #d7e0ea !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    overflow: hidden !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpander"] summary {{
    background: #ffffff !important;
    color: #172033 !important;
    border: 0 !important;
    padding: 0.9rem 1rem !important;
    transition: background-color 0.15s ease !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpander"] summary:hover {{
    background: #f8fafc !important;
    color: #111a2e !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpander"] details[open] > summary {{
    background: #f1f5f9 !important;
    color: #111a2e !important;
    border-bottom: 1px solid #e2e8f0 !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpander"] summary p,
.st-key-methodology_notes_expanders div[data-testid="stExpander"] summary span {{
    color: #172033 !important;
    font-weight: 650 !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpander"] summary svg {{
    color: #475569 !important;
    fill: #475569 !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpanderDetails"] {{
    background: #ffffff !important;
    color: #263449 !important;
    padding: 0.25rem 1rem 1rem 1rem !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpanderDetails"] p {{
    color: #263449 !important;
    line-height: 1.7 !important;
}}

.st-key-methodology_notes_expanders div[data-testid="stExpander"] {{
    margin-bottom: 10px !important;
}}

.analysis-section-divider {{
    width: 100%;
    height: 1px;
    margin: 22px 0 30px 0;
    border: 0;
    background: linear-gradient(
        90deg,
        rgba(148, 163, 184, 0) 0%,
        rgba(148, 163, 184, 0.42) 10%,
        rgba(148, 163, 184, 0.42) 90%,
        rgba(148, 163, 184, 0) 100%
    );
}}

.chart-row-divider {{
    width: 100%;
    height: 1px;
    margin: 30px 0 26px 0;
    border: 0;
    background: linear-gradient(
        90deg,
        rgba(148, 163, 184, 0) 0%,
        rgba(148, 163, 184, 0.32) 12%,
        rgba(148, 163, 184, 0.32) 88%,
        rgba(148, 163, 184, 0) 100%
    );
}}

.dashboard-header {{
    width: 100%;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 30px 36px;
    margin: 0 0 28px 0;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d7e0ea;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
}}

.dashboard-header-logo {{
    display: block;
    width: 72px;
    height: 72px;
    flex: 0 0 72px;
    object-fit: contain;
    margin: 0;
}}

.dashboard-header-content {{
    min-width: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 10px;
}}

.dashboard-header-content h1 {{
    margin: 0;
    padding: 0;
    color: #111a2e;
    font-size: clamp(2rem, 3.1vw, 3.15rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.025em;
}}

.dashboard-header-content p {{
    margin: 0;
    padding: 0;
    color: #334155;
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.45;
}}

@media (max-width: 700px) {{
    .dashboard-header {{
        gap: 16px;
        padding: 22px 20px;
        border-radius: 16px;
    }}

    .dashboard-header-logo {{
        width: 54px;
        height: 54px;
        flex-basis: 54px;
    }}

    .dashboard-header-content {{
        gap: 7px;
    }}

    .dashboard-header-content h1 {{
        font-size: clamp(1.55rem, 7vw, 2.1rem);
        line-height: 1.12;
    }}

    .dashboard-header-content p {{
        font-size: 0.86rem;
        line-height: 1.45;
    }}
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
    border-radius: 16px;
    padding: 22px 26px;
    line-height: 1.65;
    color: #0f172a;
    font-size: 0.97rem;
    font-weight: 450;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}}

/* Scoped: white dropdown only for the advisory language selector */
/* Closed select control permanently visible in idle state */
div.st-key-advisory_language_select div[data-baseweb="select"] > div,
div.st-key-advisory_language_select div[role="group"],
div.st-key-advisory_language_select .stSelectbox > div {{
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    min-height: 48px !important;
    color: #0f172a !important;
    box-shadow: none !important;
}}

/* Preserve clear hover state */
div.st-key-advisory_language_select div[data-baseweb="select"] > div:hover,
div.st-key-advisory_language_select div[role="group"]:hover,
div.st-key-advisory_language_select .stSelectbox > div:hover,
div.st-key-advisory_language_select div[role="group"][data-hovered] {{
    border-color: #94a3b8 !important;
}}

/* Preserve blue focus border */
div.st-key-advisory_language_select div[data-baseweb="select"] > div:focus-within,
div.st-key-advisory_language_select div[role="group"]:focus-within,
div.st-key-advisory_language_select .stSelectbox > div:focus-within,
div.st-key-advisory_language_select div[role="group"][data-focus-within] {{
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}}

/* Text styling across Streamlit versions */
div.st-key-advisory_language_select div[data-baseweb="select"] span,
div.st-key-advisory_language_select div[data-baseweb="select"] div,
div.st-key-advisory_language_select div[role="group"] input,
div.st-key-advisory_language_select input {{
    color: #0f172a !important;
}}

/* Ensure dropdown arrow remains visible */
div.st-key-advisory_language_select div[data-baseweb="select"] svg,
div.st-key-advisory_language_select div[role="group"] svg,
div.st-key-advisory_language_select svg {{
    color: #475569 !important;
    fill: #475569 !important;
}}

/* Scoped: advisory output card */
div.st-key-advisory_output_card {{
    background: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid #d7e0ea !important;
    border-left: 4px solid #1e3a8a !important;
    border-radius: 14px !important;
    padding: 1.25rem 1.4rem !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06) !important;
}}

div.st-key-advisory_output_card .stMarkdown {{
    color: #172033 !important;
    font-weight: 400 !important;
    line-height: 1.75 !important;
}}

div.st-key-advisory_output_card .stMarkdown p {{
    margin-bottom: 0.85rem;
}}

div.st-key-advisory_output_card .stMarkdown li {{
    margin-bottom: 0.4rem;
}}

/* Scoped: advisory loading card */
div.st-key-advisory_loading_card {{
    background: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid #d7e0ea !important;
    border-radius: 14px !important;
    padding: 1.25rem 1.4rem !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04) !important;
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

/* =============================================================================
   Streamlit Tab Navigation Styling (Theme-Independent & High-Contrast)
   Targets semantic roles ([role="..."]) and BaseWeb attributes for compatibility.
   Placed at the end of the stylesheet to ensure cascade precedence.
   ============================================================================= */
.stTabs [role="tablist"],
.stTabs div[data-baseweb="tab-list"] {{
    gap: 8px !important;
    padding: 8px !important;
    background: #ffffff !important;
    border: 1px solid #d7e0ea !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05) !important;
    overflow-x: auto !important;
}}

.stTabs [role="tab"],
.stTabs button[data-baseweb="tab"] {{
    background: transparent !important;
    color: #334155 !important;
    border: 0 !important;
    border-radius: 11px !important;
    padding: 0.72rem 1rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}}

.stTabs [role="tab"] *,
.stTabs button[data-baseweb="tab"] * {{
    color: inherit !important;
}}

.stTabs [role="tab"]:hover,
.stTabs button[data-baseweb="tab"]:hover {{
    background: #f1f5f9 !important;
    color: #1e3a8a !important;
}}

.stTabs [role="tab"]:hover *,
.stTabs button[data-baseweb="tab"]:hover * {{
    color: #1e3a8a !important;
}}

.stTabs [role="tab"][aria-selected="true"],
.stTabs button[data-baseweb="tab"][aria-selected="true"] {{
    background: #1e3a8a !important;
    color: #ffffff !important;
    box-shadow: 0 3px 8px rgba(30, 58, 138, 0.18) !important;
}}

.stTabs [role="tab"][aria-selected="true"] *,
.stTabs button[data-baseweb="tab"][aria-selected="true"] * {{
    color: #ffffff !important;
}}

.stTabs [role="tab"]:focus-visible,
.stTabs button[data-baseweb="tab"]:focus-visible {{
    outline: 2px solid #60a5fa !important;
    outline-offset: 2px !important;
}}

.stTabs [data-baseweb="tab-highlight"],
.stTabs div[data-baseweb="tab-highlight"] {{
    background-color: transparent !important;
}}

.stTabs [data-baseweb="tab-border"],
.stTabs div[data-baseweb="tab-border"] {{
    background-color: transparent !important;
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
    <span style="color:{NAVY_BLUE}; font-weight:900;">Advisor - Dr. Khwunta Kirimasthong</span>
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


def aggregate_openweather_daily(
    forecast_payload: dict,
    latest_pm25: float,
    current_weather: dict | None = None,
) -> pd.DataFrame:
    """Convert OpenWeather 3-hour records into daily model inputs.

    The trained PM2.5 models use one row per day, so inference must not send
    individual 3-hour weather records to them. Wind speed is converted from
    OpenWeather m/s to the km/h scale used by the historical dataset. Because
    the API does not provide observed sunshine duration, daytime cloud cover is
    used as a bounded sunshine-hours proxy.
    """
    timezone_offset = int(forecast_payload.get("city", {}).get("timezone", 0))
    rows = []

    for item in forecast_payload.get("list", []):
        local_datetime = datetime.fromtimestamp(
            int(item["dt"]) + timezone_offset,
            tz=timezone.utc,
        ).replace(tzinfo=None)
        cloud_cover = float(item.get("clouds", {}).get("all", 0))
        is_daytime = item.get("sys", {}).get("pod") == "d"

        rows.append(
            {
                "datetime": local_datetime,
                "Pressure_avg": float(item["main"]["pressure"]),
                "Temp_avg": float(item["main"]["temp"]),
                "Humidity_avg": float(item["main"]["humidity"]),
                "Precipitation": float(item.get("rain", {}).get("3h", 0)),
                "Sunshine_component": (
                    3.0 * max(0.0, 1.0 - cloud_cover / 100.0)
                    if is_daytime
                    else 0.0
                ),
                "Wind_direct": float(item["wind"].get("deg", 0)),
                "Wind_speed": float(item["wind"]["speed"]) * 3.6,
                "pm25_lag1": latest_pm25,
            }
        )

    # Add the exact-location current weather observation so the first daily
    # row represents the current-day modeled estimate rather than tomorrow.
    if current_weather:
        local_now = (
            datetime.now(timezone.utc) + timedelta(seconds=timezone_offset)
        ).replace(tzinfo=None)
        rows.append(
            {
                "datetime": local_now,
                "Pressure_avg": float(current_weather["main"]["pressure"]),
                "Temp_avg": float(current_weather["main"]["temp"]),
                "Humidity_avg": float(current_weather["main"]["humidity"]),
                "Precipitation": 0.0,
                "Sunshine_component": 0.0,
                "Wind_direct": float(current_weather.get("wind", {}).get("deg", 0)),
                "Wind_speed": float(current_weather.get("wind", {}).get("speed", 0))
                * 3.6,
                "pm25_lag1": latest_pm25,
            }
        )

    if not rows:
        return pd.DataFrame()

    three_hourly = pd.DataFrame(rows)
    three_hourly["forecast_date"] = three_hourly["datetime"].dt.normalize()

    daily_rows = []
    for forecast_date, day in three_hourly.groupby("forecast_date", sort=True):
        radians = np.deg2rad(day["Wind_direct"].to_numpy())
        circular_wind_direction = (
            np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
            + 360
        ) % 360

        daily_rows.append(
            {
                "datetime": pd.Timestamp(forecast_date),
                "Pressure_avg": float(day["Pressure_avg"].mean()),
                "Temp_avg": float(day["Temp_avg"].mean()),
                "Humidity_avg": float(day["Humidity_avg"].mean()),
                "Precipitation": float(day["Precipitation"].sum()),
                "Sunshine": float(day["Sunshine_component"].sum()),
                "Wind_direct": float(circular_wind_direction),
                "Wind_speed": float(day["Wind_speed"].mean()),
                "pm25_lag1": latest_pm25,
            }
        )

    daily = pd.DataFrame(daily_rows)
    local_today = (
        datetime.now(timezone.utc) + timedelta(seconds=timezone_offset)
    ).date()
    current_and_future = daily[
        daily["datetime"].dt.date >= local_today
    ].head(6).copy()

    # Near midnight the API window may contain fewer than five full future
    # dates. Retain the earliest available daily rows in that case.
    current_and_future["Sunshine"] = current_and_future["Sunshine"].clip(0, 12.4)
    return current_and_future.reset_index(drop=True)


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

        daily_forecast = aggregate_openweather_daily(
            forecast_weather,
            latest_pm25,
            current_weather,
        )
        return current, daily_forecast

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

    # Current fire pressure plus the three prior days are required because the
    # training feature fire_pressure_3day_avg excludes the current day.
    for offset in range(4):
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
    fire_pressure_3day_avg = round(float(np.mean(pressure_values[1:4])), 4)

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

    model_input["pm25_3Day_Avg"] = model_input[
        ["pm25_lag1", "pm25_lag2", "pm25_lag3"]
    ].mean(axis=1)

    model_input["Fire_Count"] = fire_features["fire_count"]
    model_input["Fire_Pressure"] = fire_features["fire_pressure"]
    model_input["Fire_Pressure_Lag1"] = fire_features["fire_pressure_lag1"]
    model_input["Fire_Pressure_Lag2"] = fire_features["fire_pressure_lag2"]
    model_input["Fire_Pressure_3Day_Avg"] = fire_features["fire_pressure_3day_avg"]

    forecast_month = pd.to_datetime(model_input["datetime"]).dt.month
    model_input["Month"] = forecast_month
    model_input["Is_Burning_Season"] = forecast_month.isin([1, 2, 3, 4]).astype(int)

    return model_input[FEATURE_COLS_18]


def predict_log_model(model, input_df: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(input_df))

    raw_predictions = model.predict(input_df)
    return np.expm1(raw_predictions).clip(min=0)


def predict_recursive_log_model(
    model,
    daily_base_input: pd.DataFrame,
    fire_features: dict,
    latest_pm25: float,
) -> np.ndarray:
    """Generate daily projections while advancing PM2.5 lag features."""
    if model is None:
        return np.zeros(len(daily_base_input))

    lag_history = [float(latest_pm25)] * 3
    predictions = []

    for _, base_row in daily_base_input.iterrows():
        one_day = pd.DataFrame([base_row.to_dict()])
        one_day["pm25_lag1"] = lag_history[0]
        one_day["pm25_lag2"] = lag_history[1]
        one_day["pm25_lag3"] = lag_history[2]
        model_input = build_fire_input(one_day, fire_features)
        prediction = float(predict_log_model(model, model_input)[0])
        predictions.append(prediction)
        lag_history = [prediction, lag_history[0], lag_history[1]]

    return np.asarray(predictions)


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
        f"<div><b>Current modeled PM2.5:</b> {current_pred:.1f} &micro;g/m&sup3;</div>"
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
        return "SETUP_REQUIRED::GEMINI_API_KEY is missing. Add GEMINI_API_KEY to .streamlit/secrets.toml before generating the advisory.", None

    if genai is None:
        return "SETUP_REQUIRED::The google-genai package is not installed in this virtual environment. Install it with: python -m pip install google-genai", None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
You are an environmental health advisory assistant for Mae Fah Luang University (MFU) in Chiang Rai, Thailand.

Generate the response strictly in {language}.
Audience: university students, lecturers, staff, and visitors.
Tone: calm, practical, readable, and not like a government order.

Use only this data:
- Current modeled PM2.5 estimate: {current_pred:.1f} µg/m³
- Maximum predicted PM2.5 in the next 5-day forecast: {max_pred:.1f} µg/m³
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
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                response_text = getattr(response, "text", "") or ""
                if response_text.strip():
                    return response_text.strip(), model_name
            except Exception as e:
                # If current model fails, record the error and continue to the next model in the list
                last_error = e
                continue

        # If ALL models in the list fail, only then return the error message
        return f"AI_ERROR::AI generation failed across all fallback models. Last error: {last_error}", None

    except Exception as exc:
        return f"AI_ERROR::System error during AI generation setup: {exc}", None



# =============================================================================
# LOAD DATA / MODELS
# =============================================================================

models = load_models()
current_data, forecast_df = fetch_weather_and_forecast()
fire_features = fetch_recent_fire_features()
history_df = load_historical_data()

if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""
if "ai_report_language" not in st.session_state:
    st.session_state.ai_report_language = ""
if "ai_report_model" not in st.session_state:
    st.session_state.ai_report_model = ""
if "ai_report_generated_at" not in st.session_state:
    st.session_state.ai_report_generated_at = ""


# =============================================================================
# HEADER / SIDEBAR
# =============================================================================

# Using the official MFU logo URL provided by PT
mfu_logo_url = "https://archives.mfu.ac.th/wp-content/uploads/2019/06/Mae-Fah-Luang-University-2.png"

st.markdown(
    f"""
<div class="dashboard-header">
    <img src="{mfu_logo_url}" alt="MFU Logo" class="dashboard-header-logo">
    <div class="dashboard-header-content">
        <h1>MFU PM2.5 GeoAI Warning Dashboard</h1>
        <p>Chiang Rai localized prediction · NASA FIRMS fire monitoring · GISTDA spatial visualization · AI campus advisory</p>
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
            "datetime",
            "Pressure_avg",
            "Temp_avg",
            "Humidity_avg",
            "Precipitation",
            "Sunshine",
            "Wind_direct",
            "Wind_speed",
        ]
    ].copy()

    forecast_df["lgbm_pm25"] = predict_recursive_log_model(
        models["lgbm_fire"],
        base_input,
        fire_features,
        current_data["pm25_current"],
    )
    forecast_df["xgb_pm25"] = predict_recursive_log_model(
        models["xgb_fire"],
        base_input,
        fire_features,
        current_data["pm25_current"],
    )

    current_7 = base_input[FEATURE_COLS_7]
    forecast_df["svr_pm25"] = predict_raw_model(models["svr"], current_7)
    forecast_df["mlr_pm25"] = predict_raw_model(models["mlr"], current_7)

    forecast_df["predicted_pm25"] = forecast_df["lgbm_pm25"]

    current_pred = float(forecast_df.iloc[0]["predicted_pm25"])
    future_forecast_df = forecast_df.iloc[1:6].copy()
    forecast_scope = future_forecast_df if not future_forecast_df.empty else forecast_df
    max_pred = float(forecast_scope["predicted_pm25"].max())
    avg_pred = float(forecast_scope["predicted_pm25"].mean())
    status_text, status_color, status_icon = get_risk_label(current_pred)
else:
    future_forecast_df = pd.DataFrame()
    current_pred = 0.0
    max_pred = 0.0
    avg_pred = 0.0
    status_text, status_color, status_icon = "Offline", TEXT_MUTED, "⚪"


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🗺️ Prediction & Map",
        "🤖 AI Advisory(Gemini)",
        "📊 Charts & Graphs",
        "🔬 Model Overview",
        "⚙️ System & Methodology",
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
    <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 8px;">
        <div style="font-size: 1.2rem; font-weight: 800; color: {TEXT_DARK};">Current PM2.5 estimate:</div>
        <div style="font-size: 2.85rem; font-weight: 900; color: {status_color}; letter-spacing: -0.04em; line-height: 1;">
            {current_pred:.1f} <span style="font-size: 1.2rem; font-weight: 700;">µg/m³</span>
        </div>
    </div>
    <div style="margin-top: 8px; font-size: 0.95rem; color: {TEXT_DARK};">
        Air quality: <strong style="color: {status_color};">{status_text}</strong>
    </div>
    <div style="margin-top: 4px; font-size: 0.8rem; color: {TEXT_MUTED};">
        Estimated using recent air quality, weather, and nearby fire conditions.
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
            st.iframe(map_html, width="stretch", height=520)
        else:
            st.warning("GISTDA_KEY is missing. Map cannot be displayed.")


        # --- 3. BOTTOM ROW: Charts vs Weather & Forecast Summaries ---
        st.markdown("<br>", unsafe_allow_html=True)
        bot_left, bot_right = st.columns([1.65, 1], gap="large")

        with bot_left:
            # Forecast Line Chart
            st.markdown('<div class="section-title" style="margin-top: 0px; font-size: 1.15rem;">📈 5-Day PM2.5 Forecast Trend</div>', unsafe_allow_html=True)
            fig_forecast = px.line(
                future_forecast_df,
                x="datetime",
                y="predicted_pm25",
                labels={"datetime": "Forecast date", "predicted_pm25": "PM2.5 (µg/m³)"},
            )
            fig_forecast.update_traces(line=dict(color=ROYAL_BLUE, width=3), mode="lines+markers", marker=dict(size=5))
            fig_forecast.add_hline(
                y=50,
                line_dash="dash",
                line_color=UNHEALTHY,
                annotation_text="Unhealthy threshold (50)",
                annotation_position="top left",
            )
            fig_forecast = apply_plot_style(fig_forecast, height=320)
            fig_forecast.update_layout(xaxis_title=None, margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_forecast, width="stretch", theme=None)
            st.caption(
                "Projection uses daily weather aggregates and assumes the current "
                "NASA FIRMS fire conditions persist across the five-day horizon."
            )

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
            future_forecast_df['Date_Str'] = future_forecast_df['datetime'].dt.strftime('%d %A')
            daily_summary = future_forecast_df.groupby('Date_Str', sort=False)['predicted_pm25'].max().reset_index().head(5)
            
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

    if not pipeline_ready:
        st.error("Prediction data is required before generating an advisory.")
    else:
        left, right = st.columns([0.82, 1.75], gap="large")

        with left:
            active_model = st.session_state.get("ai_report_model", "")
            if active_model:
                escaped_model = html.escape(active_model)
                ai_footer_text = f"AI model used: {escaped_model} · fallback enabled"
            else:
                ai_footer_text = "AI advisory: Gemini API · automatic fallback"

            render_html(f"""
            <div style="width: 100%; box-sizing: border-box; background: rgba(255, 255, 255, 0.96); border: 1px solid #d7e0ea; border-left: 4px solid #1e3a8a; border-radius: 14px; padding: 18px 20px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05); margin-top: 8px;">
                <div style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.07em; color: #0f172a; text-transform: uppercase; margin-bottom: 8px;">
                    Current PM2.5 estimate
                </div>
                <div style="font-size: 2.35rem; font-weight: 750; line-height: 1.05; color: {status_color}; margin: 0 0 6px 0;">
                    {current_pred:.1f} <span style="font-size: 1.15rem; font-weight: 600;">µg/m³</span>
                </div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #475569; line-height: 1.35;">
                    {status_icon} {status_text} · max forecast {max_pred:.1f} µg/m³
                </div>
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 16px 0 14px 0;">
                <div style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.07em; color: #0f172a; text-transform: uppercase; margin-bottom: 10px;">
                    MONITORING CONTEXT
                </div>
                <div style="display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 10px; padding: 3px 0; font-size: 0.9rem; align-items: baseline;">
                    <span style="color: #64748b; font-weight: 600;">Weather</span>
                    <span style="color: #172033; font-weight: 500; word-break: break-word;">{current_data.get("desc", "Unknown")}</span>
                </div>
                <div style="display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 10px; padding: 3px 0; font-size: 0.9rem; align-items: baseline;">
                    <span style="color: #64748b; font-weight: 600;">Wind</span>
                    <span style="color: #172033; font-weight: 500; word-break: break-word;">{current_data.get("wind_speed", 0):.1f} m/s at {current_data.get("wind_direction", 0):.0f}°</span>
                </div>
                <div style="display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 10px; padding: 3px 0; font-size: 0.9rem; align-items: baseline;">
                    <span style="color: #64748b; font-weight: 600;">NASA hotspots</span>
                    <span style="color: #172033; font-weight: 500; word-break: break-word;">{fire_features["fire_count"]}</span>
                </div>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed #e2e8f0; font-size: 0.82rem; color: #64748b; line-height: 1.45;">
                    {ai_footer_text}
                </div>
            </div>
            """)

            if genai is None:
                render_html("""
                <div class="setup-card">
                    Gemini advisory is not ready in this virtual environment because <code>google-genai</code> is not installed.<br>
                    Install command:<br><code>python -m pip install google-genai</code>
                </div>
                """)

        with right:
            st.markdown('<div class="section-title" style="margin-top:8px;">Advisory output</div>', unsafe_allow_html=True)

            # Language selector label (native, no pill)
            st.markdown(
                '<p style="margin:0 0 4px 0; font-size:0.85rem; font-weight:650; color:#475569;">Output language</p>',
                unsafe_allow_html=True,
            )

            language = st.selectbox(
                "Output language",
                ["English", "Thai", "Burmese", "Chinese"],
                index=1,
                label_visibility="collapsed",
                key="advisory_language_select",
            )

            generate_btn = st.button("Generate campus advisory", width="stretch", key="advisory_generate_btn")
            advisory_slot = st.empty()

            if generate_btn:
                with advisory_slot.container():
                    with st.container(key="advisory_loading_card"):
                        with st.spinner("⏳ Just a moment — generating the campus advisory..."):
                            report, model_used = generate_llm_warning(
                                language=language,
                                current_pred=current_pred,
                                max_pred=max_pred,
                                status_text=status_text,
                                current_data=current_data,
                                fire_features=fire_features,
                            )
                            st.session_state.ai_report = report
                            if report and not (report.startswith("SETUP_REQUIRED::") or report.startswith("AI_ERROR::")):
                                generated_at = datetime.now(ZoneInfo("Asia/Bangkok"))
                                st.session_state.ai_report_model = model_used or ""
                                st.session_state.ai_report_language = language
                                st.session_state.ai_report_generated_at = generated_at.isoformat()
                            else:
                                st.session_state.ai_report_model = ""
                                st.session_state.ai_report_language = ""
                                st.session_state.ai_report_generated_at = ""
                advisory_slot.empty()

            with advisory_slot.container():
                if st.session_state.ai_report:
                    if st.session_state.ai_report.startswith("SETUP_REQUIRED::"):
                        message = st.session_state.ai_report.replace("SETUP_REQUIRED::", "")
                        render_html(f"""
                        <div class="setup-card" style="margin-top: 1rem;">
                            <strong>Setup required</strong><br>{message}
                        </div>
                        """)
                    elif st.session_state.ai_report.startswith("AI_ERROR::"):
                        message = st.session_state.ai_report.replace("AI_ERROR::", "")
                        render_html(f"""
                        <div class="setup-card" style="margin-top: 1rem;">
                            <strong>AI advisory could not be generated.</strong><br>{message}
                        </div>
                        """)
                    else:
                        caption_parts = ["Generated campus advisory"]
                        generated_language = st.session_state.get("ai_report_language", "")
                        if generated_language:
                            caption_parts.append(generated_language)

                        gen_time_str = st.session_state.get("ai_report_generated_at")
                        if gen_time_str:
                            try:
                                dt = datetime.fromisoformat(gen_time_str)
                                hour = dt.strftime("%I").lstrip("0") or "12"
                                formatted_ts = f"{dt.day} {dt.strftime('%B %Y')}, {hour}:{dt.strftime('%M %p')} ICT"
                                caption_parts.append(formatted_ts)
                            except Exception:
                                pass

                        generated_model = st.session_state.get("ai_report_model", "")
                        if generated_model:
                            caption_parts.append(f"Model: {generated_model}")

                        caption_text = " · ".join(caption_parts)
                        with st.container(key="advisory_output_card"):
                            st.caption(caption_text)
                            st.markdown(st.session_state.ai_report)
                else:
                    render_html("""
                    <div style="background: rgba(255, 255, 255, 0.96); border: 1px solid #d7e0ea; border-radius: 14px; padding: 1.25rem 1.4rem; margin-top: 1rem; color: #64748b; font-size: 0.95rem; line-height: 1.65; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);">
                        Choose a language and click <strong>Generate campus advisory</strong>. The output will appear here as a readable situational analysis followed by practical recommendations.
                    </div>
                    """)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# =============================================================================
# TAB 3: CHARTS & GRAPHS
# =============================================================================

with tab3:
    render_html("""
<div class="chart-analysis-intro">
    <h2>Historical trends and fire-pressure analysis</h2>
    <p>Historical PM2.5 patterns, fire activity, weather relationships, and seasonal behavior for the Chiang Rai-focused dataset.</p>
</div>
<div class="analysis-section-divider"></div>
    """)

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
                st.plotly_chart(fig_hist, width="stretch", theme=None)

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
                st.plotly_chart(fig_scatter, width="stretch", theme=None)

        render_html('<div class="chart-row-divider"></div>')

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
                st.plotly_chart(fig_fire_count, width="stretch", theme=None)

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
                st.plotly_chart(fig_fire_pressure, width="stretch", theme=None)

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
            st.plotly_chart(fig_combined, width="stretch", theme=None)

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
            st.plotly_chart(fig_month, width="stretch", theme=None)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# =============================================================================
# TAB 4: MODEL OVERVIEW
# =============================================================================

with tab4:
    render_html("""
<div class="model-comparison-intro">
    <h2>Model Comparison and Selection</h2>
    <p>Comparison of four candidate models supporting the selection of fire-integrated LightGBM.</p>
</div>
    """)

    if pipeline_ready:
        first_projection = forecast_df.iloc[0]
        live_lgbm = float(first_projection["lgbm_pm25"])
        live_xgb = float(first_projection["xgb_pm25"])
        live_svr = float(first_projection["svr_pm25"])
        live_mlr = float(first_projection["mlr_pm25"])

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
            st.plotly_chart(fig_r2, width="stretch", theme=None)

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
            st.plotly_chart(fig_mae, width="stretch", theme=None)

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


# =============================================================================
# TAB 5: SYSTEM & METHODOLOGY
# =============================================================================

with tab5:
    render_html("""
<div class="system-methodology-intro">
<h2>System Architecture and Methodology</h2>
<p>Technical workflow, data sources, feature engineering, model development, deployment, and decision-support components of the MFU PM2.5 GeoAI Warning Platform.</p>
</div>
<div class="analysis-section-divider"></div>

<div class="system-section-heading">
<h3>Platform purpose</h3>
<p>This platform combines localized weather conditions, historical PM2.5 observations, NASA FIRMS fire activity, machine-learning estimation, spatial visualization, and multilingual AI-generated guidance to support air-quality awareness at Mae Fah Luang University.</p>
</div>
<div class="chart-row-divider"></div>

<div class="system-section-heading">
<h3>How the system works</h3>
<p>A structural separation between offline model development and the live deployed prediction and advisory pipeline.</p>
</div>

<div class="pipeline-grid">
<div class="pipeline-card">
<h4>Model-development pipeline</h4>
<div class="pipeline-step">
<div class="pipeline-step-header">01 — Historical data acquisition</div>
<div class="pipeline-step-desc">Historical PM2.5 observations, weather variables, and NASA FIRMS fire activity are collected for model development.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">02 — Daily alignment and validation</div>
<div class="pipeline-step-desc">The sources are aligned to a common daily structure and checked according to the actual preprocessing implemented in the repository.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">03 — Feature engineering</div>
<div class="pipeline-step-desc">Weather variables, PM2.5 lag features, temporal indicators, and fire-pressure features are prepared for model training.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">04 — Chronological evaluation</div>
<div class="pipeline-step-desc">Evaluation follows a strict chronological split: training on 2018–2021 (1,455 daily observations) and holdout testing on 2022 (365 daily observations). Chronological separation eliminates future-to-past information leakage and accurately reflects real-world operational deployment.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">05 — Controlled model comparison</div>
<div class="pipeline-step-desc">LightGBM and XGBoost are evaluated using weather-only and weather-plus-fire variants under identical train/test splits and target transformations.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">06 — Selected predictive model</div>
<div class="pipeline-step-desc">The deployed numerical estimator uses the confirmed fire-integrated LightGBM model.</div>
</div>
</div>
<div class="pipeline-card">
<h4>Deployed prediction and advisory pipeline</h4>
<div class="pipeline-step">
<div class="pipeline-step-header">01 — Live contextual inputs</div>
<div class="pipeline-step-desc">Collect the most recent available PM2.5 lag input, MFU-coordinate weather and wind information, and recent NASA FIRMS fire activity.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">02 — Feature alignment</div>
<div class="pipeline-step-desc">Convert and aggregate live inputs so their units and daily semantics match the trained model features.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">03 — Current localized estimate</div>
<div class="pipeline-step-desc">Generate the current model-based PM2.5 estimate for the MFU location. This is an engineered model estimate for the campus valley, not a direct on-campus physical sensor measurement.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">04 — Five-day daily scenario projection</div>
<div class="pipeline-step-desc">Generate the following five daily values recursively by updating PM2.5 lag features with previous model outputs. This represents a daily scenario projection, not an official regulatory forecast.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">05 — Spatial and visual context</div>
<div class="pipeline-step-desc">Present the prediction, weather, fire information, charts, and GISTDA spatial visualization in the Streamlit interface.</div>
</div>
<div class="pipeline-step-arrow">&darr;</div>
<div class="pipeline-step">
<div class="pipeline-step-header">06 — Multilingual campus advisory</div>
<div class="pipeline-step-desc">Provide the structured prediction and environmental context to the Gemini API to generate practical guidance in the selected language. Gemini explains the model output; it does not calculate the numerical PM2.5 prediction.</div>
</div>
</div>
</div>
<div class="chart-row-divider"></div>

<div class="system-section-heading">
<h3>Model inputs and engineered factors</h3>
<p>Eighteen confirmed input features categorized into autoregressive history, meteorology, temporal seasonality, and spatial fire pressure.</p>
</div>

<div class="methodology-table-container">
<table class="methodology-table">
<thead>
<tr>
<th style="width: 18%;">Category</th>
<th style="width: 24%;">Engineered Feature</th>
<th style="width: 28%;">Display Label & Semantics</th>
<th style="width: 30%;">Operational Definition</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4" style="font-weight: 700; color: #1e3a8a; background: #fafcff;">PM2.5 History</td>
<td><span class="methodology-code">pm25_lag1</span></td>
<td>1-Day Prior PM2.5</td>
<td>Previous day observed PM2.5 concentration (&micro;g/m&sup3;)</td>
</tr>
<tr>
<td><span class="methodology-code">pm25_lag2</span></td>
<td>2-Day Prior PM2.5</td>
<td>Two days prior observed PM2.5 concentration (&micro;g/m&sup3;)</td>
</tr>
<tr>
<td><span class="methodology-code">pm25_lag3</span></td>
<td>3-Day Prior PM2.5</td>
<td>Three days prior observed PM2.5 concentration (&micro;g/m&sup3;)</td>
</tr>
<tr>
<td><span class="methodology-code">pm25_3day_avg</span></td>
<td>3-Day Moving Average</td>
<td>Rolling 3-day arithmetic mean of antecedent PM2.5 values (&micro;g/m&sup3;)</td>
</tr>
<tr>
<td rowspan="7" style="font-weight: 700; color: #1e3a8a; background: #fafcff;">Weather</td>
<td><span class="methodology-code">pressure_avg</span></td>
<td>Atmospheric Pressure</td>
<td>Mean daily barometric surface pressure (hPa)</td>
</tr>
<tr>
<td><span class="methodology-code">temperature_avg</span></td>
<td>Ambient Temperature</td>
<td>Mean daily surface air temperature (&deg;C)</td>
</tr>
<tr>
<td><span class="methodology-code">humidity_avg</span></td>
<td>Relative Humidity</td>
<td>Mean daily relative atmospheric humidity (%)</td>
</tr>
<tr>
<td><span class="methodology-code">precipitation</span></td>
<td>Daily Precipitation</td>
<td>Total 24-hour liquid precipitation accumulation (mm)</td>
</tr>
<tr>
<td><span class="methodology-code">sunshine</span></td>
<td>Sunshine Duration</td>
<td>Estimated daily sunshine duration (hours, bounded at 12.4 h)</td>
</tr>
<tr>
<td><span class="methodology-code">wind_direction</span></td>
<td>Wind Direction</td>
<td>Circular vector mean compass wind direction (0&deg;&ndash;360&deg;)</td>
</tr>
<tr>
<td><span class="methodology-code">wind_speed</span></td>
<td>Wind Speed</td>
<td>Mean daily horizontal wind velocity (km/h)</td>
</tr>
<tr>
<td rowspan="2" style="font-weight: 700; color: #1e3a8a; background: #fafcff;">Temporal</td>
<td><span class="methodology-code">month</span></td>
<td>Calendar Month</td>
<td>Month index (1&ndash;12) capturing broad annual meteorological seasonality</td>
</tr>
<tr>
<td><span class="methodology-code">is_burning_season</span></td>
<td>Burning Season Indicator</td>
<td>Binary indicator (1 if month &isin; [January, February, March, April]; 0 otherwise)</td>
</tr>
<tr>
<td rowspan="5" style="font-weight: 700; color: #1e3a8a; background: #fafcff;">Fire Activity</td>
<td><span class="methodology-code">fire_count</span></td>
<td>Active Fire Count</td>
<td>Daily count of NASA FIRMS VIIRS hotspots detected within 100 km radius</td>
</tr>
<tr>
<td><span class="methodology-code">fire_pressure</span></td>
<td>Fire Pressure Index</td>
<td>Inverse-square distance-decay weighted fire intensity: &sum; [brightness / (distance_km + 1)&sup2;]</td>
</tr>
<tr>
<td><span class="methodology-code">fire_pressure_lag1</span></td>
<td>1-Day Prior Fire Pressure</td>
<td>Previous day spatial fire pressure index</td>
</tr>
<tr>
<td><span class="methodology-code">fire_pressure_lag2</span></td>
<td>2-Day Prior Fire Pressure</td>
<td>Two days prior spatial fire pressure index</td>
</tr>
<tr>
<td><span class="methodology-code">fire_pressure_3day_avg</span></td>
<td>3-Day Moving Fire Pressure</td>
<td>Rolling 3-day mean of the spatial fire pressure index</td>
</tr>
</tbody>
</table>
</div>

<p style="font-size: 0.88rem; color: #526176; line-height: 1.55; margin: -10px 4px 24px 4px;">
<strong>Methodological note:</strong> The burning-season indicator strictly follows the regional biomass burning definition (January through April). Features have varying predictive importance; non-linear boosting models place highest split importance on autoregressive PM2.5 lags and spatial fire-pressure metrics during high-pollution events.
</p>
<div class="chart-row-divider"></div>

<div class="system-section-heading">
<h3>Component responsibilities</h3>
<p>Operational roles of each confirmed data source, computational engine, and presentation layer in the platform architecture.</p>
</div>

<div class="methodology-table-container">
<table class="methodology-table">
<thead>
<tr>
<th style="width: 25%;">Component</th>
<th style="width: 75%;">Responsibility in System Architecture</th>
</tr>
</thead>
<tbody>
<tr>
<td style="font-weight: 750; color: #0f172a;">Air4Thai (PCD)</td>
<td>Provides official historical and recent ambient PM2.5 observations (station 73t, Chiang Rai) used as initial lag inputs by the model workflow. The displayed current MFU value is a localized model estimate, not a direct Air4Thai reading.</td>
</tr>
<tr>
<td style="font-weight: 750; color: #0f172a;">OpenWeather API</td>
<td>Provides localized MFU-coordinate (20.0443&deg; N, 99.8924&deg; E) weather and wind inputs used by the deployed feature-building and 5-day daily aggregation engine.</td>
</tr>
<tr>
<td style="font-weight: 750; color: #0f172a;">NASA FIRMS</td>
<td>Provides VIIRS active fire hotspot detections, brightness temperatures, and radiative power within 100 km of MFU, used to construct spatial fire-pressure metrics and map layers.</td>
</tr>
<tr>
<td style="font-weight: 750; color: #0f172a;">LightGBM</td>
<td>Produces the numerical localized PM2.5 estimate and recursive daily scenario projections using the confirmed 18 engineered features.</td>
</tr>
<tr>
<td style="font-weight: 750; color: #0f172a;">XGBoost</td>
<td>Serves as a comparison model in the controlled evaluation, confirming that gradient-boosted decision trees effectively capture non-linear pollution dynamics.</td>
</tr>
<tr>
<td style="font-weight: 750; color: #0f172a;">GISTDA Sphere API</td>
<td>Provides spatial visualization and environmental context by rendering campus boundaries, distance buffers (25 km, 50 km, 100 km), wind vectors, and active hotspot positions. GISTDA does not directly train the PM2.5 model.</td>
</tr>
<tr>
<td style="font-weight: 750; color: #0f172a;">Gemini API</td>
<td>Generates multilingual situational analysis and practical recommendations from structured model and environmental inputs. It explains the model output and does not calculate the numerical PM2.5 prediction.</td>
</tr>
<tr>
<td style="font-weight: 750; color: #0f172a;">Streamlit</td>
<td>Integrates predictions, maps, charts, model evidence, and advisory output into the deployed user interface.</td>
</tr>
</tbody>
</table>
</div>
<div class="chart-row-divider"></div>

<div class="system-section-heading">
<h3>Technology stack</h3>
<p>Confirmed software libraries, frameworks, external APIs, and deployment infrastructure used in the repository.</p>
</div>

<div class="methodology-table-container">
<table class="methodology-table">
<thead>
<tr>
<th style="width: 25%;">Category</th>
<th style="width: 75%;">Verified Technologies & Roles</th>
</tr>
</thead>
<tbody>
<tr>
<td style="font-weight: 750; color: #1e3a8a;">Application</td>
<td>Python 3.10+ / 3.13, Streamlit (web framework), Joblib (model persistence)</td>
</tr>
<tr>
<td style="font-weight: 750; color: #1e3a8a;">Data processing</td>
<td>Pandas (daily aggregation and lag generation), NumPy (circular trigonometry and vector math), Requests (HTTP client for live external services)</td>
</tr>
<tr>
<td style="font-weight: 750; color: #1e3a8a;">Machine learning</td>
<td>LightGBM (champion LGBMRegressor), XGBoost (comparison XGBRegressor), scikit-learn (SVR, Linear Regression, evaluation metrics)</td>
</tr>
<tr>
<td style="font-weight: 750; color: #1e3a8a;">Visualization</td>
<td>Plotly (Plotly Graph Objects & Express for historical and forecast charts), GISTDA Sphere API (interactive campus JavaScript map)</td>
</tr>
<tr>
<td style="font-weight: 750; color: #1e3a8a;">External data services</td>
<td>OpenWeather API (current & forecast weather), NASA FIRMS Area API (VIIRS fire detections), Air4Thai API (ambient station 73t)</td>
</tr>
<tr>
<td style="font-weight: 750; color: #1e3a8a;">Generative AI</td>
<td>Gemini API (<span class="methodology-code">Google GenAI SDK</span> with automatic candidate fallback sequence)</td>
</tr>
<tr>
<td style="font-weight: 750; color: #1e3a8a;">Deployment</td>
<td>Streamlit Community Cloud (managed hosting with <span class="methodology-code">.streamlit/secrets.toml</span> configuration)</td>
</tr>
</tbody>
</table>
</div>
<div class="chart-row-divider"></div>
""")

    st.markdown('<div class="system-section-heading"><h3>Methodological notes</h3><p>Key technical rationale and constraints for committee defense and academic presentation.</p></div>', unsafe_allow_html=True)

    with st.container(key="methodology_notes_expanders"):
        with st.expander("What does “Current Modeled PM2.5” mean?", expanded=False):
            st.markdown(
                "The displayed current value is a localized model-based estimate generated using the most recent available PM2.5 lag information together with MFU-coordinate weather, wind, temporal, and fire-related inputs. It is not a direct Air4Thai display value and is not an on-campus regulatory sensor measurement."
            )

        with st.expander("Why were NASA fire factors included?", expanded=False):
            st.markdown(
                "Weather-only models may not fully represent pollution behavior during the northern Thailand burning season. NASA FIRMS-derived features were therefore evaluated through a controlled ablation study comparing weather-only and weather-plus-fire variants.\n\n"
                "Fire-related features allow the non-linear boosting models to account for regional biomass burning pressure that local meteorological sensors cannot detect. The measured evidence is detailed in the Model Overview tab."
            )

        with st.expander("Why use chronological holdout testing?", expanded=False):
            st.markdown(
                "The model is trained on earlier dates (2018–2021) and evaluated on the later 2022 period. This better represents future deployment and reduces the risk of information from the future leaking into model training, unlike random cross-validation which allows future observations to inform past predictions."
            )

        with st.expander("What are the limitations of the five-day projection?", expanded=False):
            st.markdown(
                "The five-day values are daily scenario projections based on forecast environmental inputs and recursive PM2.5 lag updates. Uncertainty can accumulate across later days, and the output should not be presented as an official regulatory forecast."
            )

        with st.expander("What role does generative AI play?", expanded=False):
            st.markdown(
                "The Gemini component receives structured prediction, weather, wind, and fire context and converts it into readable multilingual guidance. It does not alter or calculate the numerical PM2.5 model output. The application uses a model fallback sequence for service resilience."
            )

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

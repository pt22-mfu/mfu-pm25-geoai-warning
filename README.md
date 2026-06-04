# 🌫️ MFU PM2.5 GeoAI Warning System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM%20%7C%20XGBoost-orange.svg)
![Generative AI](https://img.shields.io/badge/Generative%20AI-Gemini%20Flash-purple.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-85.9%25%20(R²)-success.svg)

**Developed by:** The Outliers (Computer Engineering, MFU)  
**Advisor:** Aj. Khwunta Kirimasthong  
**Live Dashboard:** [https://mfu-pm25-geoai-warning-system.streamlit.app/](https://mfu-pm25-geoai-warning-system.streamlit.app/)

## 📌 Project Overview
Northern Thailand frequently experiences severe air quality degradation due to seasonal agricultural burning and microclimate effects. Traditional weather-only models often fail to predict PM2.5 spikes in localized valleys accurately. 

This project is an advanced **GeoAI Warning System** designed to provide high-accuracy, real-time PM2.5 forecasting specifically for the **Mae Fah Luang University (MFU) Valley**. By strictly integrating real-time meteorological data with spatial fire pressure indexes (NASA FIRMS), this engine feeds a trained **LightGBM Champion model** to present proactive 5-day forecasts, GISTDA spatial visualizations, and LLM-powered (Gemini) situational advisories via an interactive Streamlit web dashboard.

## 🏗️ System Architecture & Data Layers
The core strength of this project lies in its multi-layered data integration and robust feature engineering (18 unique features):

1. **Weather Layer:** Real-time data from **OpenWeatherMap API** and **Air4Thai**.
2. **Spatial Fire Layer:** Dynamic hotspot acquisition from **NASA FIRMS (VIIRS)**, engineered into Fire Count and Distance-Decay Fire Pressure metrics.
3. **Visualization Layer:** Interactive localized campus maps driven by the **GISTDA Sphere API**.
4. **Advisory Layer:** Automated situational analysis generated in multiple languages using **Google Gemini AI**.

## 🧠 Machine Learning Performance (4-Model Defense Showdown)
We evaluated multiple models to scientifically justify the necessity of non-linear tree-based algorithms integrated with spatial fire data. Our Live Dashboard (Tab 4) actively tracks this comparison.

* **🥇 Ultimate Champion:** LightGBM (Fire-Integrated) — **R²: 85.90% | MAE: 3.21**
* **🔥 Strong Contender:** XGBoost (Fire-Integrated) — **R²: 85.03% | MAE: 3.19**
* **⛅ Baseline 1:** Support Vector Regressor (Weather-Only) — **R²: 22.73%**
* **📉 Baseline 2:** Multiple Linear Regression (Weather-Only) — **R²: -32.55%**

## 📂 Repository Structure
```text
├── dashboard.py                   # Main active dashboard application
├── requirements.txt               # Cleaned dependency list for Streamlit Cloud
├── models/
│   ├── lgbm_pm25_model.pkl        # LightGBM Champion Model (Required)
│   ├── pm25_model_v7.pkl          # XGBoost Contender Model (Required)
│   ├── svr_pm25_model.pkl         # SVR Baseline Model
│   └── mlr_pm25_model.pkl         # MLR Baseline Model
├── data/final/
│   └── pm25_training_dataset_2018_2022.csv  # Historical dataset (For Tab 3 Charts)
└── train_pm25_models.py           # Training pipeline script
🚀 How to Run Locally
1. Clone the repository & Install dependencies

Bash
git clone [https://github.com/pt22-mfu/mfu-pm25-geoai-warning.git](https://github.com/pt22-mfu/mfu-pm25-geoai-warning.git)
cd mfu-pm25-geoai-warning
pip install -r requirements.txt
2. Set API Keys (Strictly Required)
To fetch live data and generate AI advisories, do not hardcode keys. Create a .streamlit/secrets.toml file in the root directory:

Ini, TOML
OPENWEATHER_API_KEY = "your_openweathermap_key"
NASA_KEY = "your_nasa_firms_key"
GISTDA_KEY = "your_gistda_key"
GEMINI_API_KEY = "your_gemini_api_key"
3. Run the Dashboard

Bash
streamlit run dashboard.py

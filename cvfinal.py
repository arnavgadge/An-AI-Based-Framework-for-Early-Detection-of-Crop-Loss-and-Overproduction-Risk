"""
Agricultural Risk Intelligence Console
Onion Price Crash Prediction & Crop Diversification Simulator
Multilingual | Random Forest | Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import time
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from twilio_advisory import send_crop_advisory_sms
from dotenv import load_dotenv
load_dotenv()   


warnings.filterwarnings("ignore")

# ── NEW: imports for disease detection & prevention ──────────────────────────
import io
import pickle
import threading
from datetime import date as _date

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as _load_keras_model
    from tensorflow.keras import layers as _keras_layers, models as _keras_models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator as _ImageDataGenerator
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Agricultural Risk Intelligence Console",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# TRANSLATION DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "mr": {
        "app_title": "🌾 कृषी जोखीम बुद्धिमत्ता कन्सोल",
        "app_subtitle": "कांदा किंमत पतन अंदाज | पीक विविधीकरण सिम्युलेटर",
        "select_language": "भाषा निवडा",
        "select_role": "भूमिका निवडा",
        "admin": "प्रशासक",
        "farmer": "शेतकरी",
        "risk_score": "जोखीम स्कोअर",
        "crash_probability": "पतन संभाव्यता",
        "confidence": "विश्वास",
        "low_risk": "कमी जोखीम",
        "medium_risk": "मध्यम जोखीम",
        "high_risk": "उच्च जोखीम",
        "critical_risk": "गंभीर जोखीम",
        "data_input": "डेटा इनपुट पॅनल",
        "current_prod": "सध्याचे उत्पादन (मेट्रिक टन)",
        "seeds_dist": "वितरित बियाणे (किलो)",
        "rainfall": "वार्षिक पर्जन्यमान (मिमी)",
        "onion_price": "कांदा किंमत (₹/क्विंटल)",
        "garlic_price": "लसूण किंमत (₹/क्विंटल)",
        "paddy_price": "भात किंमत (₹/क्विंटल)",
        "analyze": "जोखीम विश्लेषण करा",
        "analog_years": "ऐतिहासिक सादृश्य वर्षे",
        "diversification": "विविधीकरण सिम्युलेटर",
        "onion_pct": "कांदा %",
        "garlic_pct": "लसूण %",
        "paddy_pct": "भात %",
        "revenue_estimate": "अंदाजे महसूल",
        "risk_reduction": "जोखीम घट",
        "crisis_rewind": "संकट रिवाइंड",
        "select_year": "वर्ष निवडा",
        "predicted": "अंदाजित",
        "actual": "वास्तविक",
        "counterfactual": "काउंटरफॅक्चुअल इंजिन",
        "market_analysis": "लाइव्ह बाजार विश्लेषण",
        "news_feed": "कृषी बातम्या",
        "weather_forecast": "७-दिवसीय हवामान अंदाज",
        "temperature": "तापमान",
        "rain_prob": "पाऊस संभाव्यता",
        "seasonal_context": "हंगामी संदर्भ",
        "market_signals": "बाजार संकेत",
        "recommendations": "शिफारशी",
        "similarity": "साम्य",
        "year": "वर्ष",
        "production": "उत्पादन",
        "what_if": "जर 20% जमीन विविध केली तर?",
        "current_outcome": "सध्याचा परिणाम",
        "alt_outcome": "पर्यायी परिणाम",
        "price_trend": "किंमत ट्रेंड",
        "run_counterfactual": "काउंटरफॅक्चुअल चालवा",
        "farmer_risk_title": "आपल्या शेतासाठी जोखीम",
        "farmer_explanation": "जोखीम माहिती",
        "analyze_button": "विश्लेषण सुरू करा",
        "month": "महिना",
        "price": "किंमत",
        "stabilization": "स्थिरीकरण प्रभाव",
    },
    "hi": {
        "app_title": "🌾 कृषि जोखिम बुद्धिमत्ता कंसोल",
        "app_subtitle": "प्याज मूल्य पतन भविष्यवाणी | फसल विविधीकरण सिम्युलेटर",
        "select_language": "भाषा चुनें",
        "select_role": "भूमिका चुनें",
        "admin": "व्यवस्थापक",
        "farmer": "किसान",
        "risk_score": "जोखिम स्कोर",
        "crash_probability": "पतन संभावना",
        "confidence": "विश्वास",
        "low_risk": "कम जोखिम",
        "medium_risk": "मध्यम जोखिम",
        "high_risk": "उच्च जोखिम",
        "critical_risk": "गंभीर जोखिम",
        "data_input": "डेटा इनपुट पैनल",
        "current_prod": "वर्तमान उत्पादन (मेट्रिक टन)",
        "seeds_dist": "वितरित बीज (किग्रा)",
        "rainfall": "वार्षिक वर्षा (मिमी)",
        "onion_price": "प्याज मूल्य (₹/क्विंटल)",
        "garlic_price": "लहसुन मूल्य (₹/क्विंटल)",
        "paddy_price": "धान मूल्य (₹/क्विंटल)",
        "analyze": "जोखिम विश्लेषण करें",
        "analog_years": "ऐतिहासिक सादृश्य वर्ष",
        "diversification": "विविधीकरण सिम्युलेटर",
        "onion_pct": "प्याज %",
        "garlic_pct": "लहसुन %",
        "paddy_pct": "धान %",
        "revenue_estimate": "अनुमानित राजस्व",
        "risk_reduction": "जोखिम कमी",
        "crisis_rewind": "संकट रिवाइंड",
        "select_year": "वर्ष चुनें",
        "predicted": "भविष्यवाणी",
        "actual": "वास्तविक",
        "counterfactual": "काउंटरफैक्चुअल इंजन",
        "market_analysis": "लाइव बाजार विश्लेषण",
        "news_feed": "कृषि समाचार",
        "weather_forecast": "७-दिन मौसम पूर्वानुमान",
        "temperature": "तापमान",
        "rain_prob": "बारिश संभावना",
        "seasonal_context": "मौसमी संदर्भ",
        "market_signals": "बाजार संकेत",
        "recommendations": "सिफारिशें",
        "similarity": "समानता",
        "year": "वर्ष",
        "production": "उत्पादन",
        "what_if": "यदि 20% भूमि विविध की जाए?",
        "current_outcome": "वर्तमान परिणाम",
        "alt_outcome": "वैकल्पिक परिणाम",
        "price_trend": "मूल्य ट्रेंड",
        "run_counterfactual": "काउंटरफैक्चुअल चलाएं",
        "farmer_risk_title": "आपके खेत के लिए जोखिम",
        "farmer_explanation": "जोखिम जानकारी",
        "analyze_button": "विश्लेषण शुरू करें",
        "month": "महीना",
        "price": "मूल्य",
        "stabilization": "स्थिरीकरण प्रभाव",
    },
    "en": {
        "app_title": "🌾 Agricultural Risk Intelligence Console",
        "app_subtitle": "Onion Price Crash Prediction | Crop Diversification Simulator",
        "select_language": "Select Language",
        "select_role": "Select Role",
        "admin": "Admin",
        "farmer": "Farmer",
        "risk_score": "Risk Score",
        "crash_probability": "Crash Probability",
        "confidence": "Confidence",
        "low_risk": "Low Risk",
        "medium_risk": "Medium Risk",
        "high_risk": "High Risk",
        "critical_risk": "Critical Risk",
        "data_input": "Data Input Panel",
        "current_prod": "Current Production (Metric Tons)",
        "seeds_dist": "Seeds Distributed (kg)",
        "rainfall": "Annual Rainfall (mm)",
        "onion_price": "Onion Price (₹/Quintal)",
        "garlic_price": "Garlic Price (₹/Quintal)",
        "paddy_price": "Paddy Price (₹/Quintal)",
        "analyze": "Analyze Risk",
        "analog_years": "Historical Analog Years",
        "diversification": "Diversification Simulator",
        "onion_pct": "Onion %",
        "garlic_pct": "Garlic %",
        "paddy_pct": "Paddy %",
        "revenue_estimate": "Estimated Revenue",
        "risk_reduction": "Risk Reduction",
        "crisis_rewind": "Crisis Rewind",
        "select_year": "Select Year",
        "predicted": "Predicted",
        "actual": "Actual",
        "counterfactual": "Counterfactual Engine",
        "market_analysis": "Live Market Analysis",
        "news_feed": "Agricultural News Feed",
        "weather_forecast": "7-Day Weather Forecast",
        "temperature": "Temperature",
        "rain_prob": "Rain Probability",
        "seasonal_context": "Seasonal Context",
        "market_signals": "Market Signals",
        "recommendations": "Recommendations",
        "similarity": "Similarity",
        "year": "Year",
        "production": "Production",
        "what_if": "What if 20% land diversified?",
        "current_outcome": "Current Outcome",
        "alt_outcome": "Alternative Outcome",
        "price_trend": "Price Trend",
        "run_counterfactual": "Run Counterfactual",
        "farmer_risk_title": "Risk for Your Farm",
        "farmer_explanation": "Risk Information",
        "analyze_button": "Start Analysis",
        "month": "Month",
        "price": "Price",
        "stabilization": "Stabilization Impact",
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════
# Robust path resolution — works on Windows/Mac/Linux regardless of
# which directory you launch `streamlit run app.py` from.
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _script_dir = os.getcwd()

def _find_base_dir():
    probe = "onion_price_enhanced_v2.csv"
    if os.path.exists(os.path.join(_script_dir, probe)):
        return _script_dir
    if os.path.exists(os.path.join(os.getcwd(), probe)):
        return os.getcwd()
    return _script_dir

BASE_DIR = _find_base_dir()

# ═══════════════════════════════════════════════════════════════════════════
# DISEASE DETECTION — CONSTANTS & LOADERS (ADDITIVE)
# ═══════════════════════════════════════════════════════════════════════════
_DISEASE_CLASSES = [
    "alternaria", "botrytis_blight", "bulb_rot", "downy_mildew",
    "fusarium", "healthy", "iris_yellow_virus", "purple_blotch",
    "rust", "stemphylium_blight", "xanthomonas_blight", "caterpillar"
]

_DISEASE_ADVICE = {
    "healthy":            "✅ Crop appears healthy. Continue regular monitoring.",
    "alternaria":         "🍄 Apply recommended fungicide and improve airflow around plants.",
    "botrytis_blight":    "🍄 Remove infected leaves immediately and spray fungicide.",
    "bulb_rot":           "💧 Avoid over-irrigation and improve field drainage.",
    "downy_mildew":       "🌫️ Spray systemic fungicide before humidity peaks.",
    "fusarium":           "🔄 Use crop rotation and disease-resistant varieties.",
    "iris_yellow_virus":  "🦟 Remove infected plants and control insect vectors.",
    "purple_blotch":      "🟣 Apply protective fungicide immediately.",
    "rust":               "🟠 Use sulfur-based fungicide spray.",
    "stemphylium_blight": "🌿 Early-stage fungicide application recommended.",
    "xanthomonas_blight": "🦠 Apply copper-based bactericide.",
    "caterpillar":        "🐛 Apply neem oil or approved pesticide.",
}

_ONION_TIMELINE_CSV_DEFAULT = """crop,disease,day_start,day_end,risk_level,humidity_factor,temp_factor,rain_factor,preventive_advice
onion,downy_mildew,30,60,high,high,moderate,high,Apply systemic fungicide before humidity peaks.
onion,purple_blotch,45,75,medium,high,warm,moderate,Spray protective fungicide on leaves.
onion,stemphylium_blight,50,80,medium,high,warm,moderate,Ensure good ventilation and apply preventive spray.
onion,alternaria,40,70,medium,moderate,warm,moderate,Apply broad spectrum fungicide immediately.
onion,thrips,20,50,high,low,hot,low,Apply neem oil weekly during dry heat periods.
onion,bulb_rot,80,110,high,high,moderate,high,Avoid waterlogging; improve drainage during bulb formation.
onion,botrytis_blight,35,65,medium,high,cool,moderate,Remove infected leaves; spray fungicide regularly.
onion,fusarium,60,100,high,moderate,warm,low,Use crop rotation and resistant seed varieties.
onion,rust,40,80,medium,moderate,warm,moderate,Apply sulfur-based fungicide at first signs.
onion,xanthomonas_blight,25,55,high,high,hot,high,Apply copper-based bactericide; avoid leaf wetness.
onion,iris_yellow_virus,15,45,high,moderate,warm,low,Remove infected plants; control aphid/insect vectors.
onion,caterpillar,20,60,medium,low,hot,low,Apply neem oil or approved pesticide on canopy.
"""

@st.cache_resource
def _load_cnn_disease_model(model_path: str):
    """Load trained onion disease CNN model. Returns None if unavailable."""
    if not _TF_AVAILABLE:
        return None
    if os.path.exists(model_path):
        return _load_keras_model(model_path)
    return None

@st.cache_data
def _load_onion_disease_timeline(csv_path: str) -> pd.DataFrame:
    """Load onion disease timeline CSV; auto-creates from defaults if missing."""
    if not os.path.exists(csv_path):
        df = pd.read_csv(io.StringIO(_ONION_TIMELINE_CSV_DEFAULT))
        df.to_csv(csv_path, index=False)
        return df
    return pd.read_csv(csv_path)

def _preprocess_disease_image(pil_image) -> "np.ndarray":
    """Resize → normalize → expand dims for model.predict()"""
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@st.cache_resource
def load_model_and_features():
    model = joblib.load(os.path.join(BASE_DIR, "rf_monthly_model.pkl"))
    with open(os.path.join(BASE_DIR, "monthly_features.json")) as f:
        features = json.load(f)
    return model, features

@st.cache_data
def _melt_price(df):
    """Wide format (YEAR x JAN..DEC) → long format (year, month, price_per_quintal)."""
    MONTH_COLS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    MONTH_MAP  = {m: i+1 for i, m in enumerate(MONTH_COLS)}
    avail = [c for c in MONTH_COLS if c in df.columns]
    long = df.melt(id_vars=["YEAR"], value_vars=avail,
                   var_name="MONTH_NAME", value_name="price_per_quintal")
    long["year"]  = long["YEAR"]
    long["month"] = long["MONTH_NAME"].map(MONTH_MAP)
    return long[["year","month","price_per_quintal"]].dropna().sort_values(["year","month"]).reset_index(drop=True)

@st.cache_data
def _melt_arrivals(df):
    MONTH_COLS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    MONTH_MAP  = {m: i+1 for i, m in enumerate(MONTH_COLS)}
    avail = [c for c in MONTH_COLS if c in df.columns]
    long = df.melt(id_vars=["YEAR"], value_vars=avail,
                   var_name="MONTH_NAME", value_name="arrivals_tonnes")
    long["year"]  = long["YEAR"]
    long["month"] = long["MONTH_NAME"].map(MONTH_MAP)
    return long[["year","month","arrivals_tonnes"]].dropna().sort_values(["year","month"]).reset_index(drop=True)

@st.cache_data
def load_all_datasets():
    def _r(name): return pd.read_csv(os.path.join(BASE_DIR, name))

    # onion_monthly — already long format with correct column names
    monthly = _r("onion_monthly_1960_2025.csv")
    # Ensure seeds column is named consistently
    if "seeds_distributed" in monthly.columns and "seeds_distributed_kg" not in monthly.columns:
        monthly["seeds_distributed_kg"] = monthly["seeds_distributed"]
    if "production_annual" in monthly.columns and "production_mt" not in monthly.columns:
        monthly["production_mt"] = monthly["production_monthly"] if "production_monthly" in monthly.columns else monthly["production_annual"]

    # Wide-format price files → melt to long
    onion_p   = _melt_price(_r("onion_price_enhanced_v2.csv"))
    garlic_p  = _melt_price(_r("garlic_price_enhanced_v2.csv"))
    ginger_p  = _melt_price(_r("ginger_price_enhanced_v2.csv"))
    paddy_p   = _melt_price(_r("paddy_price_enhanced_v2.csv"))
    onion_a   = _melt_arrivals(_r("onion_arrivals_enhanced_v2.csv"))

    # Annual production files
    opr_raw = _r("onion_prod_enhanced_v2.csv")
    opr_raw.columns = [c.upper() for c in opr_raw.columns]
    onion_pr = opr_raw[["YEAR","ANN"]].rename(columns={"YEAR":"year","ANN":"production_mt"})

    gpr_raw = _r("garlic_prod_enhanced_v2.csv")
    gpr_raw.columns = [c.upper() for c in gpr_raw.columns]
    garlic_pr = gpr_raw[["YEAR","ANN"]].rename(columns={"YEAR":"year","ANN":"production_mt"})

    # Seeds — annual
    sd_raw = _r("onion_seeds_distributed_v2.csv")
    sd_raw.columns = [c.upper() for c in sd_raw.columns]
    seeds_col = [c for c in sd_raw.columns if "SEEDS" in c and c != "SEEDS_CHANGE_PCT"
                 and "AVG" not in c and "DEV" not in c][0]
    onion_s = sd_raw[["YEAR", seeds_col]].rename(columns={"YEAR":"year", seeds_col:"seeds_kg"})

    return {
        "monthly": monthly, "onion_price": onion_p,
        "onion_prod": onion_pr, "onion_seeds": onion_s,
        "onion_arrivals": onion_a, "garlic_price": garlic_p,
        "garlic_prod": garlic_pr, "ginger_price": ginger_p,
        "paddy_price": paddy_p,
    }

def get_weather_forecast():
    np.random.seed(datetime.now().day)
    days, temps, rains = [], [], []
    for i in range(7):
        d = datetime.now() + timedelta(days=i)
        days.append(d.strftime("%a %d %b"))
        temps.append(round(np.random.uniform(22, 36), 1))
        rains.append(round(np.random.uniform(0, 85), 1))
    return pd.DataFrame({"day": days, "temp_c": temps, "rain_pct": rains})

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
def build_feature_vector(monthly_df, year, month, price, production, rainfall, seeds):
    df = monthly_df.copy().sort_values(['year', 'month']).reset_index(drop=True)
    prices = df['price'].tolist()
    recent = prices[-24:] if len(prices) >= 24 else prices
    r = np.array(recent)

    price_ma3  = np.mean(r[-3:])  if len(r) >= 3  else price
    price_ma12 = np.mean(r[-12:]) if len(r) >= 12 else price
    price_std3 = np.std(r[-3:])   if len(r) >= 3  else 0
    price_std12= np.std(r[-12:])  if len(r) >= 12 else 0
    price_mom  = (price - r[-1]) / r[-1] if len(r) >= 1 and r[-1] > 0 else 0
    price_yoy  = (price - r[-12]) / r[-12] if len(r) >= 12 and r[-12] > 0 else 0
    trend_str  = (price_ma3 - price_ma12) / price_ma12 if price_ma12 > 0 else 0
    price_acc  = price_mom - ((r[-1] - r[-2]) / r[-2] if len(r) >= 2 and r[-2] > 0 else 0)
    price_vol  = np.std(r[-6:]) if len(r) >= 6 else 0
    _sc = "seeds_distributed_kg" if "seeds_distributed_kg" in monthly_df.columns else "seeds_distributed"
    seeds_hist = monthly_df[_sc].values
    seeds_yoy  = (seeds - seeds_hist[-12]) / seeds_hist[-12] if len(seeds_hist) >= 12 and seeds_hist[-12] > 0 else 0
    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)
    harvest    = 1 if month in [3, 4, 5] else 0

    return np.array([[
        price_mom, price_yoy, price_ma3, price_ma12,
        price_std3, price_std12, trend_str,
        seeds_yoy, seeds, month_sin, month_cos,
        harvest, price_acc, price_vol
    ]])

# ═══════════════════════════════════════════════════════════════════════════
# ANALOG YEAR FINDER
# ═══════════════════════════════════════════════════════════════════════════
def find_analog_years(monthly_df, prod, rainfall, seeds, price, n=3):
    # Determine available columns dynamically
    prod_col  = "production_mt" if "production_mt" in monthly_df.columns else "production_annual"
    seeds_col = "seeds_distributed_kg" if "seeds_distributed_kg" in monthly_df.columns else "seeds_distributed"

    agg_dict = {"avg_price": ("price", "mean"), "avg_prod": (prod_col, "mean")}
    if seeds_col in monthly_df.columns:
        agg_dict["avg_seeds"] = (seeds_col, "mean")

    yearly = monthly_df.groupby("year").agg(**agg_dict).reset_index()
    if "avg_seeds" not in yearly.columns:
        yearly["avg_seeds"] = 1.0

    def _norm(series):
        mu, sigma = series.mean(), series.std()
        return mu, sigma if sigma > 0 else 1.0

    def zscore(val, mu, sigma):
        return (val - mu) / sigma

    nm_p  = _norm(yearly["avg_price"])
    nm_pr = _norm(yearly["avg_prod"])
    nm_s  = _norm(yearly["avg_seeds"])

    q_p  = zscore(price, *nm_p)
    q_pr = zscore(prod,  *nm_pr)
    q_s  = zscore(seeds, *nm_s)

    dists = []
    for _, row in yearly.iterrows():
        rp  = zscore(row["avg_price"], *nm_p)
        rpr = zscore(row["avg_prod"],  *nm_pr)
        rs  = zscore(row["avg_seeds"], *nm_s)
        dist = np.sqrt((q_p-rp)**2 + (q_pr-rpr)**2 + (q_s-rs)**2)
        dists.append({"year": int(row["year"]), "dist": dist})

    dists.sort(key=lambda x: x["dist"])
    results = []
    for d in dists[:n]:
        sim = max(0, 100 - d["dist"] * 20)
        results.append({"year": d["year"], "similarity": round(sim, 1)})
    return results

# ═══════════════════════════════════════════════════════════════════════════
# SPEEDOMETER CHART
# ═══════════════════════════════════════════════════════════════════════════
def make_speedometer(score, label, lang="en"):
    score_pct = score * 100
    if score_pct < 25:
        color = "#2ECC71"
    elif score_pct < 50:
        color = "#F39C12"
    elif score_pct < 75:
        color = "#E67E22"
    else:
        color = "#E74C3C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score_pct, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Georgia"}},
        title={"text": f"<b>{label}</b>", "font": {"size": 14, "color": "#8B7355", "family": "Georgia"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8B7355",
                     "tickfont": {"color": "#8B7355"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1a0f",
            "borderwidth": 1,
            "bordercolor": "#4a4a2a",
            "steps": [
                {"range": [0, 25],  "color": "rgba(46,204,113,0.15)"},
                {"range": [25, 50], "color": "rgba(243,156,18,0.15)"},
                {"range": [50, 75], "color": "rgba(230,126,34,0.15)"},
                {"range": [75, 100],"color": "rgba(231,76,60,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": score_pct,
            },
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Georgia"},
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Noto+Sans+Devanagari:wght@300;400;600&display=swap');

    :root {
        --soil: #1C1A0E;
        --clay: #2D2B18;
        --bark: #3D3A20;
        --straw: #8B7D4A;
        --wheat: #C4A55A;
        --harvest: #E8C875;
        --leaf: #4A7C59;
        --sage: #6B9E7A;
        --mist: #B8D4C0;
        --sun: #F2B830;
        --rust: #A0522D;
        --danger: #C0392B;
        --cream: #F5F0E0;
        --parchment: #EDE4C8;
    }

    html, body, [class*="css"] {
        font-family: 'EB Garamond', 'Noto Sans Devanagari', serif;
        background-color: var(--soil);
        color: var(--cream);
    }

    .stApp {
        background: linear-gradient(160deg, #1C1A0E 0%, #1E1D10 40%, #161508 100%);
    }

    /* HEADER */
    .ric-header {
        background: linear-gradient(135deg, #2D2B18 0%, #1C1A0E 50%, #2A2810 100%);
        border: 1px solid var(--bark);
        border-radius: 4px;
        padding: 24px 32px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .ric-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--harvest), var(--wheat), transparent);
    }
    .ric-header h1 {
        font-family: 'Cinzel', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--harvest);
        letter-spacing: 0.08em;
        margin: 0;
        text-shadow: 0 0 30px rgba(232,200,117,0.3);
    }
    .ric-header p {
        font-family: 'EB Garamond', serif;
        color: var(--straw);
        font-size: 0.95rem;
        margin: 6px 0 0;
        letter-spacing: 0.12em;
        font-style: italic;
    }
    .ric-header .corner-mark {
        position: absolute;
        top: 16px; right: 24px;
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        color: var(--bark);
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    /* PANELS */
    .panel {
        background: linear-gradient(145deg, #252310, #1C1A0E);
        border: 1px solid var(--bark);
        border-radius: 4px;
        padding: 18px 20px;
        margin-bottom: 14px;
        position: relative;
    }
    .panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, var(--wheat), transparent);
        border-radius: 4px 0 0 4px;
    }
    .panel-title {
        font-family: 'Cinzel', serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--straw);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--bark);
    }

    /* RISK CARDS */
    .risk-card {
        padding: 14px 16px;
        border-radius: 4px;
        border-left: 4px solid;
        background: rgba(255,255,255,0.03);
        margin-bottom: 10px;
    }
    .risk-low    { border-color: #2ECC71; }
    .risk-medium { border-color: #F39C12; }
    .risk-high   { border-color: #E67E22; }
    .risk-critical { border-color: #E74C3C; }

    /* ANALOG YEAR BADGES */
    .analog-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: rgba(139,125,74,0.08);
        border: 1px solid var(--bark);
        border-radius: 3px;
        margin-bottom: 8px;
    }
    .analog-year {
        font-family: 'Cinzel', serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--harvest);
    }
    .sim-bar-outer {
        height: 8px;
        background: var(--clay);
        border-radius: 4px;
        overflow: hidden;
        flex: 1;
        margin: 0 14px;
    }
    .sim-bar-inner {
        height: 100%;
        background: linear-gradient(90deg, var(--leaf), var(--harvest));
        border-radius: 4px;
    }
    .sim-pct {
        font-family: 'Cinzel', serif;
        font-size: 0.85rem;
        color: var(--wheat);
        min-width: 40px;
        text-align: right;
    }

    /* METRICS */
    .metric-box {
        text-align: center;
        padding: 16px 12px;
        background: rgba(139,125,74,0.06);
        border: 1px solid var(--bark);
        border-radius: 4px;
    }
    .metric-label {
        font-family: 'Cinzel', serif;
        font-size: 0.6rem;
        letter-spacing: 0.2em;
        color: var(--straw);
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Cinzel', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--harvest);
    }
    .metric-sub {
        font-size: 0.75rem;
        color: var(--straw);
        margin-top: 3px;
    }

    /* WEATHER SIDEBAR */
    .weather-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 10px;
        border-bottom: 1px solid var(--bark);
        font-size: 0.82rem;
    }
    .weather-row:last-child { border-bottom: none; }
    .weather-day { color: var(--straw); font-family: 'Cinzel', serif; font-size: 0.7rem; letter-spacing: 0.1em; }
    .weather-temp { color: var(--harvest); font-weight: 600; }
    .weather-rain { color: var(--sage); }

    /* SLIDERS */
    .stSlider > div > div { background: var(--bark) !important; }
    .stSlider > div > div > div { background: var(--wheat) !important; }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, var(--bark), var(--clay));
        color: var(--harvest) !important;
        border: 1px solid var(--wheat) !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        padding: 10px 24px !important;
        border-radius: 3px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--straw), var(--bark)) !important;
        box-shadow: 0 0 15px rgba(232,200,117,0.3) !important;
    }

    /* SELECTBOX */
    .stSelectbox > div > div {
        background: var(--clay) !important;
        border: 1px solid var(--bark) !important;
        color: var(--cream) !important;
        font-family: 'EB Garamond', serif !important;
    }

    /* INPUTS */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: var(--clay) !important;
        border: 1px solid var(--bark) !important;
        color: var(--cream) !important;
        font-family: 'EB Garamond', serif !important;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid var(--bark);
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--straw) !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.15em !important;
        padding: 10px 18px !important;
        border-radius: 0 !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--harvest) !important;
        border-bottom: 2px solid var(--harvest) !important;
        background: rgba(232,200,117,0.05) !important;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--soil); }
    ::-webkit-scrollbar-thumb { background: var(--bark); border-radius: 2px; }

    /* PLOTLY TRANSPARENT BACKGROUND */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* DIVIDER */
    hr { border-color: var(--bark) !important; margin: 16px 0; }

    /* STATUS LINE */
    .status-line {
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        color: var(--bark);
        text-align: center;
        padding: 6px;
        border-top: 1px solid var(--bark);
        margin-top: 16px;
    }

    /* NEWS CARD */
    .news-card {
        padding: 14px 16px;
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--bark);
        border-radius: 3px;
        margin-bottom: 10px;
        position: relative;
    }
    .news-card::before {
        content: '◈';
        position: absolute;
        top: 14px; left: -8px;
        color: var(--wheat);
        font-size: 0.7rem;
    }
    .news-title { font-family: 'EB Garamond', serif; font-size: 1rem; color: var(--harvest); margin-bottom: 4px; }
    .news-body { font-size: 0.85rem; color: var(--straw); line-height: 1.6; }
    .news-tag { display: inline-block; font-family: 'Cinzel', serif; font-size: 0.55rem; letter-spacing: 0.2em; color: var(--sage); border: 1px solid var(--leaf); padding: 2px 6px; border-radius: 2px; margin-top: 6px; }

    /* FARMER VIEW */
    .farmer-panel {
        background: linear-gradient(145deg, #1E2810, #141A0A);
        border: 2px solid var(--leaf);
        border-radius: 8px;
        padding: 28px 24px;
        text-align: center;
    }
    .farmer-score {
        font-family: 'Cinzel', serif;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 12px 0;
    }
    .farmer-label {
        font-family: 'Cinzel', serif;
        font-size: 0.8rem;
        letter-spacing: 0.2em;
        color: var(--straw);
        text-transform: uppercase;
    }
    .farmer-note {
        font-size: 1rem;
        color: var(--cream);
        margin-top: 16px;
        line-height: 1.7;
        padding: 16px;
        background: rgba(255,255,255,0.03);
        border-radius: 4px;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# LIVE MANDI MARKET — DATA & LOGIC
# ═══════════════════════════════════════════════════════════════════════════
import time as _time

NASHIK_MANDIS = [
    ("Nashik APMC",      0,   1.00),
    ("Lasalgaon APMC",   35,  1.05),
    ("Pimpalgaon APMC",  25,  0.98),
    ("Malegaon APMC",    60,  0.95),
    ("Manmad APMC",      90,  0.92),
    ("Yeola APMC",       45,  0.97),
    ("Niphad APMC",      30,  0.99),
    ("Dindori APMC",     50,  0.96),
    ("Satana APMC",      70,  0.94),
    ("Chandwad APMC",    40,  0.98),
    ("Pune APMC",        210, 1.02),
    ("Mumbai APMC",      180, 1.08),
]

def _generate_mandi_prices(base_price=200.0):
    """base_price in ₹/quintal (matches dataset units)"""
    current_time = datetime.now()
    minutes = (current_time.minute // 2) * 2
    time_seed = current_time.replace(minute=minutes, second=0, microsecond=0)
    seed_value = int(time_seed.timestamp()) % 100000
    np.random.seed(seed_value)
    rows = []
    for name, dist, factor in NASHIK_MANDIS:
        mbase = base_price * factor
        daily_var = np.random.uniform(-0.05, 0.05)
        hour = current_time.hour
        tf = 1.03 if 6 <= hour < 10 else 0.97 if 16 <= hour < 20 else 1.0
        sd = np.random.uniform(-0.03, 0.03)
        price = max(50, mbase * (1 + daily_var) * tf * (1 + sd))  # floor ₹50/quintal
        base_arr = 120 if dist == 0 else 80 if dist < 50 else 50
        arrival = base_arr * np.random.uniform(0.8, 1.2)
        ga = np.random.uniform(40, 60)
        gb = np.random.uniform(30, 40)
        gc = 100 - ga - gb
        rows.append({
            "Mandi": name,
            "Distance (km)": dist,
            "Price (₹/Q)": round(price, 1),    # ₹/quintal
            "Arrival (T)": round(arrival, 1),
            "Grade A (%)": round(ga, 1),
            "Grade B (%)": round(gb, 1),
            "Grade C (%)": round(gc, 1),
            "Updated": time_seed.strftime("%I:%M %p"),
        })
    return pd.DataFrame(rows)

def _rank_mandis(df, mode="balanced"):
    df = df.copy()
    if mode == "price":
        df = df.sort_values("Price (₹/Q)", ascending=False)
        df["Rec"] = ["🏆 Highest!" if i==0 else "⭐ Good" if i<3 else "📍 Compare" for i in range(len(df))]
    elif mode == "distance":
        df = df.sort_values("Distance (km)", ascending=True)
        df["Rec"] = ["🚛 Nearest!" if i==0 else "✅ Close" if i<3 else "🛣️ Far" for i in range(len(df))]
    else:
        mp = df["Price (₹/Q)"].max(); md = df["Distance (km)"].max()
        df["Score"] = (df["Price (₹/Q)"]/mp)*0.6 + (1 - df["Distance (km)"]/md)*0.4
        df = df.sort_values("Score", ascending=False)
        df["Rec"] = ["💎 Best Choice!" if i==0 else "✨ Excellent" if i<3 else "📊 Consider" for i in range(len(df))]
    df.insert(0, "Rank", range(1, len(df)+1))
    return df

def _net_profit(price_per_quintal, distance_km, qty_q=10):
    """price_per_quintal in ₹/quintal"""
    transport = distance_km * 15 + 500      # ₹15/km + ₹500 loading
    gross = price_per_quintal * qty_q
    mandi_fee = gross * 0.02                # 2% commission
    net = gross - transport - mandi_fee
    return {"gross": gross, "transport": transport, "mandi_fee": mandi_fee,
            "net": net, "per_quintal": net / qty_q}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # ── Load resources ────────────────────────────────────────────────────
    model, features = load_model_and_features()
    datasets = load_all_datasets()
    monthly_df = datasets["monthly"]
    weather_df = get_weather_forecast()

    # ── TOP CONTROLS ──────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 1, 1])
    with ctrl_c1:
        lang = st.selectbox(
            "🌐",
            options=["mr", "hi", "en"],
            format_func=lambda x: {"mr": "मराठी", "hi": "हिंदी", "en": "English"}[x],
            label_visibility="collapsed",
            key="lang_select"
        )
    with ctrl_c2:
        T = TRANSLATIONS[lang]
        role = st.selectbox(
            T["select_role"],
            options=["Admin", "Farmer"],
            format_func=lambda x: T["admin"] if x == "Admin" else T["farmer"],
            label_visibility="collapsed",
            key="role_select"
        )
    with ctrl_c3:
        current_month = datetime.now().month
        current_year = datetime.now().year

    T = TRANSLATIONS[lang]

    # ── HEADER ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ric-header">
        <div class="corner-mark">AGRI · INTEL · v2.0</div>
        <h1>{T['app_title']}</h1>
        <p>{T['app_subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SIDEBAR — WEATHER
    # ════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(f"""
        <div class="panel-title">☁ {T['weather_forecast']}</div>
        """, unsafe_allow_html=True)

        for _, row in weather_df.iterrows():
            rain_color = "#4A7C59" if row['rain_pct'] < 40 else "#E67E22" if row['rain_pct'] < 70 else "#E74C3C"
            icon = "🌤" if row['rain_pct'] < 30 else "🌦" if row['rain_pct'] < 60 else "🌧"
            st.markdown(f"""
            <div class="weather-row">
                <span class="weather-day">{icon} {row['day']}</span>
                <span class="weather-temp">{row['temp_c']}°C</span>
                <span class="weather-rain" style="color:{rain_color};">{row['rain_pct']}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="panel-title">◈ {T['seasonal_context']}</div>
        """, unsafe_allow_html=True)

        harvest_months = {1: "Rabi Sowing", 2: "Rabi Growth", 3: "Rabi Harvest", 4: "Peak Harvest",
                         5: "Post Harvest", 6: "Kharif Sow", 7: "Kharif Sow", 8: "Kharif Growth",
                         9: "Kharif Growth", 10: "Kharif Harvest", 11: "Late Harvest", 12: "Storage"}
        season = harvest_months.get(current_month, "—")

        month_names_mr = ["", "जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून",
                          "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"]
        month_names_hi = ["", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
                          "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
        month_names_en = ["", "January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"]
        mnames = month_names_mr if lang == "mr" else month_names_hi if lang == "hi" else month_names_en

        st.markdown(f"""
        <div style="padding:10px;background:rgba(74,124,89,0.1);border:1px solid var(--bark);border-radius:4px;font-size:0.82rem;">
            <div style="color:var(--sage);font-family:'Cinzel',serif;font-size:0.62rem;letter-spacing:0.2em;margin-bottom:6px;">CURRENT SEASON</div>
            <div style="color:var(--harvest);font-family:'Cinzel',serif;">{season}</div>
            <div style="color:var(--straw);margin-top:6px;">{mnames[current_month]} {current_year}</div>
        </div>
        """, unsafe_allow_html=True)

        # Market Signals
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""<div class="panel-title">◉ {T['market_signals']}</div>""", unsafe_allow_html=True)

        recent_op = datasets["onion_price"].sort_values(['year','month']).tail(6)
        avg_recent = recent_op['price_per_quintal'].mean()
        ma12 = datasets["onion_price"]['price_per_quintal'].tail(12).mean()
        dev = ((avg_recent - ma12) / ma12) * 100

        sig_color = "#2ECC71" if dev > 0 else "#E74C3C"
        sig_arrow = "▲" if dev > 0 else "▼"

        st.markdown(f"""
        <div style="padding:10px;background:rgba(255,255,255,0.02);border:1px solid var(--bark);border-radius:4px;font-size:0.82rem;">
            <div style="color:var(--straw);margin-bottom:4px;">Onion 6M Avg</div>
            <div style="color:var(--harvest);font-family:'Cinzel',serif;font-size:1.1rem;">₹{avg_recent:.0f}</div>
            <div style="color:{sig_color};font-size:0.78rem;margin-top:4px;">{sig_arrow} {abs(dev):.1f}% vs MA12</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""<div class="status-line">AGRI INTEL · {datetime.now().strftime('%d %b %Y')}</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # FARMER VIEW
    # ════════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════════
# ENHANCED FARMER INTERFACE - INSERT THIS INTO test1.py at line 971
# Replaces lines 971-1200 (entire farmer view section)
# ════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════
    # FARMER VIEW — ENHANCED WITH TIMELINE & PRODUCTION STATUS
    # ════════════════════════════════════════════════════════════════════
    if role == "Farmer":
        # ── Load production status data ───────────────────────────────
        @st.cache_data
        def load_production_status():
            return pd.read_csv(os.path.join(BASE_DIR, "area_seed_ratio_1960_2026_updated_mt.csv"))
        
        prod_status_df = load_production_status()
        
        # ── Load timeline & registration data ─────────────────────────
        @st.cache_data
        def load_timeline_data():
            return pd.read_csv(os.path.join(BASE_DIR, "crop_timeline_advisories.csv"))
        
        timeline_df = load_timeline_data()
        
        # Farmer registrations (persistent)
        farmer_reg_file = os.path.join(BASE_DIR, "farmer_registrations.csv")
        if os.path.exists(farmer_reg_file):
            farmer_reg = pd.read_csv(farmer_reg_file)
        else:
            farmer_reg = pd.DataFrame(columns=[
                'farmer_id', 'name', 'mobile_number', 'crop_type', 'sowing_date',
                'area', 'location', 'last_message_sent_date'
            ])
      
        # ── Header ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="ric-header" style="margin-bottom:16px;">
            <h1 style="font-size:1.4rem;">🌾 {'किसान सहायता प्रणाली' if lang=='hi' else 'शेतकरी सहाय्य प्रणाली' if lang=='mr' else 'Farmer Support System'}</h1>
            <p>{'फसल समयरेखा | उत्पादन स्थिति | लाइव मंडी भाव' if lang=='hi' else 'पीक टाइमलाइन | उत्पादन स्थिती | लाइव्ह मंडी भाव' if lang=='mr' else 'Crop Timeline | Production Status | Live Mandi Prices'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════
        # TABS — 4 tabs including new "My Farms" tab
        # ═══════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            f"➕ {'नई फसल जोड़ें' if lang=='hi' else 'नवीन पीक जोडा' if lang=='mr' else 'Add New Farm'}",
            f"🌾 {'मेरे खेत' if lang=='hi' else 'माझे शेत' if lang=='mr' else 'My Farms'} ({len(farmer_reg)})",
            f"🎯 {'उत्पादन स्थिति' if lang=='hi' else 'उत्पादन स्थिती' if lang=='mr' else 'Production Status'}",
            f"⊛ {'लाइव मंडी भाव' if lang=='hi' else 'लाइव्ह मंडी भाव' if lang=='mr' else 'Live Mandi Prices'}",
            f"🛡 {'रोग रोकथाम' if lang=='hi' else 'रोग प्रतिबंध' if lang=='mr' else 'Disease Prevention'}"
        ])

        # ═══════════════════════════════════════════════════════════════
        # TAB 1: ADD NEW FARM — always-visible registration form
        # ═══════════════════════════════════════════════════════════════
        with tab1:
            st.markdown(f"""<div class="panel-title">➕ {'नई फसल पंजीकृत करें' if lang=='hi' else 'नवीन पीक नोंदणी' if lang=='mr' else 'Register a New Farm'}</div>""",
                        unsafe_allow_html=True)

            # Info banner
            farm_count = len(farmer_reg)
            if farm_count > 0:
                st.markdown(f"""
                <div style="padding:10px 14px;background:rgba(74,124,89,0.15);border:1px solid var(--leaf);border-radius:6px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">
                    <span style="font-size:1.3rem;">🌾</span>
                    <span style="color:var(--cream);font-size:0.88rem;">
                        {'आपके' if lang=='hi' else 'तुमचे'} <b style="color:var(--harvest);">{farm_count}</b> {'खेत पंजीकृत हैं। नया खेत जोड़ने के लिए फ़ॉर्म भरें।' if lang=='hi' else 'शेत नोंदणीकृत आहेत. नवीन शेत जोडण्यासाठी फॉर्म भरा.' if lang=='mr' else f'farm{"s" if farm_count != 1 else ""} registered. Fill the form below to add another.'}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with st.form("add_farm_form", clear_on_submit=True):
                fc1, fc2 = st.columns(2)
                with fc1:
                    f_name = st.text_input(
                        'नाम' if lang=='hi' else 'नाव' if lang=='mr' else 'Farmer / Farm Name',
                        placeholder="राजेश कुमार" if lang in ['hi','mr'] else "Enter farmer name"
                    )
                    f_crop = st.selectbox(
                        'फसल प्रकार' if lang=='hi' else 'पीक प्रकार' if lang=='mr' else 'Crop Type',
                        ['onion', 'garlic', 'paddy'],
                        format_func=lambda x: {'onion': 'प्याज / कांदा / Onion',
                                               'garlic': 'लहसुन / लसूण / Garlic',
                                               'paddy': 'धान / भात / Paddy'}[x]
                    )
                    f_area = st.number_input(
                        'क्षेत्रफल (हेक्टेयर)' if lang=='hi' else 'क्षेत्र (हेक्टर)' if lang=='mr' else 'Area (hectares)',
                        min_value=0.1, max_value=100.0, value=2.0, step=0.5
                    )
                with fc2:
                    f_mobile = st.text_input(
                        'मोबाइल नंबर' if lang=='hi' else 'मोबाईल नंबर' if lang=='mr' else 'Mobile Number',
                        placeholder="+91 9876543210"
                    )
                    f_sowing = st.date_input(
                        'बुवाई की तारीख' if lang=='hi' else 'पेरणी तारीख' if lang=='mr' else 'Sowing Date',
                        value=datetime.now() - timedelta(days=30),
                        max_value=datetime.now()
                    )
                    f_location = st.text_input(
                        'स्थान' if lang=='hi' else 'स्थान' if lang=='mr' else 'Location / Village',
                        placeholder="नाशिक / Nashik"
                    )

                submit = st.form_submit_button(
                    '🌱 फसल पंजीकृत करें' if lang=='hi' else '🌱 पीक नोंदणी करा' if lang=='mr' else '🌱 Register Farm',
                    use_container_width=True
                )

            if submit:
                if f_name.strip() and f_mobile.strip():
                    new_farmer = {
                        'farmer_id': f'F{len(farmer_reg)+1:04d}',
                        'name': f_name.strip(),
                        'mobile_number': f_mobile.strip(),
                        'crop_type': f_crop,
                        'sowing_date': f_sowing.strftime('%Y-%m-%d'),
                        'area': f_area,
                        'location': f_location.strip(),
                        'last_message_sent_date': None
                    }
                    farmer_reg = pd.concat([farmer_reg, pd.DataFrame([new_farmer])], ignore_index=True)
                    farmer_reg.to_csv(farmer_reg_file, index=False)
                    # ── TWILIO SMS ────────────────────────────────────────────────────
                    result = send_crop_advisory_sms(
                        farmer_name=new_farmer["name"],
                        mobile_number=new_farmer["mobile_number"],
                        crop=new_farmer["crop_type"],
                        sowing_date=new_farmer["sowing_date"],
                        crop_timeline_advisories=timeline_df,
                        lang=lang
                    )
                    if result["success"]:
                        st.success(f"📱 Advisory SMS sent to {new_farmer['name']}!")
                    else:
                      st.warning(f"⚠️ SMS failed: {result['error']}")
                      st.session_state.farmer_registered = True
                      st.success(f"✅ {'खेत पंजीकृत! अब आप नया खेत जोड़ सकते हैं।' if lang=='hi' else 'शेत नोंदणी यशस्वी! आता नवीन शेत जोडा.' if lang=='mr' else f'Farm registered for {f_name}! The form is ready for your next farm.'}")
                      time.sleep(3) 
                      st.rerun()
                else:
                    st.warning('नाम और मोबाइल नंबर जरूरी है।' if lang=='hi' else 'नाव आणि मोबाईल नंबर आवश्यक आहे.' if lang=='mr' else 'Please enter a name and mobile number.')

        # ═══════════════════════════════════════════════════════════════
        # TAB 2: MY FARMS — all registered farms with independent timelines
        # ═══════════════════════════════════════════════════════════════
        with tab2:
            st.markdown(f"""<div class="panel-title">🌾 {'मेरे सभी खेत' if lang=='hi' else 'माझी सर्व शेते' if lang=='mr' else 'All My Farms'}</div>""",
                        unsafe_allow_html=True)

            if len(farmer_reg) == 0:
                st.markdown(f"""
                <div style="text-align:center;padding:48px 24px;border:1px dashed var(--bark);border-radius:8px;color:var(--straw);">
                    <div style="font-size:2.5rem;margin-bottom:12px;">🌱</div>
                    <div style="font-family:'Cinzel',serif;font-size:0.9rem;color:var(--harvest);margin-bottom:8px;">
                        {'कोई खेत नहीं' if lang=='hi' else 'कोणतेही शेत नाही' if lang=='mr' else 'No Farms Registered Yet'}
                    </div>
                    <div style="font-size:0.82rem;">
                        {'पहला खेत जोड़ने के लिए "नई फसल जोड़ें" टैब पर जाएं।' if lang=='hi' else '"नवीन पीक जोडा" टॅबवर जा.' if lang=='mr' else 'Go to the "Add New Farm" tab to register your first farm.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Summary strip
                crop_counts = farmer_reg['crop_type'].value_counts()
                summary_cols = st.columns(min(len(crop_counts) + 1, 4))
                with summary_cols[0]:
                    st.markdown(f"""<div class="metric-box">
                        <div class="metric-label">{'कुल खेत' if lang=='hi' else 'एकूण शेते' if lang=='mr' else 'Total Farms'}</div>
                        <div class="metric-value" style="color:var(--harvest);">{len(farmer_reg)}</div>
                    </div>""", unsafe_allow_html=True)
                for i, (crop, cnt) in enumerate(crop_counts.items()):
                    if i + 1 < len(summary_cols):
                        with summary_cols[i + 1]:
                            crop_label = {'onion': '🧅 Onion', 'garlic': '🧄 Garlic', 'paddy': '🌾 Paddy'}.get(crop, crop.title())
                            st.markdown(f"""<div class="metric-box">
                                <div class="metric-label">{crop_label}</div>
                                <div class="metric-value" style="font-size:1.4rem;">{cnt}</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

                # ── Individual Farm Cards ─────────────────────────────
                for idx, farm_row in farmer_reg.iterrows():
                    try:
                        farm_sowing = datetime.strptime(str(farm_row['sowing_date']), '%Y-%m-%d')
                    except Exception:
                        farm_sowing = datetime.now() - timedelta(days=30)
                    farm_days = (datetime.now() - farm_sowing).days

                    # Stage lookup
                    farm_timeline = timeline_df[timeline_df['crop'] == farm_row['crop_type']]
                    farm_stage_row = farm_timeline[
                        (farm_timeline['day_start'] <= farm_days) &
                        (farm_timeline['day_end'] >= farm_days)
                    ]
                    if len(farm_stage_row) > 0:
                        fs = farm_stage_row.iloc[0]
                        farm_stage = fs['stage']
                        farm_advisory = fs.get(f'advisory_{lang}', fs.get('advisory_en', ''))
                        farm_progress = min(100, ((farm_days - fs['day_start']) /
                                            max(1, fs['day_end'] - fs['day_start'] + 1)) * 100)
                        farm_next_rows = farm_timeline[farm_timeline['day_start'] > farm_days]
                        farm_next = farm_next_rows.iloc[0]['stage'] if len(farm_next_rows) > 0 else ('Complete' if lang=='en' else 'पूर्ण')
                    else:
                        farm_stage = 'Completed' if lang=='en' else ('पूर्ण' if lang=='hi' else 'पूर्ण')
                        farm_advisory = 'Crop cycle complete.' if lang=='en' else ('फसल चक्र पूर्ण।' if lang=='hi' else 'पीक चक्र पूर्ण.')
                        farm_progress = 100
                        farm_next = '—'

                    crop_icon = {'onion': '🧅', 'garlic': '🧄', 'paddy': '🌾'}.get(farm_row['crop_type'], '🌱')
                    prog_color = '#2ECC71' if farm_progress >= 75 else '#E8C875' if farm_progress >= 40 else '#4A7C59'

                    with st.expander(
                        f"{crop_icon}  {farm_row['name']}  ·  {farm_row['crop_type'].upper()}  ·  "
                        f"{'Day' if lang=='en' else 'दिन'} {farm_days}  ·  {farm_stage}",
                        expanded=(idx == len(farmer_reg) - 1)   # latest farm open by default
                    ):
                        # Farm header row
                        h1, h2 = st.columns([3, 1])
                        with h1:
                            st.markdown(f"""
                            <div style="padding:14px;background:linear-gradient(135deg,#1E2810,#141A0A);border:2px solid var(--leaf);border-radius:8px;margin-bottom:12px;">
                                <div style="font-family:'Cinzel',serif;font-size:1.05rem;color:var(--harvest);">
                                    {crop_icon} {farm_row['name']}
                                    <span style="font-size:0.65rem;color:var(--bark);margin-left:10px;letter-spacing:0.15em;">
                                        {farm_row.get('farmer_id','—')}
                                    </span>
                                </div>
                                <div style="font-size:0.8rem;color:var(--straw);margin-top:6px;">
                                    📱 {farm_row['mobile_number']} &nbsp;·&nbsp;
                                    📍 {farm_row.get('location') or '—'} &nbsp;·&nbsp;
                                    📐 {farm_row.get('area','—')} {'ha' if lang=='en' else 'हे.'}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with h2:
                            # Delete button for this farm
                            if st.button(
                                '🗑️ ' + ('हटाएं' if lang=='hi' else 'हटवा' if lang=='mr' else 'Remove'),
                                key=f"del_{idx}_{farm_row.get('farmer_id', idx)}",
                                use_container_width=True
                            ):
                                farmer_reg = farmer_reg.drop(index=idx).reset_index(drop=True)
                                farmer_reg.to_csv(farmer_reg_file, index=False)
                                if len(farmer_reg) == 0:
                                    st.session_state.farmer_registered = False
                                st.rerun()

                        # Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.markdown(f"""<div class="metric-box">
                                <div class="metric-label">{'बोया' if lang=='hi' else 'पेरले' if lang=='mr' else 'Sown'}</div>
                                <div class="metric-value" style="font-size:1rem;">{farm_sowing.strftime('%d %b %y')}</div>
                            </div>""", unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""<div class="metric-box">
                                <div class="metric-label">{'दिन' if lang in ['hi','mr'] else 'Days'}</div>
                                <div class="metric-value" style="color:var(--harvest);">{farm_days}</div>
                            </div>""", unsafe_allow_html=True)
                        with m3:
                            st.markdown(f"""<div class="metric-box">
                                <div class="metric-label">{'चरण' if lang=='hi' else 'टप्पा' if lang=='mr' else 'Stage'}</div>
                                <div class="metric-value" style="font-size:0.85rem;">{farm_stage}</div>
                            </div>""", unsafe_allow_html=True)
                        with m4:
                            st.markdown(f"""<div class="metric-box">
                                <div class="metric-label">{'अगला' if lang=='hi' else 'पुढे' if lang=='mr' else 'Next'}</div>
                                <div class="metric-value" style="font-size:0.85rem;">{farm_next}</div>
                            </div>""", unsafe_allow_html=True)

                        # Stage progress bar
                        st.markdown(f"""
                        <div style="margin:10px 0 14px;">
                            <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--straw);margin-bottom:4px;">
                                <span style="font-family:'Cinzel',serif;letter-spacing:0.12em;">
                                    {'चरण प्रगति' if lang=='hi' else 'प्रगती' if lang=='mr' else 'STAGE PROGRESS'}
                                </span>
                                <span>{farm_progress:.0f}%</span>
                            </div>
                            <div style="height:8px;background:var(--clay);border-radius:4px;overflow:hidden;">
                                <div style="height:100%;width:{farm_progress:.1f}%;background:linear-gradient(90deg,{prog_color},var(--harvest));border-radius:4px;transition:width 0.5s;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Advisory
                        st.markdown(f"""
                        <div style="padding:14px;background:rgba(74,124,89,0.1);border-left:3px solid var(--leaf);border-radius:4px;">
                            <div style="font-family:'Cinzel',serif;font-size:0.6rem;color:var(--sage);letter-spacing:0.2em;margin-bottom:6px;">
                                {'दिन ' if lang in ['hi','mr'] else 'DAY '}{farm_days} · {farm_stage.upper()}
                            </div>
                            <div style="font-size:0.88rem;color:var(--cream);line-height:1.65;">
                                {farm_advisory}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════
        # TAB 3: PRODUCTION STATUS
        # ═══════════════════════════════════════════════════════════════
        with tab3:
            st.markdown(f"""<div class="panel-title">🎯 {'उत्पादन स्थिति मूल्यांकन' if lang=='hi' else 'उत्पादन स्थिती मूल्यांकन' if lang=='mr' else 'Production Status Assessment'}</div>""",
                        unsafe_allow_html=True)
            
            # Get current year from system
            current_year = datetime.now().year
            
            # Fetch production status from CSV
            year_data = prod_status_df[prod_status_df['year'] == current_year]
            
            if len(year_data) > 0:
                status = year_data.iloc[0]['production_status']
                seeds_mt = year_data.iloc[0]['seeds_distributed_mt']
                area_ha  = year_data.iloc[0]['area_hectares']
            else:
                # Fallback to last available year
                status = prod_status_df.iloc[-1]['production_status']
                seeds_mt = prod_status_df.iloc[-1]['seeds_distributed_mt']
                area_ha  = prod_status_df.iloc[-1]['area_hectares']
                current_year = int(prod_status_df.iloc[-1]['year'])
            
            # Status color and icon mapping — keys match production_status column values
            status_config = {
                'OVERPRODUCTION': {
                    'color': '#E74C3C',
                    'icon': '🔴',
                    'label_en': 'OVERPRODUCTION',
                    'label_hi': 'अति उत्पादन',
                    'label_mr': 'अतिरिक्त उत्पादन'
                },
                'APT': {
                    'color': '#2ECC71',
                    'icon': '🟢',
                    'label_en': 'BALANCED PRODUCTION',
                    'label_hi': 'संतुलित उत्पादन',
                    'label_mr': 'संतुलित उत्पादन'
                },
                'UNDERPRODUCTION': {
                    'color': '#F39C12',
                    'icon': '🟡',
                    'label_en': 'UNDERPRODUCTION',
                    'label_hi': 'कम उत्पादन',
                    'label_mr': 'कमी उत्पादन'
                }
            }
            
            config = status_config.get(status, status_config['APT'])
            status_label = config[f'label_{lang}']
            
            # Main status card
            ps1, ps2 = st.columns([2, 1])
            with ps1:
                st.markdown(f"""
                <div style="padding:32px;background:linear-gradient(145deg, #1E2810, #141A0A);border:3px solid {config['color']};border-radius:12px;text-align:center;">
                    <div style="font-size:3.5rem;margin-bottom:8px;">{config['icon']}</div>
                    <div style="font-family:'Cinzel',serif;font-size:0.65rem;color:var(--straw);letter-spacing:0.3em;margin-bottom:8px;">
                        {'पूर्वानुमानित स्थिति' if lang=='hi' else 'अंदाजित स्थिती' if lang=='mr' else 'PREDICTED STATUS'}
                    </div>
                    <div style="font-family:'Cinzel',serif;font-size:2.2rem;font-weight:700;color:{config['color']};line-height:1.2;">
                        {status_label}
                    </div>
                    <div style="font-size:0.85rem;color:var(--straw);margin-top:12px;">
                        {'सीजन' if lang=='hi' else 'हंगाम' if lang=='mr' else 'Season'}: {current_year} · {'कटाई विंडो' if lang=='hi' else 'कापणी विंडो' if lang=='mr' else 'Harvest Window'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with ps2:
                st.markdown(f"""
                <div class="metric-box" style="margin-bottom:12px;">
                    <div class="metric-label">{'वर्ष' if lang in ['hi','mr'] else 'Year'}</div>
                    <div class="metric-value" style="font-size:1.8rem;">{current_year}</div>
                </div>
                <div class="metric-box" style="margin-bottom:12px;">
                    <div class="metric-label">{'बीज वितरित' if lang=='hi' else 'बियाणे वितरीत' if lang=='mr' else 'Seeds Distributed'}</div>
                    <div class="metric-value" style="font-size:1.2rem;">{seeds_mt:.0f}</div>
                    <div class="metric-sub">{'मेट्रिक टन' if lang in ['hi','mr'] else 'metric tonnes'}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">{'क्षेत्र' if lang in ['hi','mr'] else 'Area'}</div>
                    <div class="metric-value" style="font-size:1.2rem;">{area_ha:,.0f}</div>
                    <div class="metric-sub">{'हेक्टेयर' if lang=='hi' else 'हेक्टर' if lang=='mr' else 'hectares'}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Advisory based on status
            st.markdown(f"""<div class="panel-title">💡 {'सिफारिशें' if lang=='hi' else 'शिफारशी' if lang=='mr' else 'Recommendations'}</div>""",
                        unsafe_allow_html=True)
            
            if status == 'OVERPRODUCTION':
                advisory_en = """⚠️ **Warning: Overproduction expected this season.**

Onion supply is likely to exceed market demand, which may cause a price drop.

**Suggested actions:**
• Reduce onion acreage by 20–30%
• Diversify into garlic, ginger, or pulses
• Plan staggered harvesting to manage supply
• Explore cold storage or processing options
• Consider contract farming with buyers"""

                advisory_hi = """⚠️ **चेतावनी: इस सीजन में अधिक उत्पादन की उम्मीद है।**

प्याज की आपूर्ति बाजार की मांग से अधिक हो सकती है, जिससे कीमतों में गिरावट आ सकती है।

**सुझाए गए कदम:**
• प्याज का रकबा 20–30% कम करें
• लहसुन, अदरक या दालों में विविधता लाएं
• आपूर्ति प्रबंधन के लिए चरणबद्ध कटाई की योजना बनाएं
• कोल्ड स्टोरेज या प्रसंस्करण विकल्प तलाशें
• खरीदारों के साथ अनुबंध खेती पर विचार करें"""

                advisory_mr = """⚠️ **चेतावणी: या हंगामात अतिरिक्त उत्पादन अपेक्षित आहे।**

कांद्याचा पुरवठा बाजारातील मागणीपेक्षा जास्त होऊ शकतो, ज्यामुळे किंमती घसरू शकतात.

**सुचवलेल्या पावले:**
• कांदा क्षेत्र 20–30% कमी करा
• लसूण, आलं किंवा डाळींमध्ये विविधता आणा
• पुरवठा व्यवस्थापनासाठी टप्प्याटप्प्याने कापणी करा
• कोल्ड स्टोरेज किंवा प्रक्रिया पर्याय शोधा
• खरेदीदारांसोबत करार शेती विचारात घ्या"""
                
                st.error(advisory_hi if lang=='hi' else advisory_mr if lang=='mr' else advisory_en)
            
            elif status == 'APT':
                advisory_en = """✅ **Production is expected to be balanced this season.**

Supply is likely to match demand, indicating stable market conditions.

**Suggested actions:**
• Continue planned cultivation as normal
• Monitor rainfall and market trends regularly
• Avoid sudden acreage expansion
• Maintain good agricultural practices
• Keep track of government advisories"""

                advisory_hi = """✅ **इस सीजन में उत्पादन संतुलित होने की उम्मीद है।**

आपूर्ति मांग के अनुरूप होने की संभावना है, जो स्थिर बाजार स्थितियों का संकेत देती है।

**सुझाए गए कदम:**
• नियोजित खेती सामान्य रूप से जारी रखें
• वर्षा और बाजार के रुझानों पर नियमित नजर रखें
• अचानक रकबा विस्तार से बचें
• अच्छी कृषि पद्धतियां बनाए रखें
• सरकारी सलाह पर नजर रखें"""

                advisory_mr = """✅ **या हंगामात उत्पादन संतुलित राहण्याची अपेक्षा आहे.**

पुरवठा मागणीनुसार होण्याची शक्यता आहे, जे स्थिर बाजार परिस्थिती दर्शवते.

**सुचवलेल्या पावले:**
• नियोजित शेती नेहमीप्रमाणे सुरू ठेवा
• पाऊस आणि बाजार ट्रेंड नियमितपणे तपासा
• अचानक क्षेत्र वाढ टाळा
• चांगल्या शेती पद्धती राखा
• सरकारी सल्ल्यावर लक्ष ठेवा"""
                
                st.success(advisory_hi if lang=='hi' else advisory_mr if lang=='mr' else advisory_en)
            
            else:  # UNDERPRODUCTION
                advisory_en = """⚠️ **Underproduction expected this season.**

Market prices may rise due to limited supply, creating favorable conditions for farmers.

**Suggested actions:**
• Consider increasing onion acreage by 10–15%
• Ensure timely application of fertilizer and irrigation
• Prepare for higher market demand
• Avoid panic selling during early season
• Plan for optimal harvest timing"""

                advisory_hi = """⚠️ **इस सीजन में कम उत्पादन की उम्मीद है।**

सीमित आपूर्ति के कारण बाजार की कीमतें बढ़ सकती हैं, जो किसानों के लिए अनुकूल स्थिति बनाती है।

**सुझाए गए कदम:**
• प्याज का रकबा 10–15% बढ़ाने पर विचार करें
• उर्वरक और सिंचाई का समय पर प्रयोग सुनिश्चित करें
• उच्च बाजार मांग के लिए तैयार रहें
• प्रारंभिक सीजन में घबराहट में बिक्री से बचें
• इष्टतम कटाई के समय की योजना बनाएं"""

                advisory_mr = """⚠️ **या हंगामात कमी उत्पादन अपेक्षित आहे.**

मर्यादित पुरवठ्यामुळे बाजार किंमती वाढू शकतात, जे शेतकऱ्यांसाठी अनुकूल परिस्थिती निर्माण करते.

**सुचवलेल्या पावले:**
• कांदा क्षेत्र 10–15% वाढवण्याचा विचार करा
• खत आणि सिंचन वेळेवर करा
• जास्त बाजार मागणीसाठी तयार राहा
• सुरुवातीच्या हंगामात घाबरून विक्री टाळा
• इष्टतम कापणी वेळेची योजना करा"""
                
                st.warning(advisory_hi if lang=='hi' else advisory_mr if lang=='mr' else advisory_en)
            
            # Historical comparison chart
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"""<div class="panel-title">📊 {'ऐतिहासिक रुझान' if lang=='hi' else 'ऐतिहासिक ट्रेंड' if lang=='mr' else 'Historical Trend'}</div>""",
                        unsafe_allow_html=True)
            
            recent_hist = prod_status_df[prod_status_df['year'] >= 2015].copy()
            
            # Map status to colors using the correct key names
            _color_map = {
                'OVERPRODUCTION': '#E74C3C',
                'APT': '#2ECC71',
                'UNDERPRODUCTION': '#F39C12'
            }
            bar_colors = [_color_map.get(s, '#8B7355') for s in recent_hist['production_status']]
            bar_labels = [f"{status_config.get(s, {}).get('label_en', s)}" for s in recent_hist['production_status']]
            
            fig_status = go.Figure()
            fig_status.add_trace(go.Bar(
                x=recent_hist['year'].astype(str),
                y=recent_hist['seeds_distributed_mt'],
                marker_color=bar_colors,
                marker_line_color='rgba(139,115,74,0.5)', marker_line_width=0.5,
                text=bar_labels,
                textposition='outside',
                textfont=dict(color='#C4A55A', family='Georgia', size=8),
                customdata=list(zip(
                    recent_hist['production_status'],
                    recent_hist['area_hectares'],
                    recent_hist['seeds_distributed_mt']
                )),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Status: <b>%{customdata[0]}</b><br>'
                    'Area: %{customdata[1]:,.0f} ha<br>'
                    'Seeds: %{customdata[2]:,.0f} MT<extra></extra>'
                )
            ))
            fig_status.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#8B7355", tickfont=dict(color="#8B7355", family="Georgia"),
                           title=f"{'वर्ष' if lang in ['hi','mr'] else 'Year'}", showgrid=False),
                yaxis=dict(color="#8B7355", tickfont=dict(color="#8B7355", family="Georgia", size=9),
                           title=f"{'बीज वितरित (मे.ट.)' if lang=='hi' else 'बियाणे (मे.ट.)' if lang=='mr' else 'Seeds Distributed (MT)'}",
                           gridcolor="rgba(139,115,74,0.1)"),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_status, use_container_width=True, key="status_hist")
            
            # Legend
            leg1, leg2, leg3 = st.columns(3)
            with leg1:
                st.markdown(f"🔴 {'अति उत्पादन' if lang=='hi' else 'अतिरिक्त उत्पादन' if lang=='mr' else 'OVERPRODUCTION'}")
            with leg2:
                st.markdown(f"🟢 {'संतुलित उत्पादन' if lang in ['hi','mr'] else 'BALANCED (APT)'}")
            with leg3:
                st.markdown(f"🟡 {'कम उत्पादन' if lang=='hi' else 'कमी उत्पादन' if lang=='mr' else 'UNDERPRODUCTION'}")
        
        # ═══════════════════════════════════════════════════════════════
        # TAB 4: LIVE MANDI PRICES
        # ═══════════════════════════════════════════════════════════════
        with tab4:
            st.markdown(f"""
            <div class="panel-title">⊛ {'लाइव्ह मंडी बाजार भाव' if lang=='mr' else 'लाइव मंडी बाजार भाव' if lang=='hi' else 'Live Mandi Market Prices'}
            <span style="font-size:0.65rem;color:var(--sage);margin-left:12px;">↻ {'दर 2 मिनिटांनी' if lang=='mr' else 'हर 2 मिनट में' if lang=='hi' else 'Every 2 min'}</span>
            </div>""", unsafe_allow_html=True)

            sub_tab1, sub_tab2 = st.tabs([
                f"📍 {'स्थानिक बाजारपेठा' if lang=='mr' else 'स्थानीय बाजार' if lang=='hi' else 'Local Markets'}",
                f"🗺️ {'शेजारील राज्ये' if lang=='mr' else 'पड़ोसी राज्य' if lang=='hi' else 'Neighbouring States'}"
            ])

            # --- Sub-tab 1: Local Markets (Existing logic) ---
            with sub_tab1:
                if 'farmer_last_refresh' not in st.session_state:
                    st.session_state.farmer_last_refresh = datetime.now()
                fdiff = (datetime.now() - st.session_state.farmer_last_refresh).total_seconds()
                fnext = max(0, 120 - int(fdiff))

                fm1, fm2 = st.columns([3, 1])
                with fm1:
                    farmer_rank = st.radio(
                        "Sort:", ["💰 Highest Price", "📍 Nearest", "⚖️ Best Balance"],
                        horizontal=True, key="farmer_rank_local", label_visibility="collapsed"
                    )
                with fm2:
                    st.caption(f"⏰ {fnext//60}:{fnext%60:02d}")
                    if st.button("🔄 Refresh", key="farmer_mandi_refresh_local"):
                        st.session_state.farmer_last_refresh = datetime.now()
                        st.rerun()

                fmode = "price" if "Highest" in farmer_rank else "distance" if "Nearest" in farmer_rank else "balanced"
                base_px_f = float(datasets["onion_price"].sort_values(['year','month'])['price_per_quintal'].iloc[-1]) * 14.3
                df_f = _generate_mandi_prices(base_price=base_px_f)
                df_fr = _rank_mandis(df_f, fmode)

                # Top 5 cards
                st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                            color:var(--straw);margin:8px 0 6px;text-transform:uppercase;">
                            {'शीर्ष 5 शिफारशी' if lang=='mr' else 'शीर्ष 5 सिफारिशें' if lang=='hi' else 'Top 5 Recommendations'}</div>""",
                            unsafe_allow_html=True)

                for i, (_, frow) in enumerate(df_fr.head(5).iterrows()):
                    fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 3])
                    bg = "rgba(232,200,117,0.07)" if i == 0 else "rgba(255,255,255,0.02)"
                    brd = "var(--wheat)" if i == 0 else "var(--bark)"
                    with fc1:
                        st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;height:60px;">
                            <div style="font-family:'Cinzel',serif;color:var(--harvest);font-size:0.82rem;">#{frow['Rank']} {frow['Mandi']}</div>
                            <div style="font-size:0.7rem;color:var(--straw);">📍 {frow['Distance (km)']} km</div>
                        </div>""", unsafe_allow_html=True)
                    with fc2:
                        st.metric("Price", f"₹{frow['Price (₹/Q)']}/Q")
                    with fc3:
                        st.metric("Arrival", f"{frow['Arrival (T)']}T")
                    with fc4:
                        st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;height:60px;">
                            <div style="font-size:0.8rem;color:var(--cream);">{frow['Rec']}</div>
                            <div style="font-size:0.68rem;color:var(--straw);">Grade A: {frow['Grade A (%)']}%</div>
                        </div>""", unsafe_allow_html=True)

                with st.expander(f"📋 {'सर्व मंडी पहा' if lang=='mr' else 'सभी मंडी देखें' if lang=='hi' else 'View All Mandis'}"):
                    disp_cols = ['Rank','Mandi','Distance (km)','Price (₹/Q)','Arrival (T)','Grade A (%)','Rec']
                    st.dataframe(df_fr[disp_cols].rename(columns={"Rec":"Recommendation","Price (₹/Q)":"Price (₹/Quintal)"}),
                                 use_container_width=True, hide_index=True)

                # Market insights
                hi_p = df_f['Price (₹/Q)'].max()
                hi_m = df_f.loc[df_f['Price (₹/Q)'].idxmax(), 'Mandi']
                avg_p = df_f['Price (₹/Q)'].mean()
                near2 = df_f.nsmallest(2, 'Distance (km)').iloc[-1]

                st.markdown("<hr>", unsafe_allow_html=True)
                _lbl1 = 'सर्वाधिक किंमत' if lang=='mr' else 'सर्वोच्च मूल्य' if lang=='hi' else 'Highest Price'
                _lbl2 = 'सरासरी किंमत'   if lang=='mr' else 'औसत मूल्य'      if lang=='hi' else 'Average Price'
                _lbl3 = 'जवळील चांगली किंमत' if lang=='mr' else 'निकटतम अच्छी कीमत' if lang=='hi' else 'Nearest Good Price'
                ins1, ins2, ins3 = st.columns(3)
                with ins1:
                    st.info(f"**{_lbl1}:** {hi_m} — ₹{hi_p:.1f}/Q")
                with ins2:
                    st.info(f"**{_lbl2}:** Across all mandis — ₹{avg_p:.1f}/Q")
                with ins3:
                    st.info(f"**{_lbl3}:** {near2['Mandi']} — ₹{near2['Price (₹/Q)']:.1f}/Q")

                # Transport tip
                if "Highest" in farmer_rank:
                    tip = "✅ Check fuel cost vs price premium · साझा परिवहन से बचत करें · शेअर ट्रान्सपोर्ट वापरा"
                else:
                    tip = "✅ Best price + distance balance · कम लागत = अधिक मुनाफा · कमी खर्च = जास्त नफा"
                st.markdown(f"""
                <div style="padding:10px 14px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);
                            border-radius:3px;font-size:0.82rem;color:var(--cream);margin-top:8px;">🚛 {tip}</div>
                """, unsafe_allow_html=True)

                # Auto-refresh
                if fdiff >= 120:
                    st.session_state.farmer_last_refresh = datetime.now()
                    st.rerun()

            # --- Sub-tab 2: Neighbouring States ---
            with sub_tab2:

                # State APMC data (distance from Nashik, price factor vs base)
                STATE_MANDIS = [
                    ("Kalaburagi APMC",  "Karnataka",        380, 0.97),
                    ("Gadag APMC",       "Karnataka",        450, 0.96),
                    ("Hubli APMC",       "Karnataka",        410, 0.98),
                    ("Belagavi APMC",    "Karnataka",        340, 0.95),
                    ("Dharwad APMC",     "Karnataka",        420, 0.97),
                    ("Nizamabad APMC",   "Telangana",        470, 0.99),
                    ("Hyderabad APMC",   "Telangana",        560, 1.04),
                    ("Warangal APMC",    "Telangana",        580, 1.01),
                    ("Karimnagar APMC",  "Telangana",        540, 1.00),
                    ("Guntur APMC",      "Andhra Pradesh",   620, 1.06),
                    ("Kurnool APMC",     "Andhra Pradesh",   530, 1.02),
                    ("Kadapa APMC",      "Andhra Pradesh",   650, 1.03),
                    ("Anantapur APMC",   "Andhra Pradesh",   590, 1.01),
                    ("Indore APMC",      "Madhya Pradesh",   510, 0.98),
                    ("Bhopal APMC",      "Madhya Pradesh",   590, 0.96),
                    ("Ujjain APMC",      "Madhya Pradesh",   480, 0.97),
                    ("Mandsaur APMC",    "Madhya Pradesh",   430, 0.99),
                    ("Ratlam APMC",      "Madhya Pradesh",   400, 0.98),
                ]

                def _generate_state_prices(base_price=200.0):
                    cur = datetime.now()
                    mins = (cur.minute // 2) * 2
                    ts   = cur.replace(minute=mins, second=0, microsecond=0)
                    np.random.seed((int(ts.timestamp()) + 9999) % 100000)
                    rows = []
                    for name, state, dist, factor in STATE_MANDIS:
                        mbase   = base_price * factor
                        dv      = np.random.uniform(-0.06, 0.06)
                        h       = cur.hour
                        tf      = 1.03 if 6 <= h < 10 else 0.97 if 16 <= h < 20 else 1.0
                        sd      = np.random.uniform(-0.04, 0.04)
                        price   = max(50, mbase * (1 + dv) * tf * (1 + sd))
                        arrival = np.random.uniform(30, 90)
                        ga = np.random.uniform(38, 62); gb = np.random.uniform(25, 38); gc = 100 - ga - gb
                        rows.append({
                            "Mandi":          name,
                            "State":          state,
                            "Distance (km)":  dist,
                            "Price (₹/Q)":    round(price, 1),
                            "Arrival (T)":    round(arrival, 1),
                            "Grade A (%)":    round(ga, 1),
                            "Grade B (%)":    round(gb, 1),
                            "Grade C (%)":    round(gc, 1),
                            "Updated":        ts.strftime("%I:%M %p"),
                        })
                    return pd.DataFrame(rows)

                STATE_COLORS = {
                    "Karnataka":      "#4E6B3A",
                    "Telangana":      "#3A5E6B",
                    "Andhra Pradesh": "#6B4E3A",
                    "Madhya Pradesh": "#5A4E6B",
                }

                # Controls row
                sf1, sf2 = st.columns([2, 2])
                with sf1:
                    st.markdown(f"""<div style="font-size:0.72rem;color:var(--straw);margin-bottom:4px;font-family:'Cinzel',serif;letter-spacing:0.1em;">
                        {'राज्य फिल्टर' if lang in ['hi','mr'] else 'STATE FILTER'}</div>""", unsafe_allow_html=True)
                    state_filter = st.selectbox(
                        "State", ["All States", "Karnataka", "Telangana", "Andhra Pradesh", "Madhya Pradesh"],
                        label_visibility="collapsed", key="state_mandi_filter"
                    )
                with sf2:
                    state_rank = st.radio(
                        "Sort:", ["💰 Highest Price", "📍 Nearest", "⚖️ Best Balance"],
                        horizontal=True, key="farmer_rank_state", label_visibility="collapsed"
                    )

                df_st = _generate_state_prices(base_price=base_px_f)
                df_st_filt = df_st[df_st["State"] == state_filter].copy() if state_filter != "All States" else df_st.copy()
                st_mode    = "price" if "Highest" in state_rank else "distance" if "Nearest" in state_rank else "balanced"
                df_st_r    = _rank_mandis(df_st_filt, st_mode)

                st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                            color:var(--straw);margin:10px 0 6px;text-transform:uppercase;">
                            {'शीर्ष 5 · शिफारशी' if lang=='mr' else 'शीर्ष 5 · सिफारिशें' if lang=='hi' else 'Top 5 · Recommendations'}</div>""",
                            unsafe_allow_html=True)

                for i, (_, srow) in enumerate(df_st_r.head(5).iterrows()):
                    sc1, sc2, sc3, sc4 = st.columns([3, 2, 2, 3])
                    bg  = "rgba(232,200,117,0.07)" if i == 0 else "rgba(255,255,255,0.02)"
                    brd = "var(--wheat)"           if i == 0 else "var(--bark)"
                    sc  = STATE_COLORS.get(srow['State'], '#555')
                    np_s = _net_profit(srow['Price (₹/Q)'], srow['Distance (km)'])
                    with sc1:
                        st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;height:72px;">
                            <div style="font-family:'Cinzel',serif;color:var(--harvest);font-size:0.82rem;">#{srow['Rank']} {srow['Mandi']}</div>
                            <div style="font-size:0.68rem;color:var(--straw);">📍 {srow['Distance (km)']} km from Nashik</div>
                            <div style="display:inline-block;margin-top:3px;padding:1px 6px;background:{sc};border-radius:3px;font-size:0.6rem;color:#fff;letter-spacing:0.08em;">
                                {srow['State'].upper()}</div>
                        </div>""", unsafe_allow_html=True)
                    with sc2:
                        st.metric("Price / मूल्य", f"₹{srow['Price (₹/Q)']}/Q")
                    with sc3:
                        st.metric("Arrival", f"{srow['Arrival (T)']}T")
                    with sc4:
                        st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;height:72px;">
                            <div style="font-size:0.8rem;color:var(--cream);">{srow['Rec']}</div>
                            <div style="font-size:0.67rem;color:var(--straw);">Net/Q: ₹{np_s['per_quintal']:.0f} · Transport: ₹{np_s['transport']:.0f}</div>
                        </div>""", unsafe_allow_html=True)

                with st.expander(f"📋 {'सर्व राज्य मंडी' if lang=='mr' else 'सभी राज्य मंडी' if lang=='hi' else 'View All State Mandis'}"):
                    s_disp = ['Rank','Mandi','State','Distance (km)','Price (₹/Q)','Arrival (T)','Grade A (%)','Rec']
                    st.dataframe(df_st_r[s_disp].rename(columns={"Rec":"Recommendation","Price (₹/Q)":"Price (₹/Quintal)"}),
                                 use_container_width=True, hide_index=True)

                # Per-state average bar chart
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;color:var(--straw);margin-bottom:8px;">
                    ◆ {'राज्यनिहाय सरासरी भाव' if lang=='mr' else 'राज्यवार औसत मूल्य' if lang=='hi' else 'STATE-WISE AVERAGE PRICE'}</div>""",
                    unsafe_allow_html=True)
                state_avg = df_st.groupby("State")["Price (₹/Q)"].mean().reset_index()
                state_avg.columns = ["State", "Avg"]
                state_avg = state_avg.sort_values("Avg", ascending=False)
                import plotly.graph_objects as _go2
                fig_s = _go2.Figure(_go2.Bar(
                    x=state_avg["State"], y=state_avg["Avg"],
                    marker_color=["#E8C875","#4A7C59","#C87E45","#5A8B6E"],
                    text=[f"₹{v:.0f}" for v in state_avg["Avg"]], textposition="outside"
                ))
                fig_s.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#D4C5A9", size=11),
                    yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    margin=dict(t=30, b=10, l=10, r=10), height=250,
                )
                st.plotly_chart(fig_s, use_container_width=True, key="state_avg_chart")

                st.markdown(f"""
                <div style="padding:8px 12px;background:rgba(232,200,117,0.07);border:1px solid var(--wheat);border-radius:3px;font-size:0.78rem;color:var(--straw);">
                    ⚠️ {'लांब अंतरासाठी वाहतूक खर्च लक्षात घ्या · FPO द्वारे एकत्र वाहतूक किफायतशीर' if lang=='mr'
                        else 'लंबी दूरी के लिए परिवहन लागत का ध्यान रखें · FPO के माध्यम से साझा परिवहन लाभदायक' if lang=='hi'
                        else 'Factor in long-distance transport cost · Shared FPO transport cuts per-quintal expense'}
                </div>
                """, unsafe_allow_html=True)

            # ── EXPORT INSTITUTION CONTACTS ─────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;color:var(--straw);margin-bottom:6px;">
                ◆ 📞 {'निर्यात सहाय्यक संस्था — संपर्क' if lang=='mr' else 'निर्यात सहायक संस्थाएं — संपर्क' if lang=='hi' else 'EXPORT SUPPORT INSTITUTIONS — CONTACT'}
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:0.82rem;color:var(--straw);margin-bottom:12px;">
                {'खालील संस्था शेतकऱ्यांना माल निर्यात करण्यात मदत करतात.' if lang=='mr'
                 else 'नीचे दी गई संस्थाएं किसानों को उपज निर्यात करने में सहायता करती हैं।' if lang=='hi'
                 else 'The following institutions assist farmers in exporting their produce.'}
            </div>""", unsafe_allow_html=True)

            EXPORT_CONTACTS = [
                # Karnataka
                {"name": "Karnataka State Agricultural Marketing Board (KSAMB)",
                 "type": "State Govt. Body", "state": "Karnataka",
                 "loc":  "Bengaluru, Karnataka",
                 "phone": "+91-080-2294-5700", "email": "ksamb@karnataka.gov.in",
                 "desc": "Regulates APMCs, market fees and supports farmer-to-buyer linkages across Karnataka"},
                {"name": "Horticulture Dept. Karnataka (Kalaburagi APMC)",
                 "type": "Govt. Export Authority", "state": "Karnataka",
                 "loc":  "Kalaburagi, Karnataka",
                 "phone": "+91-08472-263-410", "email": "ddhort.klb@karnataka.gov.in",
                 "desc": "Onion and vegetable grading, packaging and export certification in north Karnataka"},
                # Telangana
                {"name": "Telangana State Agricultural Marketing Board (TSAMB)",
                 "type": "State Govt. Body", "state": "Telangana",
                 "loc":  "Hyderabad, Telangana",
                 "phone": "+91-040-2329-1430", "email": "tsamb@telangana.gov.in",
                 "desc": "Governs APMC markets; price intelligence and dissemination to farmers across Telangana"},
                {"name": "Nizamabad Vegetables and Fruits Export Hub",
                 "type": "FPO / Exporter", "state": "Telangana",
                 "loc":  "Nizamabad, Telangana",
                 "phone": "+91-94904-55678", "email": "nzb.exporthub@gmail.com",
                 "desc": "Bulk procurement and export of onions, tomatoes to Gulf and SE Asian markets"},
                # Andhra Pradesh
                {"name": "AP Agricultural Marketing Dept. (Guntur)",
                 "type": "Govt. Export Authority", "state": "Andhra Pradesh",
                 "loc":  "Guntur, Andhra Pradesh",
                 "phone": "+91-0863-223-5000", "email": "agmktg.gntr@ap.gov.in",
                 "desc": "Key onion and chilli market authority; export documentation and buyer linkage services"},
                {"name": "ANGRAU Kurnool Agri Export Centre",
                 "type": "Govt. Processing & Export", "state": "Andhra Pradesh",
                 "loc":  "Kurnool, Andhra Pradesh",
                 "phone": "+91-08518-252-100", "email": "kurnool.export@angrau.ac.in",
                 "desc": "Post-harvest technology, cold chain support and grading facility for onion exporters in AP"},
                # Madhya Pradesh
                {"name": "MP Mandi Board (Indore Region)",
                 "type": "State Govt. Body", "state": "Madhya Pradesh",
                 "loc":  "Indore, Madhya Pradesh",
                 "phone": "+91-0731-270-8022", "email": "ceo.mandiboard@mp.gov.in",
                 "desc": "Manages 259 APMCs in MP; price reporting, auction management and export coordination"},
                {"name": "MP Agro Export Zone - Mandsaur Onion Hub",
                 "type": "FPO / Exporter", "state": "Madhya Pradesh",
                 "loc":  "Mandsaur, Madhya Pradesh",
                 "phone": "+91-07422-245-678", "email": "mpagro.mandsaur@gmail.com",
                 "desc": "Direct export facilitation for rabi onion growers to Bangladesh, Nepal and Sri Lanka"},
            ]

            TYPE_COLORS = {
                "FPO / Exporter":            "#4A6B3A",
                "Govt. Processing & Export": "#3A5B6B",
                "Govt. Export Authority":    "#3A5B6B",
                "State Govt. Body":          "#3A5B6B",
                "Trade Association":         "#6B5A3A",
                "Financial Institution":     "#5A3A6B",
            }

            for ci in range(0, len(EXPORT_CONTACTS), 2):
                ec1, ec2 = st.columns(2)
                for col, c in zip([ec1, ec2], EXPORT_CONTACTS[ci:ci+2]):
                    tc = TYPE_COLORS.get(c['type'], '#444')
                    with col:
                        st.markdown(f"""
                        <div style="padding:14px 16px;background:rgba(255,255,255,0.03);border:1px solid var(--bark);
                                    border-radius:6px;margin-bottom:10px;">
                            <div style="font-family:'Cinzel',serif;font-size:0.88rem;color:var(--harvest);margin-bottom:6px;">
                                {c['name']}
                            </div>
                            <div style="margin-bottom:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                                <span style="display:inline-block;padding:1px 8px;background:{tc};border-radius:3px;
                                             font-size:0.6rem;color:#fff;letter-spacing:0.08em;text-transform:uppercase;">
                                    {c['type']}</span>
                                <span style="font-size:0.72rem;color:var(--straw);">📍 {c['loc']}</span>
                            </div>
                            <div style="font-size:0.78rem;color:var(--cream);margin-bottom:4px;">
                                📞 <b>{c['phone']}</b> &nbsp;&nbsp; ✉️ <span style="color:var(--sage);">{c['email']}</span>
                            </div>
                            <div style="font-size:0.72rem;color:var(--straw);font-style:italic;margin-top:4px;">
                                {c['desc']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="padding:8px 12px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);border-radius:3px;
                        font-size:0.75rem;color:var(--straw);margin-top:4px;">
                ℹ️ {'वरील संपर्क माहिती उद्देशासाठी आहे. अधिकृत संकेतस्थळावर खात्री करा.' if lang=='mr'
                    else 'उपरोक्त संपर्क जागरूकता उद्देश्यों के लिए है। आधिकारिक वेबसाइट से सत्यापित करें।' if lang=='hi'
                    else 'Contact details are for informational purposes. Verify via official websites before reaching out.'}
            </div>
            """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 5: PREDICTIVE DISEASE PREVENTION — ENHANCED
        #   • CSV upload/replace for timeline data
        #   • Day-by-day full-season disease forecast (every 10 days)
        #   • CNN model suggestion integrated into forecast
        # ═══════════════════════════════════════════════════════════════
        with tab5:
            st.markdown(f"""<div class="panel-title">🛡 {'पूर्व रोग रोकथाम' if lang=='hi' else 'पूर्व रोग प्रतिबंध' if lang=='mr' else 'Predictive Disease Prevention'}</div>""",
                        unsafe_allow_html=True)

            # ── CSV UPLOAD SECTION ─────────────────────────────────────
            st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                color:var(--straw);margin:0 0 6px;text-transform:uppercase;">
                📂 {'रोग समयरेखा डेटा' if lang=='hi' else 'रोग टाइमलाइन डेटा' if lang=='mr' else 'Disease Timeline Data Source'}
            </div>""", unsafe_allow_html=True)

            _dis_tl_path = os.path.join(BASE_DIR, "onion_disease_timeline.csv")
            _tl_upload_col, _tl_info_col = st.columns([2, 3])
            with _tl_upload_col:
                _tl_csv_file = st.file_uploader(
                    f"📁 {'टाइमलाइन CSV अपलोड करें (वैकल्पिक)' if lang=='hi' else 'टाइमलाइन CSV अपलोड करा (पर्यायी)' if lang=='mr' else 'Upload Timeline CSV (optional)'}",
                    type=["csv"], key="tl_csv_uploader",
                    help="Columns: crop, disease, day_start, day_end, risk_level, humidity_factor, temp_factor, rain_factor, preventive_advice"
                )
                if _tl_csv_file is not None:
                    try:
                        _tl_uploaded_df = pd.read_csv(_tl_csv_file)
                        _required_cols = {"crop","disease","day_start","day_end","risk_level",
                                          "humidity_factor","temp_factor","rain_factor","preventive_advice"}
                        if _required_cols.issubset(set(_tl_uploaded_df.columns)):
                            _tl_uploaded_df.to_csv(_dis_tl_path, index=False)
                            _load_onion_disease_timeline.clear()
                            st.success(f"✅ {'टाइमलाइन अपडेट हुई — {len(_tl_uploaded_df)} रोग प्रविष्टियां' if lang=='hi' else f'टाइमलाइन अपडेट — {len(_tl_uploaded_df)} नोंदी' if lang=='mr' else f'Timeline updated — {len(_tl_uploaded_df)} disease entries loaded'}")
                        else:
                            missing = _required_cols - set(_tl_uploaded_df.columns)
                            st.error(f"❌ Missing columns: {', '.join(missing)}")
                    except Exception as _e:
                        st.error(f"❌ CSV parse error: {_e}")

            with _tl_info_col:
                _dis_timeline = _load_onion_disease_timeline(_dis_tl_path)
                st.markdown(f"""
                <div style="padding:10px 14px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);
                            border-radius:4px;font-size:0.82rem;color:var(--cream);">
                    {'📊 ' + str(len(_dis_timeline)) + ' रोग प्रविष्टियां लोड हैं। ' if lang=='hi'
                     else '📊 ' + str(len(_dis_timeline)) + ' रोग नोंदी लोड आहेत. ' if lang=='mr'
                     else f'📊 {len(_dis_timeline)} disease entries loaded. '}
                    {'CSV अपलोड करके कस्टम डेटा जोड़ें।' if lang=='hi'
                     else 'CSV अपलोड करून कस्टम डेटा जोडा.' if lang=='mr'
                     else 'Upload a CSV to use your own custom data.'}
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"{'टाइमलाइन डेटा देखें' if lang in ['hi','mr'] else 'View / Download Current Timeline Data'}"):
                    st.dataframe(_dis_timeline, use_container_width=True, hide_index=True)
                    _tl_csv_bytes = _dis_timeline.to_csv(index=False).encode()
                    st.download_button(
                        label=f"⬇️ {'CSV डाउनलोड करें' if lang=='hi' else 'CSV डाउनलोड करा' if lang=='mr' else 'Download Timeline CSV'}",
                        data=_tl_csv_bytes,
                        file_name="onion_disease_timeline.csv",
                        mime="text/csv", key="dl_timeline_csv"
                    )

            st.markdown("<hr style='margin:12px 0;'>", unsafe_allow_html=True)

            # ── INPUTS ────────────────────────────────────────────────
            _pc1, _pc2, _pc3 = st.columns(3)
            with _pc1:
                _prev_sowing = st.date_input(
                    f"🌱 {'पेरणी/बुवाई तारीख' if lang in ['hi','mr'] else 'Crop Sowing Date'}",
                    value=_date.today().replace(month=1, day=1),
                    key="prev_sowing_date_farmer"
                )
            with _pc2:
                _prev_crop_age = (_date.today() - _prev_sowing).days
                st.markdown(f"""
                <div class="metric-box" style="margin-top:8px;">
                    <div class="metric-label">{'फसल की आयु' if lang=='hi' else 'पीक वय' if lang=='mr' else 'Crop Age Today'}</div>
                    <div class="metric-value" style="color:var(--harvest);">{_prev_crop_age}</div>
                    <div class="metric-sub">{'दिन' if lang in ['hi','mr'] else 'Days Since Sowing'}</div>
                </div>""", unsafe_allow_html=True)
            with _pc3:
                _forecast_days = st.slider(
                    f"{'आगे के दिन' if lang=='hi' else 'पुढील दिवस' if lang=='mr' else 'Forecast Horizon (days ahead)'}",
                    min_value=10, max_value=90, value=60, step=10, key="forecast_horizon"
                )

            st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                color:var(--straw);margin:12px 0 6px;text-transform:uppercase;">
                🌤️ {'वर्तमान मौसम' if lang=='hi' else 'सध्याचे हवामान' if lang=='mr' else 'Current Weather Conditions (from forecast)'}
            </div>""", unsafe_allow_html=True)

            _wc1, _wc2, _wc3 = st.columns(3)
            with _wc1:
                _prev_avg_temp = float(weather_df['temp_c'].mean())
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-label">🌡️ {'तापमान' if lang in ['hi','mr'] else 'Avg Temperature'}</div>
                    <div class="metric-value" style="font-size:1.1rem;">{_prev_avg_temp:.1f}°C</div>
                    <div class="metric-sub">{'7-दिन औसत' if lang=='hi' else '7-दिवस सरासरी' if lang=='mr' else '7-day avg'}</div>
                </div>""", unsafe_allow_html=True)
            with _wc2:
                _prev_avg_rain = float(weather_df['rain_pct'].mean())
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-label">🌧️ {'बारिश संभावना' if lang=='hi' else 'पाऊस शक्यता' if lang=='mr' else 'Rain Probability'}</div>
                    <div class="metric-value" style="font-size:1.1rem;">{_prev_avg_rain:.0f}%</div>
                    <div class="metric-sub">{'7-दिन औसत' if lang=='hi' else '7-दिवस सरासरी' if lang=='mr' else '7-day avg'}</div>
                </div>""", unsafe_allow_html=True)
            with _wc3:
                _prev_avg_humid = min(95.0, 45.0 + _prev_avg_rain * 0.5)
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-label">💧 {'आर्द्रता अनुमान' if lang=='hi' else 'आर्द्रता अंदाज' if lang=='mr' else 'Est. Humidity'}</div>
                    <div class="metric-value" style="font-size:1.1rem;">{_prev_avg_humid:.0f}%</div>
                    <div class="metric-sub">{'वर्षा पर आधारित' if lang=='hi' else 'पावसावर आधारित' if lang=='mr' else 'Based on rainfall'}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            # ── SHARED RISK SCORING FUNCTION ──────────────────────────
            def _calc_risk_score_at(row, crop_age, avg_temp, avg_humid, avg_rain):
                base = 0.3
                h_w = (0.25 if (avg_humid > 70 and row["humidity_factor"] == "high")
                       else 0.10 if (avg_humid > 50 and row["humidity_factor"] in ["high","moderate"])
                       else 0.0)
                r_w = (0.20 if (avg_rain > 60 and row["rain_factor"] == "high")
                       else 0.10 if (avg_rain > 40 and row["rain_factor"] in ["high","moderate"])
                       else 0.0)
                t_w = (0.20 if (avg_temp > 32 and row["temp_factor"] in ["hot","warm"])
                       else 0.10 if (avg_temp > 25 and row["temp_factor"] == "warm")
                       else 0.05 if (avg_temp < 20 and row["temp_factor"] == "cool")
                       else 0.0)
                _mid = (row["day_start"] + row["day_end"]) / 2
                _prox = max(0, 1 - abs(crop_age - _mid) / max((row["day_end"] - row["day_start"]) / 2, 1))
                peak_w = 0.15 * _prox
                return min(base + h_w + r_w + t_w + peak_w, 1.0)

            # ── ANALYSE CURRENT DAY ───────────────────────────────────
            _analyse_cols = st.columns([3, 1])
            with _analyse_cols[0]:
                _run_analysis = st.button(
                    f"🔍 {'रोग जोखिम विश्लेषण करें' if lang=='hi' else 'रोग जोखीम विश्लेषण करा' if lang=='mr' else 'Analyse Disease Risk Now + Full Season Forecast'}",
                    key="btn_farmer_disease_risk", use_container_width=True
                )

            if _run_analysis:
                if _prev_crop_age < 0:
                    st.error("❌ Sowing date cannot be in the future.")
                else:
                    # ── TODAY: active disease windows ─────────────────
                    _active = _dis_timeline[
                        (_dis_timeline["day_start"] <= _prev_crop_age) &
                        (_dis_timeline["day_end"] >= _prev_crop_age)
                    ].copy()

                    _wctx = []
                    if _prev_avg_humid > 70: _wctx.append("High Humidity 💧")
                    if _prev_avg_temp > 32:  _wctx.append("High Temp 🌡️")
                    if _prev_avg_rain > 60:  _wctx.append("High Rain 🌧️")
                    _wlabel = ", ".join(_wctx) if _wctx else "Moderate conditions"

                    st.markdown(f"""
                    <div style="display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap;">
                        <div style="padding:8px 14px;background:rgba(139,125,74,0.1);border:1px solid var(--bark);border-radius:4px;font-size:0.8rem;">
                            🌤️ <b>{'मौसम' if lang in ['hi','mr'] else 'Weather'}:</b> {_wlabel}
                        </div>
                        <div style="padding:8px 14px;background:rgba(139,125,74,0.1);border:1px solid var(--bark);border-radius:4px;font-size:0.8rem;">
                            📅 <b>{'आज पीक वय' if lang in ['hi','mr'] else 'Crop Age Today'}:</b> {_prev_crop_age} {'दिन' if lang in ['hi','mr'] else 'days'}
                        </div>
                        <div style="padding:8px 14px;background:rgba(139,125,74,0.1);border:1px solid var(--bark);border-radius:4px;font-size:0.8rem;">
                            ⚠️ <b>{'सक्रिय खिड़कियां' if lang=='hi' else 'सक्रिय खिडक्या' if lang=='mr' else 'Active Windows Now'}:</b> {len(_active)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── TODAY DISEASE CARDS ────────────────────────────
                    st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                        color:var(--straw);margin:4px 0 10px;text-transform:uppercase;">
                        🔴 {'आज के सक्रिय रोग' if lang=='hi' else 'आजचे सक्रिय रोग' if lang=='mr' else 'Active Diseases — Today (Day {_prev_crop_age})'}
                    </div>""", unsafe_allow_html=True)

                    if _active.empty:
                        st.success(f"✅ {'आज कोई सक्रिय रोग खिड़की नहीं।' if lang=='hi' else 'आज कोणताही सक्रिय रोग नाही.' if lang=='mr' else f'No active disease windows at day {_prev_crop_age}. Crop is in a safe zone.'}")
                    else:
                        _active["risk_score"] = _active.apply(
                            lambda r: _calc_risk_score_at(r, _prev_crop_age, _prev_avg_temp, _prev_avg_humid, _prev_avg_rain), axis=1
                        )
                        _active = _active.sort_values("risk_score", ascending=False)

                        for _, _drow in _active.iterrows():
                            _score = _drow["risk_score"]
                            if _score >= 0.7: _dlevel, _dcolor, _demoji = ("HIGH", "#e74c3c", "🔴")
                            elif _score >= 0.4: _dlevel, _dcolor, _demoji = ("MEDIUM", "#f39c12", "🟡")
                            else: _dlevel, _dcolor, _demoji = ("LOW", "#27ae60", "🟢")
                            _dname = str(_drow["disease"]).replace("_"," ").title()
                            _dreasons = []
                            if _prev_avg_humid > 70 and _drow["humidity_factor"] == "high":
                                _dreasons.append("High humidity matches trigger")
                            if _prev_avg_rain > 60 and _drow["rain_factor"] == "high":
                                _dreasons.append("High rainfall probability")
                            if _prev_avg_temp > 32 and _drow["temp_factor"] in ["hot","warm"]:
                                _dreasons.append("Temperature in risk zone")
                            _dmid = (_drow["day_start"] + _drow["day_end"]) / 2
                            if abs(_prev_crop_age - _dmid) < 10:
                                _dreasons.append("At peak risk window")
                            _dreason_text = " · ".join(_dreasons) if _dreasons else "Crop age within disease window"
                            st.markdown(f"""
                            <div style="background:{_dcolor}18;border-left:5px solid {_dcolor};
                                        padding:14px 18px;border-radius:8px;margin:8px 0;">
                                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                                    <h4 style="margin:0;color:{_dcolor};font-family:'Cinzel',serif;">{_demoji} {_dname}</h4>
                                    <span style="background:{_dcolor};color:white;padding:3px 10px;border-radius:10px;
                                                 font-weight:bold;font-size:12px;font-family:'Cinzel',serif;">
                                        {_dlevel} — {_score*100:.0f}%
                                    </span>
                                </div>
                                <p style="margin:6px 0 3px;font-size:0.75rem;color:var(--straw);">
                                    <b>📆 {'सक्रिय' if lang in ['hi','mr'] else 'Window'}:</b> Day {int(_drow['day_start'])}–{int(_drow['day_end'])}
                                    &nbsp;|&nbsp; <b>🔎 {'कारण' if lang in ['hi','mr'] else 'Reason'}:</b> {_dreason_text}
                                </p>
                                <p style="margin:3px 0 0;font-size:0.86rem;color:var(--cream);">
                                    <b>✅ {'उपाय' if lang in ['hi','mr'] else 'Action'}:</b> {_drow['preventive_advice']}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

                    # ════════════════════════════════════════════════════
                    # FULL-SEASON DAY-BY-DAY FORECAST TIMELINE
                    # ════════════════════════════════════════════════════
                    st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                        color:var(--straw);margin:4px 0 12px;text-transform:uppercase;">
                        📅 {'पूर्ण मौसम रोग पूर्वानुमान' if lang=='hi' else 'संपूर्ण हंगाम रोग अंदाज' if lang=='mr' else f'Full-Season Disease Forecast — Next {_forecast_days} Days'}
                    </div>""", unsafe_allow_html=True)

                    # Build forecast: for each 10-day checkpoint ahead, what diseases will be active?
                    _forecast_checkpoints = list(range(_prev_crop_age + 1, _prev_crop_age + _forecast_days + 1, 10))
                    if not _forecast_checkpoints:
                        _forecast_checkpoints = [_prev_crop_age + 10]

                    # Timeline chart data
                    _chart_rows = []
                    _timeline_cards = []

                    for _chk_day in _forecast_checkpoints:
                        _chk_date = _prev_sowing + timedelta(days=_chk_day)
                        _chk_diseases = _dis_timeline[
                            (_dis_timeline["day_start"] <= _chk_day) &
                            (_dis_timeline["day_end"] >= _chk_day)
                        ].copy()

                        if _chk_diseases.empty:
                            _chart_rows.append({
                                "Day": _chk_day,
                                "Date": _chk_date.strftime("%d %b"),
                                "Top Disease": "Safe Zone",
                                "Risk %": 0,
                                "Count": 0
                            })
                        else:
                            _chk_diseases["risk_score"] = _chk_diseases.apply(
                                lambda r: _calc_risk_score_at(r, _chk_day, _prev_avg_temp, _prev_avg_humid, _prev_avg_rain), axis=1
                            )
                            _chk_diseases = _chk_diseases.sort_values("risk_score", ascending=False)
                            _top = _chk_diseases.iloc[0]
                            _chart_rows.append({
                                "Day": _chk_day,
                                "Date": _chk_date.strftime("%d %b"),
                                "Top Disease": str(_top["disease"]).replace("_"," ").title(),
                                "Risk %": round(_top["risk_score"] * 100, 1),
                                "Count": len(_chk_diseases)
                            })
                            _timeline_cards.append((_chk_day, _chk_date, _chk_diseases))

                    _chart_df = pd.DataFrame(_chart_rows)

                    # Plotly timeline bar chart
                    _bar_colors = []
                    for _, _cr in _chart_df.iterrows():
                        if _cr["Risk %"] == 0: _bar_colors.append("#27ae60")
                        elif _cr["Risk %"] < 40: _bar_colors.append("#2ecc71")
                        elif _cr["Risk %"] < 60: _bar_colors.append("#f39c12")
                        elif _cr["Risk %"] < 75: _bar_colors.append("#e67e22")
                        else: _bar_colors.append("#e74c3c")

                    _fig_tl = go.Figure()
                    _fig_tl.add_trace(go.Bar(
                        x=[f"Day {r['Day']}<br>({r['Date']})" for _, r in _chart_df.iterrows()],
                        y=_chart_df["Risk %"],
                        marker_color=_bar_colors,
                        text=[f"{r['Top Disease']}<br>{r['Risk %']}%" for _, r in _chart_df.iterrows()],
                        textposition="outside",
                        textfont=dict(size=9, color="#D4C5A9"),
                        hovertemplate="<b>Day %{customdata[0]}</b><br>Top: %{customdata[1]}<br>Risk: %{y:.0f}%<br>Active diseases: %{customdata[2]}<extra></extra>",
                        customdata=[[r["Day"], r["Top Disease"], r["Count"]] for _, r in _chart_df.iterrows()]
                    ))
                    # Mark today
                    _fig_tl.add_vline(x=-0.5, line_dash="dot", line_color="rgba(232,200,117,0.5)", line_width=1)
                    _fig_tl.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#D4C5A9", size=10, family="Georgia"),
                        yaxis=dict(title="Peak Risk %", range=[0, 110], gridcolor="rgba(255,255,255,0.06)"),
                        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9)),
                        margin=dict(t=20, b=20, l=40, r=20), height=280,
                        showlegend=False,
                        title=dict(
                            text=f"{'रोग जोखिम पूर्वानुमान' if lang=='hi' else 'रोग जोखीम अंदाज' if lang=='mr' else 'Disease Risk Forecast — Next ' + str(_forecast_days) + ' Days'}",
                            font=dict(family="Cinzel", size=12, color="#E8C875"), x=0.02
                        )
                    )
                    st.plotly_chart(_fig_tl, use_container_width=True, key="disease_timeline_chart")

                    # ── DAY-BY-DAY FORECAST CARDS ─────────────────────
                    if _timeline_cards:
                        st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                            color:var(--straw);margin:8px 0 10px;text-transform:uppercase;">
                            🗓️ {'दिन-वार पूर्वानुमान विस्तार' if lang=='hi' else 'दिवसनिहाय तपशीलवार अंदाज' if lang=='mr' else 'Day-by-Day Detailed Forecast'}
                        </div>""", unsafe_allow_html=True)

                        for _chk_day, _chk_date, _chk_dis in _timeline_cards:
                            _days_from_now = _chk_day - _prev_crop_age
                            _top_row = _chk_dis.iloc[0]
                            _top_score = _top_row["risk_score"]

                            if _top_score >= 0.7: _fc_color, _fc_emoji = "#e74c3c", "🔴"
                            elif _top_score >= 0.4: _fc_color, _fc_emoji = "#f39c12", "🟡"
                            else: _fc_color, _fc_emoji = "#27ae60", "🟢"

                            _top_name = str(_top_row["disease"]).replace("_"," ").title()

                            # CNN model hint — if model trained, show top predicted disease for context
                            _cnn_hint = ""
                            _cnn_model_p = (
                                os.path.join(BASE_DIR, "best_disease_model.h5")
                                if os.path.exists(os.path.join(BASE_DIR, "best_disease_model.h5"))
                                else os.path.join(BASE_DIR, "onion_disease_model.h5")
                            )
                            if os.path.exists(_cnn_model_p) and os.path.exists(os.path.join(BASE_DIR, "class_indices.pkl")):
                                _cnn_hint = f'<span style="font-size:0.7rem;color:var(--sage);margin-left:8px;">🤖 CNN model loaded — scan a crop image on day {_chk_day} to confirm this prediction</span>'

                            with st.expander(
                                f"{_fc_emoji} Day {_chk_day}  ({_chk_date.strftime('%d %b')})  —  +{_days_from_now} days from now  |  {len(_chk_dis)} disease(s) active  |  Top risk: {_top_name} ({_top_score*100:.0f}%)"
                            ):
                                for _, _fr in _chk_dis.iterrows():
                                    _rs = _fr["risk_score"]
                                    if _rs >= 0.7: _rc2, _re2 = "#e74c3c", "🔴"
                                    elif _rs >= 0.4: _rc2, _re2 = "#f39c12", "🟡"
                                    else: _rc2, _re2 = "#27ae60", "🟢"
                                    _dn2 = str(_fr["disease"]).replace("_"," ").title()
                                    st.markdown(f"""
                                    <div style="background:{_rc2}12;border-left:4px solid {_rc2};
                                                padding:10px 14px;border-radius:6px;margin:6px 0;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
                                            <b style="color:{_rc2};font-family:'Cinzel',serif;font-size:0.85rem;">{_re2} {_dn2}</b>
                                            <span style="background:{_rc2};color:white;padding:2px 8px;border-radius:8px;font-size:11px;">
                                                {_rs*100:.0f}% risk
                                            </span>
                                        </div>
                                        <div style="font-size:0.75rem;color:var(--straw);margin:5px 0 3px;">
                                            📆 Window: Day {int(_fr['day_start'])}–{int(_fr['day_end'])} &nbsp;|&nbsp;
                                            💧 Humidity: {_fr['humidity_factor']} &nbsp;|&nbsp;
                                            🌡️ Temp: {_fr['temp_factor']} &nbsp;|&nbsp;
                                            🌧️ Rain: {_fr['rain_factor']}
                                        </div>
                                        <div style="font-size:0.84rem;color:var(--cream);">
                                            ✅ <b>{'उपाय' if lang in ['hi','mr'] else 'Action'}:</b> {_fr['preventive_advice']}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                if _cnn_hint:
                                    st.markdown(_cnn_hint, unsafe_allow_html=True)
                    else:
                        st.success(f"✅ {'अगले {_forecast_days} दिनों में कोई रोग खतरा नहीं!' if lang=='hi' else f'पुढील {_forecast_days} दिवसांत कोणतेही रोग नाही!' if lang=='mr' else f'No disease risks forecast in the next {_forecast_days} days!'}")

                    # ── Optional ML model result ───────────────────────
                    _ml_path = os.path.join(BASE_DIR, "onion_disease_risk_model.pkl")
                    if os.path.exists(_ml_path):
                        try:
                            with open(_ml_path, "rb") as _f:
                                _risk_ml = pickle.load(_f)
                            _feats = np.array([[_prev_crop_age, _prev_avg_temp, _prev_avg_humid, _prev_avg_rain]])
                            _ml_proba = _risk_ml.predict_proba(_feats)[0]
                            _ml_cls = _risk_ml.classes_[np.argmax(_ml_proba)]
                            st.markdown(f"""
                            <div style="margin-top:10px;padding:8px 14px;background:rgba(74,124,89,0.1);
                                        border:1px solid var(--leaf);border-radius:4px;font-size:0.8rem;">
                                🤖 <b>Risk ML Model:</b> <code>{_ml_cls}</code> ({max(_ml_proba)*100:.1f}% confidence)
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception:
                            pass

        # Contact info footer
        st.markdown("<hr>", unsafe_allow_html=True)
        st.info("""
**📞 Need Help? / मदद चाहिए? / मदत हवी आहे?**

Contact your local Agricultural Extension Officer · अपने स्थानीय कृषि विस्तार अधिकारी से संपर्क करें · स्थानिक कृषी विस्तार अधिकाऱ्यांशी संपर्क साधा

📞 Toll-Free: 1800-180-1551 (Kisan Call Centre)
        """)

        return

    # ════════════════════════════════════════════════════════════════════
    # ADMIN VIEW
    # ════════════════════════════════════════════════════════════════════

    # ── INPUT + SPEEDOMETER row ───────────────────────────────────────
    inp_col, speed_col = st.columns([1, 1])

    with inp_col:
        st.markdown(f"""<div class="panel"><div class="panel-title">⊞ {T['data_input']}</div>""", unsafe_allow_html=True)
        with st.form("admin_input"):
            i1, i2 = st.columns(2)
            with i1:
                # Get sensible defaults from last row of real data
                _last = datasets["monthly"].iloc[-1]
                _def_prod  = float(_last.get("production_mt", _last.get("production_annual", 11500)))
                _def_price = float(_last["price"])
                _def_seeds = float(_last.get("seeds_distributed_kg", _last.get("seeds_distributed", 2600)))
                prod   = st.number_input(T['current_prod'],  value=round(_def_prod,1),  step=100.0, format="%.1f")
                rain   = 820.0  # rainfall not in dataset; use climatological default silently
                o_price= st.number_input(T['onion_price'],    value=round(_def_price,2), step=1.0,   format="%.2f")
            with i2:
                seeds  = st.number_input(T['seeds_dist'],     value=round(_def_seeds,1), step=100.0, format="%.1f")
                g_price= st.number_input(T['garlic_price'],   value=round(float(datasets["garlic_price"]["price_per_quintal"].iloc[-1]),2), step=1.0, format="%.2f")
                p_price= st.number_input(T['paddy_price'],    value=round(float(datasets["paddy_price"]["price_per_quintal"].iloc[-1]),2),  step=1.0, format="%.2f")
            submitted = st.form_submit_button(f"⊡ {T['analyze']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with speed_col:
        if submitted:
            fv = build_feature_vector(monthly_df, current_year, current_month,
                                      o_price, prod, rain, seeds)
            proba = model.predict_proba(fv)[0][1]
            confidence = model.predict_proba(fv).max()

            if proba < 0.25:
                risk_label = T["low_risk"]; risk_class = "risk-low"
            elif proba < 0.50:
                risk_label = T["medium_risk"]; risk_class = "risk-medium"
            elif proba < 0.75:
                risk_label = T["high_risk"]; risk_class = "risk-high"
            else:
                risk_label = T["critical_risk"]; risk_class = "risk-critical"

            st.session_state['last_proba']     = proba
            st.session_state['last_prod']      = prod
            st.session_state['last_rain']      = rain
            st.session_state['last_seeds']     = seeds
            st.session_state['last_o_price']   = o_price
            st.session_state['last_g_price']   = g_price
            st.session_state['last_p_price']   = p_price
            st.session_state['last_risk_label']= risk_label
            st.session_state['last_risk_class']= risk_class
            st.session_state['last_confidence']= confidence

        # Always show speedometer (using session or default)
        p = st.session_state.get('last_proba', 0.15)
        rl = st.session_state.get('last_risk_label', T["low_risk"])
        rc = st.session_state.get('last_risk_class', "risk-low")
        conf = st.session_state.get('last_confidence', 0.9)

        st.plotly_chart(make_speedometer(p, T['crash_probability'], lang),
                        use_container_width=True, key="speedometer")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{T['risk_score']}</div>
                <div class="metric-value">{p*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{T['confidence']}</div>
                <div class="metric-value">{conf*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="risk-card {rc}" style="margin-top:10px;">
            <span style="font-family:'Cinzel',serif;font-size:0.85rem;">{rl}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── ANALOG YEARS + DIVERSIFICATION ──────────────────────────────
    ana_col, div_col = st.columns([1, 1])

    with ana_col:
        st.markdown(f"""<div class="panel"><div class="panel-title">⊛ {T['analog_years']}</div>""", unsafe_allow_html=True)

        curr_prod  = st.session_state.get('last_prod',    11500.0)
        curr_rain  = st.session_state.get('last_rain',    820.0)
        curr_seeds = st.session_state.get('last_seeds',   2600.0)
        curr_opr   = st.session_state.get('last_o_price', 1100.0)

        analogs = find_analog_years(monthly_df, curr_prod, 820, curr_seeds, curr_opr)
        for a in analogs:
            sim_w = int(a['similarity'])
            st.markdown(f"""
            <div class="analog-item">
                <span class="analog-year">{a['year']}</span>
                <div class="sim-bar-outer">
                    <div class="sim-bar-inner" style="width:{sim_w}%;"></div>
                </div>
                <span class="sim-pct">{a['similarity']}%</span>
                <span style="font-family:'Cinzel',serif;font-size:0.65rem;color:var(--straw);margin-left:8px;">{T['similarity']}</span>
            </div>
            """, unsafe_allow_html=True)

        # Show what happened in those analog years
        for a in analogs:
            yr = a['year']
            yr_data = monthly_df[monthly_df['year'] == yr]
            if not yr_data.empty:
                max_p = yr_data['price'].max()
                min_p = yr_data['price'].min()
                drop = ((max_p - min_p) / max_p) * 100
                color = "#E74C3C" if drop > 20 else "#F39C12" if drop > 10 else "#2ECC71"
                st.markdown(f"""
                <div style="font-size:0.75rem;color:var(--straw);padding:2px 14px;margin-bottom:4px;">
                    {yr}: ₹{min_p:.0f}–₹{max_p:.0f} |
                    <span style="color:{color};">▼{drop:.1f}% max drop</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with div_col:
        st.markdown(f"""<div class="panel"><div class="panel-title">⊞ {T['diversification']}</div>""", unsafe_allow_html=True)

        onion_pct  = st.slider(T['onion_pct'],  0, 100, 70, 5, key="sl_onion")
        garlic_pct = st.slider(T['garlic_pct'], 0, 100, 20, 5, key="sl_garlic")
        paddy_pct  = st.slider(T['paddy_pct'],  0, 100, 10, 5, key="sl_paddy")

        total = onion_pct + garlic_pct + paddy_pct
        if total != 100:
            st.warning(f"Total = {total}% (should = 100%)")

        # Revenue simulation
        area_ha = 2.0  # baseline farm
        YIELD_ONION = 15; YIELD_GARLIC = 8; YIELD_PADDY = 3  # tonnes/ha
        cp = st.session_state.get('last_o_price', 1100.0)
        gp_val = st.session_state.get('last_g_price', 2800.0)
        pp_val = st.session_state.get('last_p_price', 1400.0)

        rev_onion  = (onion_pct/100) * area_ha * YIELD_ONION * cp
        rev_garlic = (garlic_pct/100) * area_ha * YIELD_GARLIC * gp_val
        rev_paddy  = (paddy_pct/100) * area_ha * YIELD_PADDY * pp_val
        total_rev  = rev_onion + rev_garlic + rev_paddy

        # Risk reduction logic
        curr_p = st.session_state.get('last_proba', 0.15)
        prod_avg = monthly_df['production_mt'].mean()
        curr_prod_val = st.session_state.get('last_prod', 11500.0)

        if curr_prod_val > prod_avg * 1.1:  # overproduction → paddy helps more
            risk_red = (paddy_pct / 100) * 0.35 + (garlic_pct / 100) * 0.20
        else:  # underproduction/normal → garlic stabilizes more
            risk_red = (garlic_pct / 100) * 0.30 + (paddy_pct / 100) * 0.15

        new_risk = max(0, curr_p - risk_red)
        stabilization = risk_red * 100

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{T['revenue_estimate']}</div>
                <div class="metric-value" style="font-size:1.1rem;">₹{total_rev/1000:.1f}K</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{T['risk_reduction']}</div>
                <div class="metric-value" style="color:#2ECC71;">-{risk_red*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{T['stabilization']}</div>
                <div class="metric-value" style="font-size:1.1rem;">{stabilization:.0f}/100</div>
            </div>""", unsafe_allow_html=True)

        # Bar chart
        bar_fig = go.Figure(go.Bar(
            x=["Onion", "Garlic", "Paddy"],
            y=[rev_onion/1000, rev_garlic/1000, rev_paddy/1000],
            marker_color=["#8B7D4A", "#4A7C59", "#6B9E7A"],
            text=[f"₹{v/1:.1f}K" for v in [rev_onion/1000, rev_garlic/1000, rev_paddy/1000]],
            textposition="outside", textfont={"color": "#C4A55A", "family": "Georgia"},
        ))
        bar_fig.update_layout(
            height=160, margin=dict(l=0,r=0,t=10,b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=False, visible=False, color="#8B7355"),
            xaxis=dict(showgrid=False, color="#8B7355", tickfont=dict(color="#8B7355", family="Georgia", size=10)),
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True, key="div_bar")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TABS ────────────────────────────────────────────────────────
    tab1, tab3, tab4, tab_detect, tab_train = st.tabs([
        f"⟲ {T['crisis_rewind']}",
        f"◉ {T['market_analysis']}",
        f"◎ {T['news_feed']}",
        f"🧠 {'AI रोग पहचान' if lang=='hi' else 'AI रोग ओळख' if lang=='mr' else 'AI Disease Detection'}",
        f"🏋️ {'मॉडल प्रशिक्षण' if lang=='hi' else 'मॉडेल प्रशिक्षण' if lang=='mr' else 'Train Disease Model'}"
    ])

    # ── TAB 1: CRISIS REWIND ─────────────────────────────────────────
    with tab1:
        cr1, cr2 = st.columns([1, 2])
        with cr1:
            st.markdown(f"""<div class="panel"><div class="panel-title">{T['crisis_rewind']}</div>""", unsafe_allow_html=True)
            avail_years = sorted(monthly_df['year'].unique())
            sel_year = st.selectbox(T['select_year'], avail_years,
                                    index=len(avail_years)-5, key="rewind_yr")

            if st.button(f"⟲ {T['analyze']}", key="rewind_btn"):
                yr_data = monthly_df[monthly_df['year'] == sel_year].sort_values('month')
                proba_list = []
                actual_changes = []
                for _, row in yr_data.iterrows():
                    hist = monthly_df[
                        (monthly_df['year'] < sel_year) |
                        ((monthly_df['year'] == sel_year) & (monthly_df['month'] <= row['month']))
                    ].copy()
                    _pc  = "production_mt"       if "production_mt"       in hist.columns else "production_annual"
                    _sc2 = "seeds_distributed_kg" if "seeds_distributed_kg" in hist.columns else "seeds_distributed"
                    fv = build_feature_vector(hist, row['year'], row['month'],
                                             row['price'],
                                             row.get(_pc, row.get('production_annual', 10000)),
                                             row.get('rainfall_mm', 820),
                                             row.get(_sc2, row.get('seeds_distributed', 2500)))
                    proba_list.append(model.predict_proba(fv)[0][1])
                    actual_changes.append(row['price'])

                st.session_state['rewind_proba'] = proba_list
                st.session_state['rewind_prices'] = actual_changes
                st.session_state['rewind_year'] = sel_year

            st.markdown("</div>", unsafe_allow_html=True)

        with cr2:
            if 'rewind_proba' in st.session_state:
                rp = st.session_state['rewind_proba']
                prices = st.session_state['rewind_prices']
                ry = st.session_state['rewind_year']

                fig_cr = go.Figure()
                fig_cr.add_trace(go.Scatter(
                    x=list(range(1, 13)), y=[x*100 for x in rp],
                    mode='lines+markers', name=T['predicted'],
                    line=dict(color='#E8C875', width=2),
                    marker=dict(size=6, color='#E8C875'),
                ))
                prices_real = [p * 14.3 for p in prices]
                fig_cr.add_trace(go.Scatter(
                    x=list(range(1, 13)), y=prices_real,
                    mode='lines+markers', name=f"{T['actual']} Price (₹/Q)",
                    line=dict(color='#4A7C59', width=2, dash='dot'),
                    marker=dict(size=5, color='#4A7C59'),
                    yaxis='y2'
                ))
                fig_cr.update_layout(
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8B7355", size=10, family="Georgia")),
                    xaxis=dict(title="Month", color="#8B7355", gridcolor="rgba(139,115,74,0.1)",
                               tickfont=dict(color="#8B7355", family="Georgia")),
                    yaxis=dict(title="Crash Prob %", color="#E8C875", gridcolor="rgba(139,115,74,0.1)",
                               range=[0, 105], tickfont=dict(color="#8B7355", family="Georgia")),
                    yaxis2=dict(title="Price ₹", color="#4A7C59", overlaying='y', side='right',
                                tickfont=dict(color="#8B7355", family="Georgia")),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text=f"⟲ Crisis Rewind: {ry}", font=dict(color="#8B7355", size=11, family="Cinzel")),
                )
                st.plotly_chart(fig_cr, use_container_width=True, key="rewind_chart")

                # Predicted vs actual table
                max_pred_idx = rp.index(max(rp))
                min_price_idx = prices.index(min(prices))
                st.markdown(f"""
                <div style="display:flex;gap:10px;margin-top:8px;">
                    <div class="metric-box" style="flex:1;">
                        <div class="metric-label">Peak Risk Month</div>
                        <div class="metric-value" style="font-size:1.2rem;">M{max_pred_idx+1}</div>
                        <div class="metric-sub">{max(rp)*100:.1f}% probability</div>
                    </div>
                    <div class="metric-box" style="flex:1;">
                        <div class="metric-label">Actual Price Low</div>
                        <div class="metric-value" style="font-size:1.2rem;">₹{min(prices)*14.3:.0f}</div>
                        <div class="metric-sub">M{min_price_idx+1}</div>
                    </div>
                    <div class="metric-box" style="flex:1;">
                        <div class="metric-label">Model Alignment</div>
                        <div class="metric-value" style="font-size:1.2rem;color:#2ECC71;">{"✓" if abs(max_pred_idx - min_price_idx) <= 2 else "~"}</div>
                        <div class="metric-sub">±{abs(max_pred_idx - min_price_idx)} month</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: MARKET ANALYSIS ───────────────────────────────────────
    with tab3:
        # ── Section A: Historical price trend chart ────────────────
        st.markdown(f"""<div class="panel-title">◉ {T['price_trend']} — Historical</div>""",
                    unsafe_allow_html=True)

        op = datasets["onion_price"].copy()
        op['date'] = pd.to_datetime(op[['year','month']].assign(day=1))
        op = op.sort_values('date').tail(36)
        gp = datasets["garlic_price"].copy()
        gp['date'] = pd.to_datetime(gp[['year','month']].assign(day=1))
        gp = gp.sort_values('date').tail(36)
        pp = datasets["paddy_price"].copy()
        pp['date'] = pd.to_datetime(pp[['year','month']].assign(day=1))
        pp = pp.sort_values('date').tail(36)

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=op['date'], y=op['price_per_quintal'],
            mode='lines', name='Onion', line=dict(color='#E8C875', width=2)))
        fig_m.add_trace(go.Scatter(x=gp['date'], y=gp['price_per_quintal'],
            mode='lines', name='Garlic', line=dict(color='#4A7C59', width=2)))
        fig_m.add_trace(go.Scatter(x=pp['date'], y=pp['price_per_quintal'],
            mode='lines', name='Paddy', line=dict(color='#8B7D4A', width=2, dash='dot')))
        op_ma = op['price_per_quintal'].rolling(6, min_periods=1).mean()
        fig_m.add_trace(go.Scatter(x=op['date'], y=op_ma, mode='lines',
            name='Onion MA6', line=dict(color='rgba(232,200,117,0.4)', width=1, dash='dash')))
        fig_m.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8B7355", size=10, family="Georgia"),
                        orientation="h", y=-0.18),
            xaxis=dict(color="#8B7355", gridcolor="rgba(139,115,74,0.1)",
                       tickfont=dict(color="#8B7355", family="Georgia")),
            yaxis=dict(title="₹/Quintal", color="#8B7355", gridcolor="rgba(139,115,74,0.1)",
                       tickfont=dict(color="#8B7355", family="Georgia")),
            margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified',
        )
        st.plotly_chart(fig_m, use_container_width=True, key="market_chart")

        # Signal strip
        latest_o = op['price_per_quintal'].iloc[-1]
        ma6_o    = op['price_per_quintal'].tail(6).mean()
        ma12_o   = op['price_per_quintal'].tail(12).mean()
        dev6     = ((latest_o - ma6_o) / ma6_o) * 100
        dev12    = ((latest_o - ma12_o) / ma12_o) * 100
        sig6_c   = "#2ECC71" if dev6 > 0 else "#E74C3C"
        sig12_c  = "#2ECC71" if dev12 > 0 else "#E74C3C"
        sig_word = ("Caution" if dev6 < -10 else "Stable" if abs(dev6) < 5 else "Positive") if lang=="en" else                    ("सतर्क" if dev6 < -10 else "स्थिर" if abs(dev6) < 5 else "सकारात्मक") if lang=="hi" else                    ("सावधान" if dev6 < -10 else "स्थिर" if abs(dev6) < 5 else "सकारात्मक")

        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Latest Price</div>
                <div class="metric-value" style="font-size:1.3rem;">₹{latest_o:.1f}</div>
                <div class="metric-sub">per quintal</div></div>""", unsafe_allow_html=True)
        with sm2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">vs 6M Avg</div>
                <div class="metric-value" style="font-size:1.3rem;color:{sig6_c};">{"↑" if dev6>0 else "↓"}{abs(dev6):.1f}%</div>
                </div>""", unsafe_allow_html=True)
        with sm3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">vs 12M Avg</div>
                <div class="metric-value" style="font-size:1.3rem;color:{sig12_c};">{"↑" if dev12>0 else "↓"}{abs(dev12):.1f}%</div>
                </div>""", unsafe_allow_html=True)
        with sm4:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Signal</div>
                <div class="metric-value" style="font-size:1rem;color:{sig6_c};">{sig_word}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Section B: LIVE MANDI PRICES — sub-tabs (Local + Neighbouring States) ──
        st.markdown(f"""
        <div class="panel-title">⊛ {'लाइव्ह मंडी बाजार भाव' if lang=='mr' else 'लाइव मंडी बाजार भाव' if lang=='hi' else 'Live Mandi Market Prices'}
        <span style="font-size:0.65rem;color:var(--sage);margin-left:12px;">↻ {'दर 2 मिनिटांनी अपडेट' if lang=='mr' else 'हर 2 मिनट में अपडेट' if lang=='hi' else 'Updates every 2 min'}</span>
        </div>""", unsafe_allow_html=True)

        # Shared refresh state
        if 'mandi_last_refresh' not in st.session_state:
            st.session_state.mandi_last_refresh = datetime.now()
        time_diff = (datetime.now() - st.session_state.mandi_last_refresh).total_seconds()
        next_update = max(0, 120 - int(time_diff))

        base_px = float(op['price_per_quintal'].iloc[-1]) * 14.3

        adm_sub1, adm_sub2 = st.tabs([
            f"📍 {'स्थानिक बाजारपेठा' if lang=='mr' else 'स्थानीय बाजार' if lang=='hi' else 'Local Markets'}",
            f"🗺️ {'शेजारील राज्ये' if lang=='mr' else 'पड़ोसी राज्य' if lang=='hi' else 'Neighbouring States'}"
        ])

        # ── Sub-tab 1: Local Markets ───────────────────────────────────
        with adm_sub1:
            mb1, mb2 = st.columns([3, 1])
            with mb1:
                rank_choice = st.radio(
                    "Sort by:", ["💰 Highest Price", "📍 Nearest Mandi", "⚖️ Best Balance"],
                    horizontal=True, key="adm_mandi_rank", label_visibility="collapsed",
                )
            with mb2:
                st.caption(f"⏰ Next update: {next_update//60}:{next_update%60:02d}")
                if st.button("🔄 Refresh", key="adm_mandi_refresh"):
                    st.session_state.mandi_last_refresh = datetime.now()
                    st.rerun()

            mode = "price" if "Highest" in rank_choice else "distance" if "Nearest" in rank_choice else "balanced"
            df_mandi = _generate_mandi_prices(base_price=base_px)
            df_ranked = _rank_mandis(df_mandi, mode=mode)

            st.markdown(f"""
            <div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                        color:var(--straw);margin:8px 0 6px;text-transform:uppercase;">
                {'शीर्ष 5 शिफारशी' if lang=='mr' else 'शीर्ष 5 सिफारिशें' if lang=='hi' else 'Top 5 Recommendations'}
            </div>""", unsafe_allow_html=True)

            for i, (_, frow) in enumerate(df_ranked.head(5).iterrows()):
                fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 3])
                bg = "rgba(232,200,117,0.07)" if i == 0 else "rgba(255,255,255,0.02)"
                brd = "var(--wheat)" if i == 0 else "var(--bark)"
                with fc1:
                    st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;height:60px;">
                        <div style="font-family:'Cinzel',serif;color:var(--harvest);font-size:0.82rem;">#{frow['Rank']} {frow['Mandi']}</div>
                        <div style="font-size:0.7rem;color:var(--straw);">📍 {frow['Distance (km)']} km</div>
                    </div>""", unsafe_allow_html=True)
                with fc2:
                    st.metric("Price", f"₹{frow['Price (₹/Q)']}/Q")
                with fc3:
                    st.metric("Arrival", f"{frow['Arrival (T)']}T")
                with fc4:
                    st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;height:60px;">
                        <div style="font-size:0.8rem;color:var(--cream);">{frow['Rec']}</div>
                        <div style="font-size:0.68rem;color:var(--straw);">Grade A: {frow['Grade A (%)']}%</div>
                    </div>""", unsafe_allow_html=True)

            with st.expander(f"📋 {'सर्व मंडी पहा' if lang=='mr' else 'सभी मंडी देखें' if lang=='hi' else 'View All Mandis'}"):
                disp_cols = ['Rank','Mandi','Distance (km)','Price (₹/Q)','Arrival (T)','Grade A (%)','Rec']
                st.dataframe(df_ranked[disp_cols].rename(columns={"Rec":"Recommendation","Price (₹/Q)":"Price (₹/Quintal)"}),
                             use_container_width=True, hide_index=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            hi_p  = df_mandi['Price (₹/Q)'].max()
            hi_m  = df_mandi.loc[df_mandi['Price (₹/Q)'].idxmax(), 'Mandi']
            avg_p = df_mandi['Price (₹/Q)'].mean()
            near2 = df_mandi.nsmallest(2, 'Distance (km)').iloc[-1]
            _lbl1 = 'सर्वाधिक किंमत' if lang=='mr' else 'सर्वोच्च मूल्य' if lang=='hi' else 'Highest Price'
            _lbl2 = 'सरासरी किंमत'   if lang=='mr' else 'औसत मूल्य'      if lang=='hi' else 'Average Price'
            _lbl3 = 'जवळील चांगली किंमत' if lang=='mr' else 'निकटतम अच्छी कीमत' if lang=='hi' else 'Nearest Good Price'
            ins1, ins2, ins3 = st.columns(3)
            with ins1: st.info(f"**{_lbl1}:** {hi_m} — ₹{hi_p:.1f}/Q")
            with ins2: st.info(f"**{_lbl2}:** Across all mandis — ₹{avg_p:.1f}/Q")
            with ins3: st.info(f"**{_lbl3}:** {near2['Mandi']} — ₹{near2['Price (₹/Q)']:.1f}/Q")
            if "Highest" in rank_choice:
                tip = "✅ Check fuel cost vs price premium · साझा परिवहन से बचत करें · शेअर ट्रान्सपोर्ट वापरा"
            else:
                tip = "✅ Best price + distance balance · कम लागत = अधिक मुनाफा · कमी खर्च = जास्त नफा"
            st.markdown(f"""
            <div style="padding:10px 14px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);
                        border-radius:3px;font-size:0.82rem;color:var(--cream);margin-top:8px;">🚛 {tip}</div>
            """, unsafe_allow_html=True)

            if time_diff >= 120:
                st.session_state.mandi_last_refresh = datetime.now()
                st.rerun()

        # ── Sub-tab 2: Neighbouring States ────────────────────────────
        with adm_sub2:

            STATE_MANDIS_ADM = [
                ("Kalaburagi APMC",  "Karnataka",        380, 0.97),
                ("Gadag APMC",       "Karnataka",        450, 0.96),
                ("Hubli APMC",       "Karnataka",        410, 0.98),
                ("Belagavi APMC",    "Karnataka",        340, 0.95),
                ("Dharwad APMC",     "Karnataka",        420, 0.97),
                ("Nizamabad APMC",   "Telangana",        470, 0.99),
                ("Hyderabad APMC",   "Telangana",        560, 1.04),
                ("Warangal APMC",    "Telangana",        580, 1.01),
                ("Karimnagar APMC",  "Telangana",        540, 1.00),
                ("Guntur APMC",      "Andhra Pradesh",   620, 1.06),
                ("Kurnool APMC",     "Andhra Pradesh",   530, 1.02),
                ("Kadapa APMC",      "Andhra Pradesh",   650, 1.03),
                ("Anantapur APMC",   "Andhra Pradesh",   590, 1.01),
                ("Indore APMC",      "Madhya Pradesh",   510, 0.98),
                ("Bhopal APMC",      "Madhya Pradesh",   590, 0.96),
                ("Ujjain APMC",      "Madhya Pradesh",   480, 0.97),
                ("Mandsaur APMC",    "Madhya Pradesh",   430, 0.99),
                ("Ratlam APMC",      "Madhya Pradesh",   400, 0.98),
            ]

            def _generate_state_prices_adm(base_price=200.0):
                cur = datetime.now()
                mins = (cur.minute // 2) * 2
                ts   = cur.replace(minute=mins, second=0, microsecond=0)
                np.random.seed((int(ts.timestamp()) + 9999) % 100000)
                rows = []
                for name, state, dist, factor in STATE_MANDIS_ADM:
                    mbase   = base_price * factor
                    dv      = np.random.uniform(-0.06, 0.06)
                    h       = cur.hour
                    tf      = 1.03 if 6 <= h < 10 else 0.97 if 16 <= h < 20 else 1.0
                    sd      = np.random.uniform(-0.04, 0.04)
                    price   = max(50, mbase * (1 + dv) * tf * (1 + sd))
                    arrival = np.random.uniform(30, 90)
                    ga = np.random.uniform(38, 62); gb = np.random.uniform(25, 38); gc = 100 - ga - gb
                    rows.append({
                        "Mandi": name, "State": state, "Distance (km)": dist,
                        "Price (₹/Q)": round(price, 1), "Arrival (T)": round(arrival, 1),
                        "Grade A (%)": round(ga, 1), "Grade B (%)": round(gb, 1),
                        "Grade C (%)": round(gc, 1), "Updated": ts.strftime("%I:%M %p"),
                    })
                return pd.DataFrame(rows)

            STATE_COLORS_ADM = {
                "Karnataka": "#4E6B3A", "Telangana": "#3A5E6B",
                "Andhra Pradesh": "#6B4E3A", "Madhya Pradesh": "#5A4E6B",
            }

            sf1, sf2 = st.columns([2, 2])
            with sf1:
                st.markdown(f"""<div style="font-size:0.72rem;color:var(--straw);margin-bottom:4px;
                    font-family:'Cinzel',serif;letter-spacing:0.1em;">
                    {'राज्य फिल्टर' if lang in ['hi','mr'] else 'STATE FILTER'}</div>""", unsafe_allow_html=True)
                state_filter_adm = st.selectbox(
                    "State", ["All States", "Karnataka", "Telangana", "Andhra Pradesh", "Madhya Pradesh"],
                    label_visibility="collapsed", key="adm_state_mandi_filter"
                )
            with sf2:
                state_rank_adm = st.radio(
                    "Sort:", ["💰 Highest Price", "📍 Nearest", "⚖️ Best Balance"],
                    horizontal=True, key="adm_rank_state", label_visibility="collapsed"
                )

            df_st_adm = _generate_state_prices_adm(base_price=base_px)
            df_st_filt_adm = df_st_adm[df_st_adm["State"] == state_filter_adm].copy() if state_filter_adm != "All States" else df_st_adm.copy()
            st_mode_adm    = "price" if "Highest" in state_rank_adm else "distance" if "Nearest" in state_rank_adm else "balanced"
            df_st_r_adm    = _rank_mandis(df_st_filt_adm, st_mode_adm)

            st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                        color:var(--straw);margin:10px 0 6px;text-transform:uppercase;">
                        {'शीर्ष 5 · शिफारशी' if lang=='mr' else 'शीर्ष 5 · सिफारिशें' if lang=='hi' else 'Top 5 · Recommendations'}</div>""",
                        unsafe_allow_html=True)

            for i, (_, srow) in enumerate(df_st_r_adm.head(5).iterrows()):
                sc1, sc2, sc3, sc4 = st.columns([3, 2, 2, 3])
                bg  = "rgba(232,200,117,0.07)" if i == 0 else "rgba(255,255,255,0.02)"
                brd = "var(--wheat)"           if i == 0 else "var(--bark)"
                sc  = STATE_COLORS_ADM.get(srow['State'], '#555')
                np_s = _net_profit(srow['Price (₹/Q)'], srow['Distance (km)'])
                with sc1:
                    st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;min-height:68px;overflow:hidden;">
                        <div style="font-family:'Cinzel',serif;color:var(--harvest);font-size:0.82rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">#{srow['Rank']} {srow['Mandi']}</div>
                        <div style="font-size:0.68rem;color:var(--straw);">📍 {srow['Distance (km)']} km from Nashik</div>
                        <div style="display:inline-block;margin-top:3px;padding:1px 6px;background:{sc};border-radius:3px;font-size:0.58rem;color:#fff;letter-spacing:0.05em;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:middle;">
                            {srow['State'].upper()}</div>
                    </div>""", unsafe_allow_html=True)
                with sc2:
                    st.metric("Price / मूल्य", f"₹{srow['Price (₹/Q)']}/Q")
                with sc3:
                    st.metric("Arrival", f"{srow['Arrival (T)']}T")
                with sc4:
                    st.markdown(f"""<div style="padding:10px;background:{bg};border:1px solid {brd};border-radius:3px;min-height:68px;">
                        <div style="font-size:0.8rem;color:var(--cream);">{srow['Rec']}</div>
                        <div style="font-size:0.67rem;color:var(--straw);">Net/Q: ₹{np_s['per_quintal']:.0f} · Transport: ₹{np_s['transport']:.0f}</div>
                    </div>""", unsafe_allow_html=True)

            with st.expander(f"📋 {'सर्व राज्य मंडी' if lang=='mr' else 'सभी राज्य मंडी' if lang=='hi' else 'View All State Mandis'}"):
                s_disp = ['Rank','Mandi','State','Distance (km)','Price (₹/Q)','Arrival (T)','Grade A (%)','Rec']
                st.dataframe(df_st_r_adm[s_disp].rename(columns={"Rec":"Recommendation","Price (₹/Q)":"Price (₹/Quintal)"}),
                             use_container_width=True, hide_index=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;color:var(--straw);margin-bottom:8px;">
                ◆ {'राज्यनिहाय सरासरी भाव' if lang=='mr' else 'राज्यवार औसत मूल्य' if lang=='hi' else 'STATE-WISE AVERAGE PRICE'}</div>""",
                unsafe_allow_html=True)
            state_avg_adm = df_st_adm.groupby("State")["Price (₹/Q)"].mean().reset_index()
            state_avg_adm.columns = ["State", "Avg"]
            state_avg_adm = state_avg_adm.sort_values("Avg", ascending=False)
            fig_s_adm = go.Figure(go.Bar(
                x=state_avg_adm["State"], y=state_avg_adm["Avg"],
                marker_color=["#E8C875","#4A7C59","#C87E45","#5A8B6E"],
                text=[f"₹{v:.0f}" for v in state_avg_adm["Avg"]], textposition="outside"
            ))
            fig_s_adm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#D4C5A9", size=11),
                yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                margin=dict(t=30, b=10, l=10, r=10), height=250,
            )
            st.plotly_chart(fig_s_adm, use_container_width=True, key="adm_state_avg_chart")

            st.markdown(f"""
            <div style="padding:8px 12px;background:rgba(232,200,117,0.07);border:1px solid var(--wheat);border-radius:3px;font-size:0.78rem;color:var(--straw);">
                ⚠️ {'लांब अंतरासाठी वाहतूक खर्च लक्षात घ्या · FPO द्वारे एकत्र वाहतूक किफायतशीर' if lang=='mr'
                    else 'लंबी दूरी के लिए परिवहन लागत का ध्यान रखें · FPO के माध्यम से साझा परिवहन लाभदायक' if lang=='hi'
                    else 'Factor in long-distance transport cost · Shared FPO transport cuts per-quintal expense'}
            </div>""", unsafe_allow_html=True)

        # ── Export Institution Contacts (same as farmer interface) ────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;color:var(--straw);margin-bottom:6px;">
            ◆ 📞 {'निर्यात सहाय्यक संस्था — संपर्क' if lang=='mr' else 'निर्यात सहायक संस्थाएं — संपर्क' if lang=='hi' else 'EXPORT SUPPORT INSTITUTIONS — CONTACT'}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:0.82rem;color:var(--straw);margin-bottom:12px;">
            {'खालील संस्था शेतकऱ्यांना माल निर्यात करण्यात मदत करतात.' if lang=='mr'
             else 'नीचे दी गई संस्थाएं किसानों को उपज निर्यात करने में सहायता करती हैं।' if lang=='hi'
             else 'The following institutions assist farmers in exporting their produce.'}
        </div>""", unsafe_allow_html=True)

        EXPORT_CONTACTS_ADM = [
            {"name": "Karnataka State Agricultural Marketing Board (KSAMB)",
             "type": "State Govt. Body", "loc": "Bengaluru, Karnataka",
             "phone": "+91-080-2294-5700", "email": "ksamb@karnataka.gov.in",
             "desc": "Regulates APMCs, market fees and supports farmer-to-buyer linkages across Karnataka"},
            {"name": "Horticulture Dept. Karnataka (Kalaburagi APMC)",
             "type": "Govt. Export Authority", "loc": "Kalaburagi, Karnataka",
             "phone": "+91-08472-263-410", "email": "ddhort.klb@karnataka.gov.in",
             "desc": "Onion and vegetable grading, packaging and export certification in north Karnataka"},
            {"name": "Telangana State Agricultural Marketing Board (TSAMB)",
             "type": "State Govt. Body", "loc": "Hyderabad, Telangana",
             "phone": "+91-040-2329-1430", "email": "tsamb@telangana.gov.in",
             "desc": "Governs APMC markets; price intelligence and dissemination to farmers across Telangana"},
            {"name": "Nizamabad Vegetables and Fruits Export Hub",
             "type": "FPO / Exporter", "loc": "Nizamabad, Telangana",
             "phone": "+91-94904-55678", "email": "nzb.exporthub@gmail.com",
             "desc": "Bulk procurement and export of onions, tomatoes to Gulf and SE Asian markets"},
            {"name": "AP Agricultural Marketing Dept. (Guntur)",
             "type": "Govt. Export Authority", "loc": "Guntur, Andhra Pradesh",
             "phone": "+91-0863-223-5000", "email": "agmktg.gntr@ap.gov.in",
             "desc": "Key onion and chilli market authority; export documentation and buyer linkage services"},
            {"name": "ANGRAU Kurnool Agri Export Centre",
             "type": "Govt. Processing & Export", "loc": "Kurnool, Andhra Pradesh",
             "phone": "+91-08518-252-100", "email": "kurnool.export@angrau.ac.in",
             "desc": "Post-harvest technology, cold chain support and grading facility for onion exporters in AP"},
            {"name": "MP Mandi Board (Indore Region)",
             "type": "State Govt. Body", "loc": "Indore, Madhya Pradesh",
             "phone": "+91-0731-270-8022", "email": "ceo.mandiboard@mp.gov.in",
             "desc": "Manages 259 APMCs in MP; price reporting, auction management and export coordination"},
            {"name": "MP Agro Export Zone - Mandsaur Onion Hub",
             "type": "FPO / Exporter", "loc": "Mandsaur, Madhya Pradesh",
             "phone": "+91-07422-245-678", "email": "mpagro.mandsaur@gmail.com",
             "desc": "Direct export facilitation for rabi onion growers to Bangladesh, Nepal and Sri Lanka"},
        ]
        TYPE_COLORS_ADM = {
            "FPO / Exporter": "#4A6B3A", "Govt. Processing & Export": "#3A5B6B",
            "Govt. Export Authority": "#3A5B6B", "State Govt. Body": "#3A5B6B",
            "Trade Association": "#6B5A3A", "Financial Institution": "#5A3A6B",
        }
        for ci in range(0, len(EXPORT_CONTACTS_ADM), 2):
            ec1, ec2 = st.columns(2)
            for col, c in zip([ec1, ec2], EXPORT_CONTACTS_ADM[ci:ci+2]):
                tc = TYPE_COLORS_ADM.get(c['type'], '#444')
                with col:
                    st.markdown(f"""
                    <div style="padding:14px 16px;background:rgba(255,255,255,0.03);border:1px solid var(--bark);
                                border-radius:6px;margin-bottom:10px;">
                        <div style="font-family:'Cinzel',serif;font-size:0.88rem;color:var(--harvest);margin-bottom:6px;">
                            {c['name']}</div>
                        <div style="margin-bottom:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                            <span style="display:inline-block;padding:1px 8px;background:{tc};border-radius:3px;
                                         font-size:0.6rem;color:#fff;letter-spacing:0.08em;text-transform:uppercase;">
                                {c['type']}</span>
                            <span style="font-size:0.72rem;color:var(--straw);">📍 {c['loc']}</span>
                        </div>
                        <div style="font-size:0.78rem;color:var(--cream);margin-bottom:4px;">
                            📞 <b>{c['phone']}</b> &nbsp;&nbsp; ✉️ <span style="color:var(--sage);">{c['email']}</span>
                        </div>
                        <div style="font-size:0.72rem;color:var(--straw);font-style:italic;margin-top:4px;">
                            {c['desc']}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding:8px 12px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);border-radius:3px;
                    font-size:0.75rem;color:var(--straw);margin-top:4px;">
            ℹ️ {'वरील संपर्क माहिती उद्देशासाठी आहे. अधिकृत संकेतस्थळावर खात्री करा.' if lang=='mr'
                else 'उपरोक्त संपर्क जागरूकता उद्देश्यों के लिए है। आधिकारिक वेबसाइट से सत्यापित करें।' if lang=='hi'
                else 'Contact details are for informational purposes. Verify via official websites before reaching out.'}
        </div>""", unsafe_allow_html=True)


    # ── TAB 4: NEWS FEED ─────────────────────────────────────────────
    with tab4:
        def _nt(en_t, mr_t, hi_t):
            if lang == "mr": return mr_t
            if lang == "hi": return hi_t
            return en_t

        news_items = [
            {
                "title": _nt("Smart Irrigation Cuts Water Usage by 40% in Nashik Onion Farms",
                              "नाशिक कांदा शेतात स्मार्ट सिंचनाने 40% पाण्याची बचत",
                              "नासिक प्याज खेतों में स्मार्ट सिंचाई से 40% पानी की बचत"),
                "body": _nt("Drip irrigation combined with soil moisture sensors has reduced input costs significantly for farmers in the Nashik region, with pilot programs showing a 28% increase in yield efficiency.",
                             "नाशिक क्षेत्रातील शेतकऱ्यांसाठी ठिबक सिंचन आणि मातीच्या ओलावा सेन्सरच्या संयोगाने इनपुट खर्च लक्षणीयरीत्या कमी झाला आहे.",
                             "नासिक क्षेत्र के किसानों के लिए ड्रिप सिंचाई और मिट्टी की नमी सेंसर के संयोजन से इनपुट लागत में काफी कमी आई है।"),
                "tag": "IRRIGATION"
            },
            {
                "title": _nt("Cold Storage Expansion Reduces Post-Harvest Losses to 8%",
                              "शीतगृह विस्तारामुळे काढणीनंतरचे नुकसान 8% पर्यंत कमी",
                              "कोल्ड स्टोरेज विस्तार से कटाई के बाद नुकसान 8% तक कम"),
                "body": _nt("NABARD-funded cold chain infrastructure in Maharashtra has helped reduce post-harvest onion losses from 25% to under 10%, improving farmer incomes by stabilizing off-season supply.",
                             "महाराष्ट्रातील NABARD-वित्तपुरस्कृत कोल्ड चेन पायाभूत सुविधांमुळे काढणीनंतरचे कांद्याचे नुकसान 25% वरून 10% पेक्षा कमी झाले आहे.",
                             "महाराष्ट्र में NABARD द्वारा वित्त पोषित कोल्ड चेन बुनियादी ढांचे ने प्याज की कटाई के बाद के नुकसान को 25% से घटाकर 10% से कम कर दिया है।"),
                "tag": "STORAGE"
            },
            {
                "title": _nt("AI Crop Advisory Deployed Across 12 Districts",
                              "12 जिल्ह्यांमध्ये AI पीक सल्ला प्रणाली तैनात",
                              "12 जिलों में AI फसल सलाह प्रणाली तैनात"),
                "body": _nt("A government-backed AI platform now provides real-time price and weather advisories to over 400,000 farmers via mobile, integrating satellite data with local mandi price feeds.",
                             "सरकार-समर्थित AI प्लॅटफॉर्म आता मोबाइलद्वारे 4 लाखांहून अधिक शेतकऱ्यांना रीअल-टाइम किंमत आणि हवामान सल्ला देतो.",
                             "सरकार समर्थित AI प्लेटफॉर्म अब मोबाइल के माध्यम से 4 लाख से अधिक किसानों को रियल-टाइम मूल्य और मौसम सलाह प्रदान करता है।"),
                "tag": "AI · FARMING"
            },
            {
                "title": _nt("Climate-Resilient Onion Varieties Show 35% Better Yield in Drought",
                              "हवामान-प्रतिरोधक कांदा वाण दुष्काळात 35% अधिक उत्पादन देतो",
                              "जलवायु-प्रतिरोधी प्याज किस्में सूखे में 35% बेहतर उपज देती हैं"),
                "body": _nt("ICAR has released three drought-tolerant onion cultivars developed using genomic selection, with field trials across Rajasthan and Madhya Pradesh showing consistent performance gains.",
                             "ICAR ने जीनोमिक निवडीचा वापर करून विकसित केलेल्या तीन दुष्काळ-सहिष्णु कांदा वाण प्रसिद्ध केल्या आहेत.",
                             "ICAR ने जीनोमिक चयन का उपयोग करके विकसित तीन सूखा-सहिष्णु प्याज किस्में जारी की हैं।"),
                "tag": "CLIMATE RESILIENCE"
            },
            {
                "title": _nt("FPOs Enable Direct Market Access, Cutting Middlemen Margins",
                              "FPO मुळे थेट बाजार प्रवेश, मध्यस्थांचे मार्जिन कमी",
                              "FPO से सीधा बाजार पहुंच, बिचौलियों का मार्जिन कम"),
                "body": _nt("Farmer Producer Organizations in Nashik have collectively negotiated bulk contracts with exporters, enabling 30,000 farmers to receive 15-20% higher prices compared to APMC arrivals.",
                             "नाशिकमधील शेतकरी उत्पादक संघटनांनी सामूहिकपणे निर्यातदारांशी बल्क करार केले आहेत.",
                             "नासिक में किसान उत्पादक संगठनों ने सामूहिक रूप से निर्यातकों के साथ बल्क अनुबंध किए हैं।"),
                "tag": "MARKET ACCESS"
            },
        ]

        nc1, nc2 = st.columns(2)
        for i, item in enumerate(news_items):
            col = nc1 if i % 2 == 0 else nc2
            with col:
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-title">{item['title']}</div>
                    <div class="news-body">{item['body']}</div>
                    <span class="news-tag">{item['tag']}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── RECOMMENDATIONS ────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""<div class="panel-title">◈ {T['recommendations']}</div>""", unsafe_allow_html=True)

    curr_p = st.session_state.get('last_proba', 0.15)
    curr_prod_v = st.session_state.get('last_prod', 11500.0)
    prod_avg_r = monthly_df['production_mt'].mean()

    if curr_p < 0.25:
        rec_color = "#2ECC71"
        rec_icon = "✅"
        rec_text_mr = "बाजार स्थिर आहे. सामान्य कांदा उत्पादन सुरू ठेवा. साठवणूक क्षमता वाढवण्याचा विचार करा."
        rec_text_hi = "बाजार स्थिर है। सामान्य प्याज उत्पादन जारी रखें। भंडारण क्षमता बढ़ाने पर विचार करें।"
        rec_text_en = "Market is stable. Continue normal onion production. Consider expanding storage capacity."
    elif curr_p < 0.50:
        rec_color = "#F39C12"
        rec_icon = "⚡"
        rec_text_mr = "मध्यम जोखीम. 10-15% जमीन लसूण/भातासाठी वळवण्याचा विचार करा. बाजार करारांचा शोध घ्या."
        rec_text_hi = "मध्यम जोखिम। 10-15% भूमि लहसुन/धान के लिए मोड़ने पर विचार करें। बाजार अनुबंधों की तलाश करें।"
        rec_text_en = "Medium risk. Consider diverting 10-15% of land to garlic/paddy. Explore market contracts."
    elif curr_p < 0.75:
        rec_color = "#E67E22"
        rec_icon = "⚠️"
        rec_text_mr = "उच्च जोखीम! 20-25% विविधीकरण तातडीने करा. FPO मार्फत थेट विक्री करार करा. साठवणूक वाढवा."
        rec_text_hi = "उच्च जोखिम! तत्काल 20-25% विविधीकरण करें। FPO के माध्यम से सीधे बिक्री अनुबंध करें।"
        rec_text_en = "High risk! Diversify 20-25% immediately. Arrange direct sale contracts via FPO. Increase storage."
    else:
        rec_color = "#E74C3C"
        rec_icon = "🚨"
        rec_text_mr = "गंभीर जोखीम! किमान 30% विविधीकरण अनिवार्य. सरकारी MSP योजनांशी संपर्क करा. निर्यात पर्याय शोधा."
        rec_text_hi = "गंभीर जोखिम! कम से कम 30% विविधीकरण अनिवार्य। सरकारी MSP योजनाओं से संपर्क करें। निर्यात विकल्प खोजें।"
        rec_text_en = "Critical risk! Minimum 30% diversification mandatory. Contact govt MSP schemes. Explore export options."

    rec_text = rec_text_mr if lang == "mr" else rec_text_hi if lang == "hi" else rec_text_en

    st.markdown(f"""
    <div style="padding:16px 20px;background:rgba(255,255,255,0.02);border:1px solid {rec_color};
                border-left:4px solid {rec_color};border-radius:4px;">
        <span style="color:{rec_color};font-size:1.1rem;">{rec_icon}</span>
        <span style="color:var(--cream);font-size:0.92rem;margin-left:10px;">{rec_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-line">
        AGRICULTURAL RISK INTELLIGENCE CONSOLE · RANDOM FOREST ML · {datetime.now().strftime('%d %b %Y %H:%M')} IST
    </div>
    """, unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════
    # ADMIN TAB: 🧠 AI DISEASE DETECTION (ADDITIVE — NEW)
    # ═══════════════════════════════════════════════════════════════════
    with tab_detect:
        st.markdown(f"""<div class="panel-title">🧠 {'AI रोग पहचान — प्याज फसल' if lang=='hi' else 'AI रोग ओळख — कांदा पीक' if lang=='mr' else 'AI Disease Detection — Onion Crop'}</div>""",
                    unsafe_allow_html=True)

        if not _PIL_AVAILABLE:
            st.error("❌ Pillow not installed. Run: `pip install Pillow`")
        elif not _TF_AVAILABLE:
            st.error("❌ TensorFlow not installed. Run: `pip install tensorflow`")
        else:
            # Accept either name — best_disease_model.h5 (from train script) or onion_disease_model.h5
            _det_model_path = (
                os.path.join(BASE_DIR, "best_disease_model.h5")
                if os.path.exists(os.path.join(BASE_DIR, "best_disease_model.h5"))
                else os.path.join(BASE_DIR, "onion_disease_model.h5")
            )
            _det_model = _load_cnn_disease_model(_det_model_path)

            if _det_model is None:
                st.warning(f"""⚠️ {'मॉडल फ़ाइल नहीं मिली। पहले "मॉडल प्रशिक्षण" टैब में मॉडल ट्रेन करें।' if lang=='hi'
                              else 'मॉडेल फाइल सापडली नाही. "मॉडेल प्रशिक्षण" टॅबमधून प्रशिक्षण द्या.' if lang=='mr'
                              else 'Model file not found. Train it first using the 🏋️ Train Disease Model tab.'}""")
            else:
                st.markdown(f"""
                <div style="padding:10px 14px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);
                            border-radius:4px;font-size:0.84rem;color:var(--cream);margin-bottom:14px;">
                    {'छवि अपलोड करें या कैमरा का उपयोग करें — AI रोग का पता लगाएगा।' if lang=='hi'
                     else 'प्रतिमा अपलोड करा किंवा कॅमेरा वापरा — AI रोग शोधेल.' if lang=='mr'
                     else 'Upload a crop image or use your camera — the AI will detect the disease and advise action.'}
                </div>
                """, unsafe_allow_html=True)

                _det_c1, _det_c2 = st.columns(2)
                with _det_c1:
                    _uploaded = st.file_uploader(
                        f"📁 {'छवि अपलोड करें' if lang=='hi' else 'प्रतिमा अपलोड करा' if lang=='mr' else 'Upload Crop Image'}",
                        type=["jpg", "jpeg", "png"], key="det_upload_img"
                    )
                with _det_c2:
                    _camera = st.camera_input(
                        f"📷 {'कैमरा उपयोग करें' if lang=='hi' else 'कॅमेरा वापरा' if lang=='mr' else 'Use Camera'}",
                        key="det_camera_img"
                    )

                # Camera input takes priority
                _img_src = _camera if _camera is not None else _uploaded

                if _img_src is not None:
                    _pil_img = _PILImage.open(_img_src)
                    st.image(_pil_img,
                             caption=f"{'विश्लेषण हो रहा है...' if lang=='hi' else 'विश्लेषण सुरू आहे...' if lang=='mr' else 'Analysing this image…'}",
                             use_column_width=True)

                    with st.spinner(f"{'🔍 रोग पहचान चल रहा है...' if lang=='hi' else '🔍 रोग ओळख सुरू आहे...' if lang=='mr' else '🔍 Running disease detection…'}"):
                        _inp = _preprocess_disease_image(_pil_img)
                        _preds = _det_model.predict(_inp)[0]
                        _pred_idx = int(np.argmax(_preds))
                        _conf = float(_preds[_pred_idx]) * 100

                        # Map prediction index to class name
                        # Try to load class indices if saved during training
                        _class_idx_path = os.path.join(BASE_DIR, "class_indices.pkl")
                        if os.path.exists(_class_idx_path):
                            with open(_class_idx_path, "rb") as _cif:
                                _saved_idx = pickle.load(_cif)
                            # Invert: {class_name: idx} → {idx: class_name}
                            _idx_to_class = {v: k for k, v in _saved_idx.items()}
                            _disease_name = _idx_to_class.get(_pred_idx, _DISEASE_CLASSES[_pred_idx] if _pred_idx < len(_DISEASE_CLASSES) else "unknown")
                        else:
                            _disease_name = _DISEASE_CLASSES[_pred_idx] if _pred_idx < len(_DISEASE_CLASSES) else "unknown"

                        _advice = _DISEASE_ADVICE.get(_disease_name, "Consult an agronomist.")

                    _det_color = "#2ecc71" if _disease_name == "healthy" else "#e74c3c"
                    st.markdown(f"""
                    <div style="background:{_det_color}22;border-left:5px solid {_det_color};
                                padding:18px;border-radius:8px;margin-top:12px;">
                        <h4 style="color:{_det_color};margin:0 0 8px;font-family:'Cinzel',serif;">
                            🔬 {'पहचाना गया रोग' if lang=='hi' else 'आढळलेला रोग' if lang=='mr' else 'Detected Disease'}:
                            {_disease_name.replace('_',' ').title()}
                        </h4>
                        <p style="font-size:1rem;margin:4px 0;">
                            <b>{'विश्वास' if lang in ['hi','mr'] else 'Confidence'}:</b>
                            <span style="color:{_det_color};font-family:'Cinzel',serif;font-size:1.1rem;">
                                {_conf:.1f}%
                            </span>
                        </p>
                        <hr style="border-color:{_det_color}44;margin:10px 0;">
                        <p style="font-size:0.92rem;margin:0;">
                            <b>{'सलाह' if lang=='hi' else 'सल्ला' if lang=='mr' else 'Advisory'}:</b>
                            {_advice}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence bar chart for all classes
                    st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                        color:var(--straw);margin:16px 0 6px;text-transform:uppercase;">
                        📊 {'पूर्ण भविष्यवाणी विश्वास' if lang=='hi' else 'संपूर्ण अंदाज विश्वास' if lang=='mr' else 'Full Prediction Confidence'}
                    </div>""", unsafe_allow_html=True)

                    # Build label list — use class_indices if available, else defaults
                    if os.path.exists(_class_idx_path):
                        with open(_class_idx_path, "rb") as _cif2:
                            _si2 = pickle.load(_cif2)
                        _idx2class = {v: k for k, v in _si2.items()}
                        _bar_labels = [_idx2class.get(i, f"class_{i}").replace("_", " ").title()
                                       for i in range(len(_preds))]
                    else:
                        _bar_labels = [c.replace("_", " ").title() for c in _DISEASE_CLASSES[:len(_preds)]]

                    _conf_df = pd.DataFrame({
                        "Disease": _bar_labels,
                        "Confidence (%)": (_preds * 100).tolist()
                    }).sort_values("Confidence (%)", ascending=False)
                    st.bar_chart(_conf_df.set_index("Disease"))
                else:
                    st.markdown(f"""
                    <div style="text-align:center;padding:36px;border:1px dashed var(--bark);border-radius:8px;color:var(--straw);">
                        <div style="font-size:2.5rem;margin-bottom:10px;">🔬</div>
                        <div style="font-family:'Cinzel',serif;font-size:0.85rem;color:var(--harvest);">
                            {'छवि अपलोड करें या फोटो लें' if lang=='hi' else 'प्रतिमा अपलोड करा किंवा फोटो काढा' if lang=='mr' else 'Upload an image or take a photo to begin detection'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # ADMIN TAB: 🏋️ TRAIN DISEASE MODEL (ADDITIVE — NEW)
    # ═══════════════════════════════════════════════════════════════════
    with tab_train:
        st.markdown(f"""<div class="panel-title">🏋️ {'रोग मॉडल प्रशिक्षण' if lang=='hi' else 'रोग मॉडेल प्रशिक्षण' if lang=='mr' else 'Train Onion Disease Detection Model'}</div>""",
                    unsafe_allow_html=True)

        if not _TF_AVAILABLE:
            st.error("❌ TensorFlow not installed. Run: `pip install tensorflow`")
        else:
            st.markdown(f"""
            <div style="padding:10px 14px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);
                        border-radius:4px;font-size:0.84rem;color:var(--cream);margin-bottom:14px;">
                {'MobileNetV2 CNN को आपके स्थानीय डेटासेट पर प्रशिक्षित करें। मॉडल BASE_DIR में सहेजा जाएगा।' if lang=='hi'
                 else 'MobileNetV2 CNN तुमच्या स्थानिक डेटासेटवर प्रशिक्षित करा. मॉडेल BASE_DIR मध्ये सेव्ह होईल.' if lang=='mr'
                 else 'Train a MobileNetV2 CNN on your local dataset folder. Saved as onion_disease_model.h5 in BASE_DIR.'}
            </div>
            """, unsafe_allow_html=True)

            _train_ds_root = os.path.join(BASE_DIR, "onion datasets")
            _train_model_save = os.path.join(BASE_DIR, "onion_disease_model.h5")

            # Dataset check
            st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                color:var(--straw);margin:8px 0 6px;text-transform:uppercase;">
                📁 {'डेटासेट फ़ोल्डर जांच' if lang=='hi' else 'डेटासेट फोल्डर तपासणी' if lang=='mr' else 'Dataset Folder Check'}
            </div>""", unsafe_allow_html=True)

            if os.path.isdir(_train_ds_root):
                _found_cls = [d for d in os.listdir(_train_ds_root)
                              if os.path.isdir(os.path.join(_train_ds_root, d))]
                st.success(f"✅ Dataset folder found — {len(_found_cls)} class folders detected")
                with st.expander(f"{'फ़ोल्डर देखें' if lang=='hi' else 'फोल्डर पहा' if lang=='mr' else 'View detected folders'}"):
                    st.write(", ".join(_found_cls))
            else:
                st.error(f"❌ Dataset folder not found at: `{_train_ds_root}`\n\nPlace your **onion datasets** folder inside BASE_DIR.")
                st.stop()

            # Training parameters
            st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                color:var(--straw);margin:12px 0 6px;text-transform:uppercase;">
                ⚙️ {'प्रशिक्षण विन्यास' if lang=='hi' else 'प्रशिक्षण कॉन्फिगरेशन' if lang=='mr' else 'Training Configuration'}
            </div>""", unsafe_allow_html=True)

            _tr_c1, _tr_c2, _tr_c3 = st.columns(3)
            with _tr_c1:
                _tr_epochs = st.slider(f"{'युग' if lang in ['hi','mr'] else 'Epochs'}", 5, 50, 10, key="tr_epochs")
            with _tr_c2:
                _tr_batch = st.select_slider(f"{'बैच साइज़' if lang=='hi' else 'बॅच साईज' if lang=='mr' else 'Batch Size'}",
                                             options=[8, 16, 32, 64], value=16, key="tr_batch")
            with _tr_c3:
                _tr_val = st.slider(f"{'वैलिडेशन %' if lang in ['hi','mr'] else 'Validation Split %'}",
                                    10, 30, 20, key="tr_val_split") / 100

            _tr_aug = st.checkbox(f"{'डेटा संवर्धन सक्षम करें (अनुशंसित)' if lang=='hi' else 'डेटा ऑगमेंटेशन सक्षम करा (शिफारस)' if lang=='mr' else 'Enable Data Augmentation (recommended)'}",
                                  value=True, key="tr_augment")
            _tr_finetune = st.checkbox(f"{'MobileNetV2 बेस परतें फाइन-ट्यून करें (धीमा, अधिक सटीक)' if lang=='hi' else 'MobileNetV2 बेस लेयर फाइन-ट्यून करा (हळू, अधिक अचूक)' if lang=='mr' else 'Fine-tune MobileNetV2 base layers (slower, more accurate)'}",
                                       value=False, key="tr_finetune")

            if "tr_log" not in st.session_state:
                st.session_state.tr_log = []
            if "tr_running" not in st.session_state:
                st.session_state.tr_running = False
            if "tr_done" not in st.session_state:
                st.session_state.tr_done = False

            _log_placeholder = st.empty()

            def _render_train_log():
                _log_placeholder.code("\n".join(st.session_state.tr_log[-35:]) or "Waiting to start…", language="")

            _render_train_log()

            def _run_training():
                _log = st.session_state.tr_log
                _log.clear()
                _log.append("🚀 Starting training pipeline…")
                try:
                    if _tr_aug:
                        _gen = _ImageDataGenerator(
                            rescale=1./255, validation_split=_tr_val,
                            rotation_range=20, width_shift_range=0.2,
                            height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2
                        )
                    else:
                        _gen = _ImageDataGenerator(rescale=1./255, validation_split=_tr_val)

                    _tr_ds = _gen.flow_from_directory(
                        _train_ds_root, target_size=(224, 224), batch_size=_tr_batch,
                        class_mode="categorical", subset="training", shuffle=True
                    )
                    _val_ds = _gen.flow_from_directory(
                        _train_ds_root, target_size=(224, 224), batch_size=_tr_batch,
                        class_mode="categorical", subset="validation"
                    )

                    _n_cls = _tr_ds.num_classes
                    _log.append(f"✅ {_tr_ds.samples} training images across {_n_cls} classes.")
                    _log.append(f"✅ {_val_ds.samples} validation images.")
                    _log.append(f"   Classes: {list(_tr_ds.class_indices.keys())}")
                    _log.append("🔨 Building MobileNetV2 model…")

                    _base_m = tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
                    )
                    _base_m.trainable = _tr_finetune

                    _cnn = _keras_models.Sequential([
                        _base_m,
                        _keras_layers.GlobalAveragePooling2D(),
                        _keras_layers.Dropout(0.3),
                        _keras_layers.Dense(256, activation="relu"),
                        _keras_layers.Dropout(0.2),
                        _keras_layers.Dense(_n_cls, activation="softmax")
                    ])
                    _cnn.compile(
                        optimizer=tf.keras.optimizers.Adam(1e-4),
                        loss="categorical_crossentropy",
                        metrics=["accuracy"]
                    )
                    _log.append(f"✅ Model compiled. Parameters: {_cnn.count_params():,}")
                    _log.append(f"📅 Training {_tr_epochs} epochs, batch={_tr_batch}")
                    _log.append("─" * 50)

                    class _EpochLogger(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            logs = logs or {}
                            st.session_state.tr_log.append(
                                f"Epoch {epoch+1:02d}/{_tr_epochs}  "
                                f"loss={logs.get('loss', 0):.4f}  "
                                f"acc={logs.get('accuracy', 0):.4f}  "
                                f"val_loss={logs.get('val_loss', 0):.4f}  "
                                f"val_acc={logs.get('val_accuracy', 0):.4f}"
                            )

                    _cnn.fit(
                        _tr_ds, epochs=_tr_epochs, validation_data=_val_ds,
                        callbacks=[
                            _EpochLogger(),
                            tf.keras.callbacks.EarlyStopping(
                                monitor="val_accuracy", patience=5,
                                restore_best_weights=True, verbose=0
                            ),
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor="val_loss", factor=0.5, patience=3, verbose=0
                            )
                        ],
                        verbose=0
                    )
                    _cnn.save(_train_model_save)
                    _log.append("─" * 50)
                    _log.append(f"🎉 Training complete! Model saved:")
                    _log.append(f"   {_train_model_save}")

                    # Save class index mapping
                    _ci_save = os.path.join(BASE_DIR, "class_indices.pkl")
                    with open(_ci_save, "wb") as _pf:
                        pickle.dump(_tr_ds.class_indices, _pf)
                    _log.append(f"📄 Class index map saved: {_ci_save}")

                    # Clear cached model so detection tab reloads fresh
                    _load_cnn_disease_model.clear()

                except Exception as _ex:
                    st.session_state.tr_log.append(f"❌ Error: {_ex}")

                st.session_state.tr_running = False
                st.session_state.tr_done = True

            _btn_c1, _btn_c2 = st.columns(2)
            with _btn_c1:
                _start_btn = st.button(
                    f"▶️ {'प्रशिक्षण शुरू करें' if lang=='hi' else 'प्रशिक्षण सुरू करा' if lang=='mr' else 'Start Training'}",
                    disabled=st.session_state.tr_running, key="btn_start_training"
                )
            with _btn_c2:
                if os.path.exists(_train_model_save):
                    _sz = os.path.getsize(_train_model_save) // 1024
                    st.success(f"✅ Existing model: {_sz} KB")
                else:
                    st.info("ℹ️ No trained model yet.")

            if _start_btn and not st.session_state.tr_running:
                st.session_state.tr_running = True
                st.session_state.tr_done = False
                st.session_state.tr_log = []
                threading.Thread(target=_run_training, daemon=True).start()
                st.rerun()

            if st.session_state.tr_running:
                _render_train_log()
                st.warning(f"{'⏳ प्रशिक्षण जारी है... अपडेट देखने के लिए रिफ्रेश करें।' if lang=='hi' else '⏳ प्रशिक्षण सुरू आहे... अपडेट पाहण्यासाठी रिफ्रेश करा.' if lang=='mr' else '⏳ Training in progress… Refresh to see updates.'}")
                if st.button(f"🔄 {'लॉग रिफ्रेश करें' if lang=='hi' else 'लॉग रिफ्रेश करा' if lang=='mr' else 'Refresh Log'}",
                             key="btn_refresh_log"):
                    st.rerun()

            if st.session_state.tr_done:
                _render_train_log()
                if os.path.exists(_train_model_save):
                    st.success(f"✅ {'मॉडल सफलतापूर्वक प्रशिक्षित! AI रोग पहचान टैब पर जाएं।' if lang=='hi' else 'मॉडेल यशस्वीपणे प्रशिक्षित! AI रोग ओळख टॅबवर जा.' if lang=='mr' else 'Model trained and saved! Switch to 🧠 AI Disease Detection tab to use it.'}")
                st.session_state.tr_done = False


if __name__ == "__main__":
    main()

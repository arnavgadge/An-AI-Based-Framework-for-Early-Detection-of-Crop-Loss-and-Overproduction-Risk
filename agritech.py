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

import feedparser
from streamlit_autorefresh import st_autorefresh

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
        "disease_prevention": "रोग प्रतिबंध",
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
        "disease_prevention": "रोग रोकथाम",
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
        "disease_prevention": "Disease Prevention",
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
# LIVE AGRICULTURAL NEWS — GOOGLE RSS
# ═══════════════════════════════════════════════════════════════════════════
AGRI_KEYWORDS = [
    "onion", "agriculture", "farmer", "mandi", "crop",
    "rainfall", "monsoon", "export", "MSP", "kisan",
    "fertilizer", "harvest", "production", "horticulture",
    "kanda", "pyaj", "shetkari"
]

# Tag labels per language
_NEWS_TAGS = {
    "price":   {"en": "💰 PRICE",   "mr": "💰 किंमत",    "hi": "💰 मूल्य"},
    "weather": {"en": "🌧 WEATHER", "mr": "🌧 हवामान",   "hi": "🌧 मौसम"},
    "policy":  {"en": "🏛 POLICY",  "mr": "🏛 धोरण",     "hi": "🏛 नीति"},
    "onion":   {"en": "🧅 ONION",   "mr": "🧅 कांदा",    "hi": "🧅 प्याज"},
    "agri":    {"en": "🌾 AGRI",    "mr": "🌾 शेती",     "hi": "🌾 कृषि"},
}

def _news_tag_label(title_lower: str, lang: str) -> str:
    if "price" in title_lower or "मूल्य" in title_lower or "किंमत" in title_lower:
        key = "price"
    elif "rain" in title_lower or "monsoon" in title_lower or "पाऊस" in title_lower or "बारिश" in title_lower:
        key = "weather"
    elif "export" in title_lower or "policy" in title_lower or "ban" in title_lower or "MSP" in title_lower:
        key = "policy"
    elif "onion" in title_lower or "कांदा" in title_lower or "प्याज" in title_lower:
        key = "onion"
    else:
        key = "agri"
    return _NEWS_TAGS[key].get(lang, _NEWS_TAGS[key]["en"])

@st.cache_data(ttl=120)
def get_agri_news(lang: str = "en"):
    """Fetch live agriculture news from Google RSS. Falls back to empty list on error.
    Uses lang-aware RSS feed (en/hi/mr) and tags results in the correct language."""
    # Google News RSS endpoints per language
    _feeds = {
        "en": "https://news.google.com/rss/search?q=agriculture+onion+india+farmer&hl=en-IN&gl=IN&ceid=IN:en",
        "hi": "https://news.google.com/rss/search?q=कृषि+किसान+प्याज+भारत&hl=hi-IN&gl=IN&ceid=IN:hi",
        "mr": "https://news.google.com/rss/search?q=शेती+शेतकरी+कांदा+भारत&hl=mr-IN&gl=IN&ceid=IN:mr",
    }
    url = _feeds.get(lang, _feeds["en"])
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            tl = title.lower()
            sl = summary.lower()
            score = sum(kw in tl or kw in sl for kw in AGRI_KEYWORDS)
            if score >= 1:
                articles.append({
                    "title": title,
                    "body": summary,
                    "link": entry.get("link", "#"),
                    "published": entry.get("published", ""),
                    "tag": _news_tag_label(tl, lang),
                })
            if len(articles) >= 8:
                break
        return articles
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════════════
# DISEASE DETECTION — CONSTANTS & LOADERS (ADDITIVE)
# ═══════════════════════════════════════════════════════════════════════════
_DISEASE_CLASSES = [
    "alternaria", "botrytis_blight", "bulb_rot", "downy_mildew",
    "fusarium", "healthy", "iris_yellow_virus", "purple_blotch",
    "rust", "stemphylium_blight", "xanthomonas_blight", "caterpillar"
]

_DISEASE_ADVICE = {
    "healthy": {
        "severity": "NONE", "pathogen": "No disease detected — crop appears healthy.",
        "immediate": ["Continue weekly monitoring — scout full field","Maintain current irrigation & fertilizer schedule","Check for early pest or disease signs every 5–7 days","Keep drainage clear","Document crop status with photos"],
        "chemical": "No treatment needed. PREVENTIVE: Mancozeb 75WP @ 2.5g/L once before peak humid season.",
        "biological": "Trichoderma viride @ 4g/L soil drench monthly. Bacillus subtilis @ 2ml/L foliar once a month.",
        "cultural": "Maintain 15×10 cm spacing. Remove weeds. Crop rotation — no alliums in same field for 2+ years.",
        "fertilizer": "Balanced NPK. Potassium @ 60kg K2O/ha. Foliar 0.5% Boron + 0.2% Zinc every 3 weeks.",
        "irrigation": "Drip preferred. Irrigate early morning. Avoid waterlogging.",
        "recovery": "N/A — crop is healthy.",
        "prevention": "Plant resistant varieties. Scout from day 30. Remove all crop debris post-harvest.",
        "organic": "Cow urine 5% spray every 14 days. Dashaparni Ark 3% monthly spray.",
        "cost": "Rs.200-400/acre (preventive only)",
    
        "pathogen_hi": "कोई रोग नहीं — फसल स्वस्थ दिखती है।",
        "pathogen_mr": "कोणताही रोग आढळला नाही — पीक निरोगी दिसते.",
        "immediate_hi": ["साप्ताहिक निगरानी जारी रखें — पूरे खेत का सर्वेक्षण करें", "वर्तमान सिंचाई और उर्वरक कार्यक्रम बनाए रखें", "हर 5-7 दिनों में कीट या रोग के शुरुआती संकेत जांचें", "जल निकासी साफ रखें", "फ़ोटो से फसल की स्थिति दर्ज करें"],
        "immediate_mr": ["साप्ताहिक निरीक्षण सुरू ठेवा — संपूर्ण शेताचे सर्वेक्षण करा", "सध्याचे सिंचन व खत वेळापत्रक कायम ठेवा", "दर 5-7 दिवसांनी कीड किंवा रोगाची सुरुवातीची चिन्हे तपासा", "निचरा स्वच्छ ठेवा", "फोटोद्वारे पिकाची सद्यस्थिती नोंदवा"],
        "chemical_hi": "कोई उपचार आवश्यक नहीं। निवारक: आर्द्र मौसम से पहले Mancozeb 75WP @ 2.5g/L एक बार।",
        "chemical_mr": "कोणताही उपचार आवश्यक नाही. प्रतिबंधक: दमट हवामानापूर्वी Mancozeb 75WP @ 2.5g/L एकदा.",
        "biological_hi": "Trichoderma viride @ 4g/L मिट्टी में महीने में एक बार। Bacillus subtilis @ 2ml/L पत्तियों पर महीने में एक बार।",
        "biological_mr": "Trichoderma viride @ 4g/L जमिनीत महिन्यातून एकदा. Bacillus subtilis @ 2ml/L पानांवर महिन्यातून एकदा.",
        "cultural_hi": "15×10 सेमी पौधों की दूरी बनाए रखें। नियमित खरपतवार हटाएं। फसल चक्र — एक ही खेत में 2+ वर्ष तक कोई प्याज न लगाएं।",
        "cultural_mr": "15×10 सेमी अंतर ठेवा. नियमितपणे तण काढा. पीक फेरपालट — त्याच शेतात 2+ वर्षे कांदा नाही.",
        "fertilizer_hi": "संतुलित NPK। पोटेशियम @ 60kg K2O/हेक्टेयर। हर 3 सप्ताह में 0.5% बोरॉन + 0.2% जिंक का पत्तेदार छिड़काव।",
        "fertilizer_mr": "संतुलित NPK. पोटॅशियम @ 60kg K2O/हेक्टर. दर 3 आठवड्यांनी 0.5% बोरॉन + 0.2% झिंक पर्णीय फवारणी.",
        "irrigation_hi": "ड्रिप सिंचाई बेहतर। सुबह सिंचाई करें। जलभराव से बचें।",
        "irrigation_mr": "ठिबक सिंचन उत्तम. सकाळी पाणी द्या. साठलेले पाणी टाळा.",
        "recovery_hi": "लागू नहीं — फसल स्वस्थ है।",
        "recovery_mr": "लागू नाही — पीक निरोगी आहे.",
        "prevention_hi": "प्रतिरोधी किस्में लगाएं। दिन 30 से निगरानी शुरू करें। कटाई के बाद खेत में सारा मलबा हटाएं।",
        "prevention_mr": "प्रतिरोधक वाण लावा. दिन 30 पासून निरीक्षण सुरू करा. काढणीनंतर शेतातील सर्व अवशेष काढा.",
        "organic_hi": "गोमूत्र 5% का छिड़काव हर 14 दिनों में। दशपर्णी अर्क 3% मासिक।",
        "organic_mr": "गोमूत्र 5% फवारणी दर 14 दिवसांनी. दशपर्णी अर्क 3% मासिक.",
        "cost_hi": "₹200-400/एकड़ (केवल निवारक)",
        "cost_mr": "₹200-400/एकर (फक्त प्रतिबंधक)",
    },
    "alternaria": {
        "severity": "HIGH", "pathogen": "Alternaria porri (fungus) — spreads via wind & water splash; favoured by 25-30C humid conditions; enters through stomata & wounds.",
        "immediate": ["Remove & destroy ALL infected leaves within 24h — do NOT compost","Switch from overhead sprinklers to drip IMMEDIATELY","Thin dense areas to improve airflow","Stop nitrogen fertilizer during active infection — worsens spread","Disinfect tools with 10% bleach after each use"],
        "chemical": "PRIMARY: Iprodione 50WP @ 2g/L OR Mancozeb 75WP @ 2.5g/L | ALTERNATE: Chlorothalonil 75WP @ 2g/L | INTERVAL: Every 7-10 days; 3-4 sprays/season | TIMING: Early morning or late evening only | ROTATE fungicides each spray to prevent resistance",
        "biological": "Trichoderma viride @ 4g/L soil drench | Bacillus subtilis @ 2ml/L foliar spray | Neem oil 3% every 14 days | Pseudomonas fluorescens @ 2.5kg/ha soil application",
        "cultural": "3-year crop rotation without alliums | Seed treatment: Thiram 75WP @ 3g/kg | Remove all crop debris post-harvest | Spacing: 15x10 cm minimum for airflow",
        "fertilizer": "REDUCE Nitrogen — excess N worsens disease | Potassium @ 60kg K2O/ha boosts immunity | Foliar 0.5% Boron + 0.2% Zinc after rain | No urea top-dress during humid spells",
        "irrigation": "Drip only — keep foliage dry at all times | Irrigate early morning so leaves dry by noon | Avoid irrigation 2 days before forecast rain | Ensure field drainage; avoid waterlogging",
        "recovery": "14-21 days for new healthy growth after treatment starts.",
        "prevention": "Resistant varieties: NHRDF Red, Agrifound Dark Red | Preventive Mancozeb at onset of humid weather | Monitor weekly from day 30 of crop age",
        "organic": "Dashaparni Ark 3% spray every 10 days | Cow urine 5% every 10 days | Garlic extract 5% spray every 14 days",
        "cost": "Rs.800-1,200/acre per spray cycle",
        "pathogen_hi": "Alternaria porri (कवक) — हवा और पानी के छींटों से फैलता है; 25-30°C नमी में पनपता है; रंध्रों और घावों से प्रवेश करता है।",
        "pathogen_mr": "Alternaria porri (बुरशी) — वारा आणि पाण्याच्या शिडकाव्याने पसरते; 25-30°C दमट हवामानात वाढते; छिद्रे आणि जखमांद्वारे प्रवेश करते.",
        "immediate_hi": ["24 घंटे में सभी संक्रमित पत्तियां हटाएं और नष्ट करें — खाद न बनाएं", "ओवरहेड सिंचाई तुरंत बंद करें और ड्रिप पर जाएं", "घने क्षेत्रों को पतला करें ताकि हवा चले", "संक्रमण के दौरान नाइट्रोजन उर्वरक न दें — रोग बढ़ता है", "संक्रमित पौधों के पास उपयोग किए गए उपकरणों को 10% ब्लीच से साफ करें"],
        "immediate_mr": ["24 तासांत सर्व बाधित पाने काढा व नष्ट करा — खत करू नका", "ओव्हरहेड सिंचन ताबडतोब बंद करा व ठिबकवर जा", "दाट भागात विरळणी करा जेणेकरून वायुवीजन होईल", "संसर्गादरम्यान नत्र खत देऊ नका — रोग वाढतो", "बाधित झाडांजवळ वापरलेली अवजारे 10% ब्लीचने स्वच्छ करा"],
        "chemical_hi": "प्राथमिक: Iprodione 50WP @ 2g/L या Mancozeb 75WP @ 2.5g/L | वैकल्पिक: Chlorothalonil 75WP @ 2g/L | अंतराल: 7-10 दिनों में; मौसम में 3-4 छिड़काव | समय: केवल सुबह या शाम | प्रतिरोध रोकने के लिए हर छिड़काव में कवकनाशी बदलें",
        "chemical_mr": "प्राथमिक: Iprodione 50WP @ 2g/L किंवा Mancozeb 75WP @ 2.5g/L | पर्याय: Chlorothalonil 75WP @ 2g/L | अंतर: 7-10 दिवसांनी; हंगामात 3-4 फवारण्या | वेळ: फक्त सकाळी किंवा संध्याकाळी | प्रत्येक फवारणीत बुरशीनाशक बदला",
        "biological_hi": "Trichoderma viride @ 4g/L मिट्टी में | Bacillus subtilis @ 2ml/L पत्तियों पर | नीम तेल 3% हर 14 दिन | Pseudomonas fluorescens @ 2.5kg/हेक्टेयर मिट्टी में",
        "biological_mr": "Trichoderma viride @ 4g/L जमिनीत | Bacillus subtilis @ 2ml/L पानांवर | निंबाचे तेल 3% दर 14 दिवसांनी | Pseudomonas fluorescens @ 2.5kg/हेक्टर जमिनीत",
        "cultural_hi": "3 साल का फसल चक्र बिना प्याज के | बीज उपचार: Thiram 75WP @ 3g/kg | कटाई के बाद सारा मलबा हटाएं | दूरी: 15x10 सेमी न्यूनतम",
        "cultural_mr": "3 वर्षांचे पीक फेरपालट कांद्याशिवाय | बीज प्रक्रिया: Thiram 75WP @ 3g/kg | काढणीनंतर सर्व अवशेष काढा | अंतर: किमान 15x10 सेमी",
        "fertilizer_hi": "नाइट्रोजन कम करें | पोटेशियम @ 60kg K2O/हेक्टेयर | बारिश के बाद 0.5% बोरॉन + 0.2% जिंक का पत्तेदार छिड़काव | नमी में यूरिया टॉप-ड्रेस न करें",
        "fertilizer_mr": "नत्र कमी करा | पोटॅशियम @ 60kg K2O/हेक्टर | पावसानंतर 0.5% बोरॉन + 0.2% झिंक पर्णीय फवारणी | दमट हवामानात युरिया टॉप-ड्रेस देऊ नका",
        "irrigation_hi": "ड्रिप सिंचाई — पत्तियां हमेशा सूखी रखें | सुबह सिंचाई करें | पूर्वानुमानित बारिश से 2 दिन पहले सिंचाई न करें | जल निकासी बनाए रखें",
        "irrigation_mr": "ठिबक सिंचन — पाने नेहमी कोरडी ठेवा | सकाळी पाणी द्या | अपेक्षित पावसापूर्वी 2 दिवस सिंचन नाही | शेताचा निचरा राखा",
        "recovery_hi": "उपचार शुरू होने के 14-21 दिन बाद नई स्वस्थ वृद्धि।",
        "recovery_mr": "उपचार सुरू झाल्यानंतर 14-21 दिवसांत नवीन निरोगी वाढ.",
        "prevention_hi": "प्रतिरोधी किस्में: NHRDF Red, Agrifound Dark Red | आर्द्र मौसम शुरू होने पर निवारक Mancozeb | दिन 30 से साप्ताहिक निगरानी",
        "prevention_mr": "प्रतिरोधक वाण: NHRDF Red, Agrifound Dark Red | दमट हवामान सुरू होताच प्रतिबंधक Mancozeb | दिन 30 पासून साप्ताहिक निरीक्षण",
        "organic_hi": "दशपर्णी अर्क 3% हर 10 दिन | गोमूत्र 5% हर 10 दिन | लहसुन अर्क 5% हर 14 दिन",
        "organic_mr": "दशपर्णी अर्क 3% दर 10 दिवसांनी | गोमूत्र 5% दर 10 दिवसांनी | लसूण अर्क 5% दर 14 दिवसांनी",
        "cost_hi": "₹800-1,200/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹800-1,200/एकर प्रति फवारणी चक्र",
    },
    "botrytis_blight": {
        "severity": "HIGH", "pathogen": "Botrytis squamosa (fungus) — grey fuzzy lesions on leaf tips; thrives in cool humid overcast conditions (15-20C); spreads rapidly via airborne spores.",
        "immediate": ["Cut & remove ALL blighted leaf tips (grey fuzzy lesions) within 24h","Burn or deep-bury removed material — do NOT compost","Stop ALL overhead sprinkler irrigation at once — critical","Increase row spacing to reduce canopy humidity","Improve ventilation — remove dense foliage obstructions"],
        "chemical": "PRIMARY: Iprodione 50WP @ 2g/L (most effective against Botrytis) | ALTERNATE: Boscalid + Pyraclostrobin @ 0.75ml/L | EMERGENCY: Thiabendazole @ 1g/L | INTERVAL: Every 7 days in cool-humid conditions | ROTATE modes of action each spray",
        "biological": "Bacillus subtilis (Serenade) @ 3ml/L | Trichoderma harzianum @ 5g/L soil drench | Coniothyrium minitans for sclerotia suppression | Ampelomyces quisqualis biocontrol spray",
        "cultural": "Increase row spacing to 20cm | No late-evening irrigation | Install windbreaks to reduce humidity pockets | Remove dead leaves from field every week",
        "fertilizer": "Calcium nitrate @ 200kg/ha — strengthens cell walls | Potassium sulfate @ 100kg/ha | Foliar Calcium 0.5% after each rain | Reduce Nitrogen below 80kg N/ha for season",
        "irrigation": "Drip ONLY — critical for this disease | Irrigate every 3-4 days in small amounts | Never irrigate before overcast/foggy days | Maintain 60-70% field capacity",
        "recovery": "10-18 days. Cool weather slows recovery — expect 2-3 weeks in cold conditions.",
        "prevention": "Seed treatment: Thiram + Carbendazim @ 2g/kg | Preventive spray at 40 days crop age | Monitor for white water-soaked leaf tip spots — earliest sign",
        "organic": "Buttermilk (thin) 10% spray every 7 days — proven antifungal | Turmeric + water 2% spray | Neem cake soil incorporation @ 250kg/ha",
        "cost": "Rs.1,000-1,500/acre per spray cycle",
        "pathogen_hi": "Botrytis squamosa (कवक) — पत्ती की नोक पर भूरे रंग के धब्बे; 15-20°C ठंडी नमी में पनपता है; हवाई बीजाणुओं से तेज़ी से फैलता है।",
        "pathogen_mr": "Botrytis squamosa (बुरशी) — पानांच्या टोकावर राखाडी ठिपके; 15-20°C थंड दमट हवामानात वाढते; हवेतील बीजाणूंनी वेगाने पसरते.",
        "immediate_hi": ["24 घंटे में सभी झुलसी पत्तियां काटें और हटाएं", "जलाएं या गहरा दफनाएं — खाद न बनाएं", "ओवरहेड सिंचाई तुरंत बंद करें", "पंक्तियों के बीच दूरी बढ़ाएं", "घनी पत्तियां हटाकर वेंटिलेशन सुधारें"],
        "immediate_mr": ["24 तासांत सर्व करपलेल्या पाने काढा", "जाळा किंवा खोल गाडा — खत करू नका", "ओव्हरहेड सिंचन ताबडतोब बंद करा", "ओळींमधील अंतर वाढवा", "दाट पाने काढून वायुवीजन सुधारा"],
        "chemical_hi": "प्राथमिक: Iprodione 50WP @ 2g/L | वैकल्पिक: Boscalid + Pyraclostrobin @ 0.75ml/L | आपातकाल: Thiabendazole @ 1g/L | अंतराल: ठंडी नमी में हर 7 दिन | हर छिड़काव में कवकनाशी का तरीका बदलें",
        "chemical_mr": "प्राथमिक: Iprodione 50WP @ 2g/L | पर्याय: Boscalid + Pyraclostrobin @ 0.75ml/L | आणीबाणी: Thiabendazole @ 1g/L | अंतर: थंड दमट हवामानात दर 7 दिवस | प्रत्येक फवारणीत पद्धत बदला",
        "biological_hi": "Bacillus subtilis (Serenade) @ 3ml/L | Trichoderma harzianum @ 5g/L मिट्टी में | Coniothyrium minitans स्क्लेरोटिया दमन के लिए | Ampelomyces quisqualis जैव नियंत्रण",
        "biological_mr": "Bacillus subtilis (Serenade) @ 3ml/L | Trichoderma harzianum @ 5g/L जमिनीत | Coniothyrium minitans स्क्लेरोशिया दमनासाठी | Ampelomyces quisqualis जैव नियंत्रण",
        "cultural_hi": "पंक्तियों की दूरी 20 सेमी करें | शाम को सिंचाई न करें | वायु अवरोध लगाएं | हर हफ्ते मृत पत्तियां खेत से हटाएं",
        "cultural_mr": "ओळींचे अंतर 20 सेमी करा | संध्याकाळी सिंचन नाही | वायूरोधक लावा | दर आठवड्याला मृत पाने शेतातून काढा",
        "fertilizer_hi": "कैल्शियम नाइट्रेट @ 200kg/हेक्टेयर | पोटेशियम सल्फेट @ 100kg/हेक्टेयर | बारिश के बाद फोलियर कैल्शियम 0.5% | नाइट्रोजन 80kg N/हेक्टेयर से कम रखें",
        "fertilizer_mr": "कॅल्शियम नायट्रेट @ 200kg/हेक्टर | पोटॅशियम सल्फेट @ 100kg/हेक्टर | पावसानंतर पर्णीय कॅल्शियम 0.5% | नत्र 80kg N/हेक्टर पेक्षा कमी ठेवा",
        "irrigation_hi": "केवल ड्रिप — बहुत जरूरी | हर 3-4 दिन थोड़ी मात्रा में | बादल/कोहरे वाले दिन से पहले सिंचाई न करें | 60-70% फील्ड क्षमता बनाए रखें",
        "irrigation_mr": "फक्त ठिबक — अत्यंत महत्त्वाचे | दर 3-4 दिवसांनी थोडे पाणी | ढगाळ/धुक्याच्या दिवसापूर्वी सिंचन नाही | 60-70% क्षेत्र क्षमता राखा",
        "recovery_hi": "10-18 दिन। ठंड में रिकवरी धीमी — 2-3 सप्ताह लग सकते हैं।",
        "recovery_mr": "10-18 दिवस. थंडीत बरे होण्यास उशीर — 2-3 आठवडे लागू शकतात.",
        "prevention_hi": "बीज उपचार: Thiram + Carbendazim @ 2g/kg | 40 दिन की उम्र में निवारक छिड़काव | पत्ती की नोक पर सफेद पानी जैसे धब्बे — पहला संकेत",
        "prevention_mr": "बीज प्रक्रिया: Thiram + Carbendazim @ 2g/kg | 40 दिवसांच्या वयात प्रतिबंधक फवारणी | पानांच्या टोकावर पांढरे पाणीदार डाग — पहले लक्षण",
        "organic_hi": "छाछ (पतला) 10% छिड़काव हर 7 दिन | हल्दी + पानी 2% | नीम केक मिट्टी में @ 250kg/हेक्टेयर",
        "organic_mr": "ताक (पातळ) 10% फवारणी दर 7 दिवसांनी | हळद + पाणी 2% | निंबोळी पेंड जमिनीत @ 250kg/हेक्टर",
        "cost_hi": "₹1,000-1,500/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹1,000-1,500/एकर प्रति फवारणी चक्र",
    },
    "bulb_rot": {
        "severity": "CRITICAL", "pathogen": "Fusarium oxysporum (fungus) + Erwinia carotovora (bacterium). Fungal: brown internal rot. Bacterial: foul smell, slimy tissue. Both spread fast through wet soil.",
        "immediate": ["STOP all irrigation IMMEDIATELY — moisture spreads rot to neighbours","Pull & remove ALL rotting bulbs — they infect neighbours within 24h","Remove mulch — let soil dry in sunlight","Mark affected zones — do NOT move soil to clean areas","Drench healthy plants with Carbendazim 50WP @ 1g/L at base"],
        "chemical": "SOIL DRENCH: Carbendazim 50WP @ 1g/L at plant base | FOLIAR: Propiconazole 25EC @ 1ml/L | BACTERIAL: Copper oxychloride 50WP @ 3g/L | PRE-PLANT: Formalin 0.4% soil drench if replanting | REPEAT: Every 10 days until no new infections",
        "biological": "Trichoderma viride 4g/L soil drench — best preventive | Pseudomonas fluorescens 2.5kg/ha soil | Bacillus subtilis root dip before transplanting | VAM mycorrhizae @ 5kg/ha at transplanting",
        "cultural": "4-year rotation without alliums | Raised beds for drainage | Cure bulbs 7-10 days at 30-35C before storage | Harvest ONLY on dry days",
        "fertilizer": "Phosphorus @ 80kg P2O5/ha for root health | Zinc sulfate 25kg/ha soil | No excess Nitrogen during bulbing | Potassium @ 120kg K2O/ha — critical for rot resistance",
        "irrigation": "Drip ONLY — furrow irrigation banned | Stop irrigation 15 days before harvest | Do not irrigate at field capacity | Unclog drainage outlets weekly",
        "recovery": "Rotting bulbs CANNOT recover. Spread containment takes 7-10 days with aggressive management.",
        "prevention": "Certified disease-free seed | Seed treatment: Bavistin 2g/kg + Thiram 2g/kg | Raised bed cultivation | Inspect bulbs weekly from day 60",
        "organic": "Wood ash @ 500kg/ha soil — suppresses Fusarium | Neem cake @ 400kg/ha pre-planting | Ginger extract 5% soil drench preventively",
        "cost": "Rs.1,500-2,500/acre (soil treatment required)",
        "pathogen_hi": "Fusarium oxysporum (कवक) + Erwinia carotovora (जीवाणु)। कवक: अंदरूनी भूरा सड़न। जीवाणु: बदबू, गीली पत्ती। दोनों गीली मिट्टी में तेज़ी से फैलते हैं।",
        "pathogen_mr": "Fusarium oxysporum (बुरशी) + Erwinia carotovora (जीवाणू). बुरशी: आतील तपकिरी कूज. जीवाणू: दुर्गंधी, चिकट ऊतक. दोन्ही ओल्या जमिनीत वेगाने पसरतात.",
        "immediate_hi": ["सिंचाई तुरंत बंद करें — नमी पड़ोसी पौधों तक सड़न फैलाती है", "सभी सड़े बल्ब हटाएं — वे 24 घंटे में पड़ोसियों को संक्रमित करते हैं", "मल्च हटाएं — मिट्टी को धूप में सुखाएं", "प्रभावित क्षेत्र चिन्हित करें — मिट्टी को स्वस्थ क्षेत्रों में न ले जाएं", "स्वस्थ पौधों के आधार पर Carbendazim 50WP @ 1g/L का छिड़काव करें"],
        "immediate_mr": ["सिंचन ताबडतोब बंद करा — ओलावा शेजारच्या झाडांना कूज पसरवतो", "सर्व कुजलेले कांदे काढा — ते 24 तासांत शेजाऱ्यांना संसर्ग करतात", "मल्च काढा — जमीन उन्हात वाळवा", "बाधित क्षेत्र खुणवा — माती निरोगी भागात नेऊ नका", "निरोगी झाडांच्या मुळाशी Carbendazim 50WP @ 1g/L ओतणी करा"],
        "chemical_hi": "मिट्टी ड्रेंच: Carbendazim 50WP @ 1g/L | पत्तेदार: Propiconazole 25EC @ 1ml/L | जीवाणु: Copper oxychloride 50WP @ 3g/L | पूर्व-रोपण: Formalin 0.4% मिट्टी ड्रेंच | हर 10 दिन दोहराएं",
        "chemical_mr": "मातीत ओतणी: Carbendazim 50WP @ 1g/L | पर्णीय: Propiconazole 25EC @ 1ml/L | जीवाणू: Copper oxychloride 50WP @ 3g/L | लावणीपूर्व: Formalin 0.4% माती ओतणी | दर 10 दिवसांनी पुन्हा करा",
        "biological_hi": "Trichoderma viride 4g/L मिट्टी ड्रेंच | Pseudomonas fluorescens 2.5kg/हेक्टेयर मिट्टी में | Bacillus subtilis जड़ में डुबोएं | VAM माइकोराइजा @ 5kg/हेक्टेयर",
        "biological_mr": "Trichoderma viride 4g/L जमीन ओतणी | Pseudomonas fluorescens 2.5kg/हेक्टर जमिनीत | Bacillus subtilis मुळात बुडवा | VAM मायकोरायझा @ 5kg/हेक्टर",
        "cultural_hi": "4 साल का फसल चक्र बिना प्याज के | जल निकासी के लिए उठी क्यारियां | भंडारण से पहले 7-10 दिन 30-35°C पर बल्ब ठीक करें | केवल सूखे दिन कटाई करें",
        "cultural_mr": "4 वर्षांचे पीक फेरपालट कांद्याशिवाय | निचऱ्यासाठी उंच वाफे | साठवणुकीपूर्वी 7-10 दिवस 30-35°C वर कांदा वाळवा | फक्त कोरड्या दिवशी काढणी करा",
        "fertilizer_hi": "फास्फोरस @ 80kg P2O5/हेक्टेयर | जिंक सल्फेट 25kg/हेक्टेयर | बल्बिंग के दौरान नाइट्रोजन से बचें | पोटेशियम @ 120kg K2O/हेक्टेयर",
        "fertilizer_mr": "स्फुरद @ 80kg P2O5/हेक्टर | झिंक सल्फेट 25kg/हेक्टर | कांदा फुगण्याच्या काळात नत्र टाळा | पोटॅशियम @ 120kg K2O/हेक्टर",
        "irrigation_hi": "केवल ड्रिप — कुंड सिंचाई प्रतिबंधित | कटाई से 15 दिन पहले सिंचाई बंद | जल निकास के रास्ते साफ रखें",
        "irrigation_mr": "फक्त ठिबक — सरी सिंचन बंदी | काढणीपूर्वी 15 दिवस सिंचन बंद | निचऱ्याचे मार्ग साफ ठेवा",
        "recovery_hi": "सड़े बल्ब ठीक नहीं होते। 7-10 दिन में प्रसार नियंत्रित हो सकता है।",
        "recovery_mr": "कुजलेले कांदे बरे होत नाहीत. 7-10 दिवसांत प्रसार नियंत्रित होऊ शकतो.",
        "prevention_hi": "प्रमाणित रोग-मुक्त बीज | बीज उपचार: Bavistin 2g/kg + Thiram 2g/kg | उठी क्यारियों में खेती | दिन 60 से बल्ब की साप्ताहिक जांच",
        "prevention_mr": "प्रमाणित रोगमुक्त बियाणे | बीज प्रक्रिया: Bavistin 2g/kg + Thiram 2g/kg | उंच वाफ्यांवर लागवड | दिन 60 पासून कांद्याची साप्ताहिक तपासणी",
        "organic_hi": "लकड़ी की राख @ 500kg/हेक्टेयर — Fusarium दमन | नीम केक @ 400kg/हेक्टेयर रोपाई से पहले | अदरक अर्क 5% मिट्टी ड्रेंच",
        "organic_mr": "लाकडाची राख @ 500kg/हेक्टर — Fusarium दमन | निंबोळी पेंड @ 400kg/हेक्टर लावणीपूर्वी | आले अर्क 5% माती ओतणी",
        "cost_hi": "₹1,500-2,500/एकड़ (मिट्टी उपचार आवश्यक)",
        "cost_mr": "₹1,500-2,500/एकर (माती उपचार आवश्यक)",
    },
    "bulb_blight": {
        "severity": "MEDIUM", "pathogen": "Aspergillus niger (black powder mold) / Penicillium spp. (blue-green mold) — storage fungi; spreads in warm humid storage; enters via wounds and neck.",
        "immediate": ["Separate blighted bulbs from healthy stock IMMEDIATELY","Move healthy bulbs to well-ventilated dry shade","Do NOT wet stored bulbs — moisture causes rapid spread","Discard bulbs with black powder (Aspergillus) — unrecoverable","Fumigate storage room with sulphur dioxide pads"],
        "chemical": "PRE-STORAGE DIP: Thiabendazole 450g/L @ 2ml/L for 5 min | FIELD SPRAY: Captan 50WP @ 2.5g/L before harvest | ALTERNATE: Iprodione + Thiophanate-methyl @ 1.5g/L | STORAGE: Sulphur dioxide fumigation pads",
        "biological": "Trichoderma asperellum coating on bulb surface | Bacillus amyloliquefaciens bio-fungicide dip | Neem oil 2% + baking soda 0.5% dip before storage",
        "cultural": "Cure bulbs 7-10 days at 35C with airflow | Store at 0-4C / 65-70% RH | Use slatted wooden crates — NOT polythene bags | Inspect every 2 weeks; remove suspect bulbs",
        "fertilizer": "Pre-harvest Potassium @ 40kg K2O/ha | Foliar Calcium 0.3% two weeks before harvest | Reduce irrigation + fertilizer 3 weeks before harvest | Sulfur @ 20kg/ha toughens outer skin",
        "irrigation": "Stop ALL irrigation 3 weeks before harvest — critical | Avoid rain exposure of harvested bulbs | Use shade nets during field curing",
        "recovery": "Field: 7-14 days. Storage affected bulbs cannot be cured — remove immediately.",
        "prevention": "Harvest at correct maturity (tops 50-75% fallen) | Never store wounded or bruised bulbs | Keep storage below 5C | Fumigate storage room before each season",
        "organic": "Dry sand layers between bulb rows to absorb moisture | Turmeric powder dusting on bulb surface | Beeswax coating on neck after curing",
        "cost": "Rs.400-800/acre (storage management)",
        "pathogen_hi": "Aspergillus niger (काला पाउडर) / Penicillium spp. (नीला-हरा फफूंद) — भंडारण कवक; नम भंडारण में तेज़ी से फैलता है; घाव और गर्दन से प्रवेश।",
        "pathogen_mr": "Aspergillus niger (काळी पावडर) / Penicillium spp. (निळा-हिरवा बुरशी) — साठवण बुरशी; दमट साठवणीत वेगाने पसरते; जखमा आणि मानेद्वारे प्रवेश.",
        "immediate_hi": ["तुरंत झुलसे बल्ब स्वस्थ से अलग करें", "स्वस्थ बल्ब हवादार सूखी छाया में रखें", "भंडारित बल्ब को गीला न करें — नमी से तेज़ फैलाव", "काले पाउडर वाले बल्ब फेंकें — ठीक नहीं होते", "भंडारण कक्ष को SO2 पैड से धुआंकरण करें"],
        "immediate_mr": ["ताबडतोब कुजलेले कांदे निरोगीपासून वेगळे करा", "निरोगी कांदे हवेशीर कोरड्या सावलीत ठेवा", "साठवलेले कांदे ओले करू नका — ओलाव्याने वेगाने पसरते", "काळी पावडर असलेले कांदे टाकून द्या — बरे होत नाहीत", "साठवण खोलीत SO2 पॅडने धुरी करा"],
        "chemical_hi": "भंडारण पूर्व डुबोना: Thiabendazole 450g/L @ 2ml/L 5 मिनट | फसल छिड़काव: Captan 50WP @ 2.5g/L | वैकल्पिक: Iprodione + Thiophanate-methyl @ 1.5g/L | भंडारण धुआंकरण: SO2 पैड",
        "chemical_mr": "साठवणपूर्व बुडवणे: Thiabendazole 450g/L @ 2ml/L 5 मिनिट | पीक फवारणी: Captan 50WP @ 2.5g/L | पर्याय: Iprodione + Thiophanate-methyl @ 1.5g/L | साठवण धुरी: SO2 पॅड",
        "biological_hi": "Trichoderma asperellum बल्ब की सतह पर लेप | Bacillus amyloliquefaciens जैव-कवकनाशी डुबोना | नीम तेल 2% + बेकिंग सोडा 0.5% डुबोना",
        "biological_mr": "Trichoderma asperellum कांद्याच्या पृष्ठभागावर लेप | Bacillus amyloliquefaciens जैव-बुरशीनाशक बुडवणे | निंबाचे तेल 2% + बेकिंग सोडा 0.5% बुडवणे",
        "cultural_hi": "भंडारण से पहले 35°C पर 7-10 दिन ठीक करें | 0-4°C / 65-70% आर्द्रता पर रखें | जालीदार लकड़ी के टोकरे — पॉलीथीन बैग नहीं | हर 2 सप्ताह जांचें",
        "cultural_mr": "साठवणुकीपूर्वी 35°C वर 7-10 दिवस वाळवा | 0-4°C / 65-70% आर्द्रता राखा | जाळीदार लाकडी पेट्या — पॉलिथिन पिशव्या नाही | दर 2 आठवड्यांनी तपासा",
        "fertilizer_hi": "कटाई से पहले पोटेशियम @ 40kg K2O/हेक्टेयर | कटाई से 2 सप्ताह पहले फोलियर कैल्शियम 0.3% | कटाई से 3 सप्ताह पहले सिंचाई और उर्वरक कम करें | सल्फर @ 20kg/हेक्टेयर — त्वचा मजबूत करता है",
        "fertilizer_mr": "काढणीपूर्वी पोटॅशियम @ 40kg K2O/हेक्टर | काढणीपूर्वी 2 आठवडे पर्णीय कॅल्शियम 0.3% | काढणीपूर्वी 3 आठवडे सिंचन व खत कमी करा | सल्फर @ 20kg/हेक्टर — त्वचा मजबूत करते",
        "irrigation_hi": "कटाई से 3 सप्ताह पहले सिंचाई बंद | कटी हुई फसल को बारिश से बचाएं | खेत ठीक करने के दौरान छाया जाल का उपयोग",
        "irrigation_mr": "काढणीपूर्वी 3 आठवडे सिंचन बंद | काढलेल्या पिकाला पावसापासून वाचवा | शेतात वाळवताना सावलीचे जाळे वापरा",
        "recovery_hi": "खेत में: 7-14 दिन। भंडारण में प्रभावित बल्ब ठीक नहीं होते — तुरंत हटाएं।",
        "recovery_mr": "शेतात: 7-14 दिवस. साठवणीत बाधित कांदे बरे होत नाहीत — ताबडतोब काढा.",
        "prevention_hi": "सही परिपक्वता पर कटाई (50-75% पत्तियां गिरी हों) | घाव या चोट वाले बल्ब कभी न रखें | भंडारण 5°C से कम | हर मौसम से पहले भंडारण कक्ष धुआंकरण",
        "prevention_mr": "योग्य परिपक्वतेवर काढणी (50-75% पाने पडलेली) | जखमी किंवा दुखावलेले कांदे कधीही साठवू नका | साठवण 5°C पेक्षा कमी | प्रत्येक हंगामापूर्वी साठवण खोली धुरी",
        "organic_hi": "बल्ब पंक्तियों के बीच सूखी रेत की परत | भंडारण से पहले बल्ब की सतह पर हल्दी पाउडर | ठीक करने के बाद गर्दन पर मोम की कोटिंग",
        "organic_mr": "कांद्याच्या ओळींमध्ये कोरड्या वाळूचा थर | साठवणुकीपूर्वी कांद्याच्या पृष्ठभागावर हळद पावडर | वाळवल्यानंतर मानेवर मेणाचा लेप",
        "cost_hi": "₹400-800/एकड़ (मुख्यतः भंडारण प्रबंधन)",
        "cost_mr": "₹400-800/एकर (मुख्यतः साठवण व्यवस्थापन)",
    },
    "caterpillar": {
        "severity": "HIGH", "pathogen": "Spodoptera exigua / S. frugiperda (armyworm) — larvae feed on leaves at night; egg masses on lower leaf surface; severe infestations defoliate crop within days.",
        "immediate": ["Handpick caterpillars in early morning when sluggish","Crush all egg masses on lower leaf surfaces","Set up light traps (1/acre) overnight","Install pheromone traps: 5/acre for monitoring","Apply insecticide at DUSK — caterpillars feed at night"],
        "chemical": "PRIMARY: Chlorpyrifos 20EC @ 2ml/L | ALTERNATE: Emamectin benzoate 5SG @ 0.4g/L | ROTATION: Spinosad 45SC @ 0.3ml/L | SEVERE: Indoxacarb 14.5SC @ 0.75ml/L | ALWAYS apply at dusk — larvae hide in soil by day",
        "biological": "Bacillus thuringiensis (Bt) @ 2g/L — most effective bio-option | NPV @ 250 LE/ha | Release Trichogramma chilonis @ 50,000/ha/week | Beauveria bassiana @ 5g/L in the evening",
        "cultural": "Deep plough after harvest to expose pupae | Keep field bunds clean | Intercrop coriander to attract natural predators | Bird perches (T-shaped) @ 10/acre",
        "fertilizer": "Silicon @ 100kg/ha hardens leaf cell walls | Potassium @ 80kg/ha boosts immunity | Avoid excess Nitrogen — soft growth attracts pests | Foliar Boron 0.2% reduces leaf tenderness",
        "irrigation": "Irrigate by day — dry soil at night deters larvae | Overhead irrigation in evening knocks young caterpillars off leaves",
        "recovery": "5-10 days population knockdown. 14-21 days for full plant recovery.",
        "prevention": "Pheromone traps from week 2 of crop age | Light traps every night in season | Scout for egg masses twice/week | Apply Bt before 3rd instar — larvae hardest to kill when mature",
        "organic": "NSKE 5% spray every 7 days | Cow urine + neem soap spray every 5 days | Tobacco decoction 3% at dusk | Ash dusting on leaves in the evening",
        "cost": "Rs.600-1,000/acre per spray cycle",
        "pathogen_hi": "Spodoptera exigua / S. frugiperda (सैनिक कीट) — लार्वा रात में पत्तियां खाते हैं; अंडे पत्ती की निचली सतह पर; गंभीर संक्रमण में दिनों में पूरी फसल नष्ट हो सकती है।",
        "pathogen_mr": "Spodoptera exigua / S. frugiperda (लष्करी अळी) — अळ्या रात्री पाने खातात; अंडी पानांच्या खालच्या बाजूला; गंभीर प्रादुर्भावात दिवसांत संपूर्ण पीक उद्ध्वस्त होऊ शकते.",
        "immediate_hi": ["सुबह जल्दी इल्लियां हाथ से पकड़ें", "पत्तियों की निचली सतह पर अंडे के समूह तुरंत कुचलें", "रात को लाइट ट्रैप (1/एकड़) लगाएं", "फेरोमोन ट्रैप: 5/एकड़", "शाम को कीटनाशक डालें — इल्लियां रात में खाती हैं"],
        "immediate_mr": ["सकाळी लवकर अळ्या हाताने गोळा करा", "पानांच्या खालच्या बाजूवरील अंड्यांचे समूह त्वरित चिरडा", "रात्री प्रकाश सापळे (1/एकर) लावा", "फेरोमोन सापळे: 5/एकर", "संध्याकाळी कीटकनाशक फवारा — अळ्या रात्री खातात"],
        "chemical_hi": "प्राथमिक: Chlorpyrifos 20EC @ 2ml/L | वैकल्पिक: Emamectin benzoate 5SG @ 0.4g/L | रोटेशन: Spinosad 45SC @ 0.3ml/L | गंभीर: Indoxacarb 14.5SC @ 0.75ml/L | शाम को डालें — इल्लियां दिन में मिट्टी में छिपती हैं",
        "chemical_mr": "प्राथमिक: Chlorpyrifos 20EC @ 2ml/L | पर्याय: Emamectin benzoate 5SG @ 0.4g/L | आवर्तन: Spinosad 45SC @ 0.3ml/L | गंभीर: Indoxacarb 14.5SC @ 0.75ml/L | संध्याकाळी फवारा — अळ्या दिवसा मातीत लपतात",
        "biological_hi": "Bacillus thuringiensis (Bt) @ 2g/L — सबसे प्रभावी जैव विकल्प | NPV @ 250 LE/हेक्टेयर | Trichogramma chilonis @ 50,000/हेक्टेयर/सप्ताह | Beauveria bassiana @ 5g/L शाम को",
        "biological_mr": "Bacillus thuringiensis (Bt) @ 2g/L — सर्वात प्रभावी जैव पर्याय | NPV @ 250 LE/हेक्टर | Trichogramma chilonis @ 50,000/हेक्टर/आठवडा | Beauveria bassiana @ 5g/L संध्याकाळी",
        "cultural_hi": "कटाई के बाद गहरी जुताई — प्यूपा को धूप में उजागर करें | मेड साफ रखें | धनिया के साथ अंतरफसल | पक्षी बसने की जगह (T-आकार) @ 10/एकड़",
        "cultural_mr": "काढणीनंतर खोल नांगरणी — प्यूपे उन्हाला उघड करा | बांध स्वच्छ ठेवा | कोथिंबीर आंतरपीक | पक्षी बसण्याची जागा (T-आकार) @ 10/एकर",
        "fertilizer_hi": "सिलिकॉन @ 100kg/हेक्टेयर — पत्ती की दीवारें मजबूत करता है | पोटेशियम @ 80kg/हेक्टेयर | अधिक नाइट्रोजन से बचें | फोलियर बोरॉन 0.2%",
        "fertilizer_mr": "सिलिकॉन @ 100kg/हेक्टर — पानांच्या भिंती मजबूत करते | पोटॅशियम @ 80kg/हेक्टर | जास्त नत्र टाळा | पर्णीय बोरॉन 0.2%",
        "irrigation_hi": "दिन में सिंचाई करें — रात को सूखी मिट्टी इल्लियों को हतोत्साहित करती है | शाम को ओवरहेड सिंचाई छोटी इल्लियों को पत्तियों से गिरा देती है",
        "irrigation_mr": "दिवसा सिंचन करा — रात्री कोरडी माती अळ्यांना परावृत्त करते | संध्याकाळी ओव्हरहेड सिंचन लहान अळ्यांना पानांवरून खाली पाडते",
        "recovery_hi": "5-10 दिन में इल्ली संख्या कम होगी। 14-21 दिन में पौधे ठीक होंगे।",
        "recovery_mr": "5-10 दिवसांत अळ्यांची संख्या कमी होईल. 14-21 दिवसांत पीक बरे होईल.",
        "prevention_hi": "दिन 2 से फेरोमोन ट्रैप | रात को लाइट ट्रैप | सप्ताह में दो बार अंडे की जांच | 3rd instar से पहले Bt डालें",
        "prevention_mr": "दिन 2 पासून फेरोमोन सापळे | रात्री प्रकाश सापळे | आठवड्यातून दोनदा अंड्यांची तपासणी | 3rd instar आधी Bt फवारा",
        "organic_hi": "NSKE 5% हर 7 दिन | गोमूत्र + नीम साबुन हर 5 दिन | शाम को तंबाकू काढ़ा 3% | शाम को पत्तियों पर राख छिड़कें",
        "organic_mr": "NSKE 5% दर 7 दिवसांनी | गोमूत्र + निंब साबण दर 5 दिवसांनी | संध्याकाळी तंबाखू काढा 3% | संध्याकाळी पानांवर राख टाका",
        "cost_hi": "₹600-1,000/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹600-1,000/एकर प्रति फवारणी चक्र",
    },
    "downy_mildew": {
        "severity": "HIGH", "pathogen": "Peronospora destructor (oomycete) — pale grey/violet downy coating; spreads via wind-borne spores in cool moist conditions (10-15C nights); can wipe out entire crop within 2 weeks.",
        "immediate": ["Apply systemic fungicide within 24h — do not wait","Remove heavily infected leaves and burn immediately","Stop all overhead irrigation","Thin dense crop areas to improve ventilation","Monitor adjacent fields — spreads across farm boundaries by wind"],
        "chemical": "PRIMARY: Metalaxyl-M + Mancozeb @ 2.5g/L (systemic) | ALTERNATE: Dimethomorph 50WP @ 1g/L | CURATIVE: Cymoxanil 8% + Mancozeb 64WP @ 2.5g/L | INTERVAL: Every 7 days in humid spells | NEVER use Metalaxyl alone — always mix with a protectant",
        "biological": "Bacillus subtilis @ 3ml/L every 10 days | Trichoderma viride soil drench @ 4g/L | Neem oil 3% every 14 days in non-peak periods | Phosphorous acid-based biocontrols",
        "cultural": "Raised beds for drainage | 20cm+ row spacing | Avoid low-lying/frost-pocket planting areas | Remove volunteer onion plants between seasons",
        "fertilizer": "Potassium @ 80kg K2O/ha | Calcium nitrate @ 150kg/ha | Reduce Nitrogen in humid periods | Foliar Zinc 0.3% + Manganese 0.1% monthly",
        "irrigation": "Drip ONLY during active disease | Never water in the evening | Keep leaf surface dry | Reduce frequency during foggy/overcast periods",
        "recovery": "12-20 days with systemic fungicide under dry conditions. Cool wet weather extends to 25+ days.",
        "prevention": "Resistant varieties where available | Preventive Mancozeb before monsoon onset | Scout from day 25 in cool-humid weather windows",
        "organic": "Bordeaux mixture 1% every 10 days | NSKE 5% weekly | Cow urine 5% every 7 days during cool-wet spells",
        "cost": "Rs.700-1,200/acre per spray cycle",
        "pathogen_hi": "Peronospora destructor (ओमाइसीट) — पत्तियों पर हल्की भूरी/बैंगनी परत; ठंडी नमी (10-15°C रात) में हवाई बीजाणुओं से फैलता है; 2 हफ्तों में पूरी फसल खराब कर सकता है।",
        "pathogen_mr": "Peronospora destructor (ओमायसीट) — पानांवर फिकट राखाडी/जांभळी पातळ थर; थंड दमट (10-15°C रात्र) हवेतील बीजाणूंनी पसरते; 2 आठवड्यांत संपूर्ण पीक नष्ट करू शकते.",
        "immediate_hi": ["24 घंटे में प्रणालीगत कवकनाशी डालें", "भारी संक्रमित पत्तियां जलाएं", "ओवरहेड सिंचाई बंद करें", "घने क्षेत्रों को पतला करें", "पड़ोसी खेतों की निगरानी करें — हवा से फैलता है"],
        "immediate_mr": ["24 तासांत प्रणालीगत बुरशीनाशक फवारा", "जड बाधित पाने जाळा", "ओव्हरहेड सिंचन बंद करा", "दाट भागांची विरळणी करा", "शेजारच्या शेतांचे निरीक्षण करा — वाऱ्याने पसरते"],
        "chemical_hi": "प्राथमिक: Metalaxyl-M + Mancozeb @ 2.5g/L | वैकल्पिक: Dimethomorph 50WP @ 1g/L | उपचारात्मक: Cymoxanil 8% + Mancozeb 64WP @ 2.5g/L | अंतराल: नमी में हर 7 दिन | Metalaxyl अकेले न डालें — हमेशा रक्षक के साथ",
        "chemical_mr": "प्राथमिक: Metalaxyl-M + Mancozeb @ 2.5g/L | पर्याय: Dimethomorph 50WP @ 1g/L | उपचारात्मक: Cymoxanil 8% + Mancozeb 64WP @ 2.5g/L | अंतर: दमट हवामानात दर 7 दिवस | Metalaxyl एकट्याने नाही — नेहमी संरक्षकासोबत",
        "biological_hi": "Bacillus subtilis @ 3ml/L हर 10 दिन | Trichoderma viride मिट्टी ड्रेंच @ 4g/L | नीम तेल 3% हर 14 दिन | फॉस्फोरस एसिड-आधारित जैव नियंत्रण",
        "biological_mr": "Bacillus subtilis @ 3ml/L दर 10 दिवस | Trichoderma viride माती ओतणी @ 4g/L | निंबाचे तेल 3% दर 14 दिवस | फॉस्फोरस ॲसिड-आधारित जैव नियंत्रण",
        "cultural_hi": "जल निकासी के लिए उठी क्यारियां | 20+ सेमी पंक्ति दूरी | निचले/ठंडे क्षेत्रों में न लगाएं | मौसमों के बीच जंगली प्याज पौधे हटाएं",
        "cultural_mr": "निचऱ्यासाठी उंच वाफे | 20+ सेमी ओळ अंतर | सखल/थंड भागात लावणी नाही | हंगामांदरम्यान जंगली कांदा झाडे काढा",
        "fertilizer_hi": "पोटेशियम @ 80kg K2O/हेक्टेयर | कैल्शियम नाइट्रेट @ 150kg/हेक्टेयर | नमी में नाइट्रोजन कम करें | फोलियर जिंक 0.3% + मैंगनीज 0.1% मासिक",
        "fertilizer_mr": "पोटॅशियम @ 80kg K2O/हेक्टर | कॅल्शियम नायट्रेट @ 150kg/हेक्टर | दमट हवामानात नत्र कमी करा | पर्णीय झिंक 0.3% + मॅंगनीज 0.1% मासिक",
        "irrigation_hi": "सक्रिय रोग में केवल ड्रिप | शाम को कभी पानी न दें | पत्तियां हमेशा सूखी रखें | कोहरे/बादल में सिंचाई कम करें",
        "irrigation_mr": "सक्रिय रोगात फक्त ठिबक | संध्याकाळी कधीही पाणी देऊ नका | पाने नेहमी कोरडी ठेवा | धुके/ढगाळ हवामानात सिंचन कमी करा",
        "recovery_hi": "सूखे में प्रणालीगत कवकनाशी से 12-20 दिन।",
        "recovery_mr": "कोरड्या हवामानात प्रणालीगत बुरशीनाशकाने 12-20 दिवस.",
        "prevention_hi": "जहां उपलब्ध हो प्रतिरोधी किस्में | मानसून से पहले निवारक Mancozeb | ठंडी नमी में दिन 25 से निगरानी",
        "prevention_mr": "उपलब्ध असल्यास प्रतिरोधक वाण | पावसाळ्यापूर्वी प्रतिबंधक Mancozeb | थंड दमट हवामानात दिन 25 पासून निरीक्षण",
        "organic_hi": "बोर्डो मिश्रण 1% हर 10 दिन | NSKE 5% साप्ताहिक | ठंडे-गीले मौसम में गोमूत्र 5% हर 7 दिन",
        "organic_mr": "बोर्डो मिश्रण 1% दर 10 दिवस | NSKE 5% साप्ताहिक | थंड-दमट हवामानात गोमूत्र 5% दर 7 दिवस",
        "cost_hi": "₹700-1,200/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹700-1,200/एकर प्रति फवारणी चक्र",
    },
    "fusarium": {
        "severity": "HIGH", "pathogen": "Fusarium oxysporum f.sp. cepae (soil-borne fungus) — root rot, yellowing, plant collapse; survives 10+ years in soil; spreads via infected seed, tools & water.",
        "immediate": ["Remove & destroy all wilting/yellowing plants — active infection sources","Soil drench: Carbendazim 50WP @ 1g/L around healthy plant bases","Stop over-irrigation — wet soil accelerates spread dramatically","Mark infected zones — do not cultivate or move soil from them","Disinfect all tools with 70% alcohol between rows"],
        "chemical": "DRENCH: Carbendazim 50WP @ 1g/L every 14 days | SEED TX: Bavistin 2g/kg + Thiram 2g/kg | PRE-PLANT SOIL: Formalin 0.5% drench 3 weeks before planting | FOLIAR: Propiconazole 25EC @ 1ml/L early stage",
        "biological": "Trichoderma viride @ 4-5g/L soil drench — most effective for Fusarium | Pseudomonas fluorescens @ 2.5kg/ha | VAM mycorrhizae @ 5kg/ha at transplanting | Bacillus subtilis root dip",
        "cultural": "4-year rotation without alliums | Raised planting beds | Certified disease-free seed only | No transplanting into Fusarium-history soil without prior treatment",
        "fertilizer": "Phosphorus @ 80kg P2O5/ha for root strength | Potassium @ 100kg K2O/ha | Zinc sulfate 20kg/ha in soil | Avoid excess Nitrogen",
        "irrigation": "Drip ONLY. No overwatering — Fusarium thrives in waterlogged soil | Maintain 60% field capacity | Stop 15 days before harvest",
        "recovery": "Infected plants cannot recover. Spread stabilises in 15-25 days with aggressive management.",
        "prevention": "Soil solarisation (polyethylene sheet 6 weeks) in endemic fields | Seed treatment mandatory | Trichoderma soil treatment before each crop cycle",
        "organic": "Neem cake @ 400kg/ha before planting | Wood ash @ 500kg/ha to alter soil pH | Ginger extract 5% soil drench preventively",
        "cost": "Rs.1,200-2,000/acre (soil treatment required)",
        "pathogen_hi": "Fusarium oxysporum f.sp. cepae (मृदा-जनित कवक) — जड़ सड़न, पीलापन, पौधा गिरना; मिट्टी में 10+ साल जीवित; संक्रमित बीज, औजारों से फैलता है।",
        "pathogen_mr": "Fusarium oxysporum f.sp. cepae (मातीजनित बुरशी) — मूळ कूज, पिवळेपणा, झाड कोलमडणे; जमिनीत 10+ वर्षे जगते; बाधित बियाणे, अवजारांनी पसरते.",
        "immediate_hi": ["सभी मुरझाए/पीले पौधे हटाएं और जलाएं — सक्रिय संक्रमण स्रोत", "स्वस्थ पौधों के आधार पर Carbendazim 50WP @ 1g/L मिट्टी ड्रेंच", "अत्यधिक सिंचाई बंद करें — गीली मिट्टी में Fusarium तेज़ी से फैलता है", "प्रभावित क्षेत्र चिन्हित करें — मिट्टी को स्वस्थ क्षेत्रों में न ले जाएं", "पंक्तियों के बीच औजारों को 70% अल्कोहल से साफ करें"],
        "immediate_mr": ["सर्व कोमेजलेली/पिवळी झाडे काढा व जाळा — सक्रिय संसर्ग स्रोत", "निरोगी झाडांच्या मुळाशी Carbendazim 50WP @ 1g/L माती ओतणी", "अतिसिंचन बंद करा — ओल्या जमिनीत Fusarium वेगाने पसरतो", "बाधित क्षेत्र खुणवा — माती निरोगी भागात नेऊ नका", "ओळींमधील अवजारे 70% अल्कोहलने स्वच्छ करा"],
        "chemical_hi": "ड्रेंच: Carbendazim 50WP @ 1g/L हर 14 दिन | बीज उपचार: Bavistin 2g/kg + Thiram 2g/kg | पूर्व-रोपण: Formalin 0.5% मिट्टी ड्रेंच 3 सप्ताह पहले | पत्तेदार: Propiconazole 25EC @ 1ml/L",
        "chemical_mr": "ओतणी: Carbendazim 50WP @ 1g/L दर 14 दिवस | बीज प्रक्रिया: Bavistin 2g/kg + Thiram 2g/kg | लावणीपूर्व: Formalin 0.5% माती ओतणी 3 आठवडे आधी | पर्णीय: Propiconazole 25EC @ 1ml/L",
        "biological_hi": "Trichoderma viride @ 4-5g/L मिट्टी ड्रेंच — Fusarium के लिए सबसे प्रभावी जैव नियंत्रण | Pseudomonas fluorescens @ 2.5kg/हेक्टेयर | VAM माइकोराइजा @ 5kg/हेक्टेयर | Bacillus subtilis जड़ में डुबोएं",
        "biological_mr": "Trichoderma viride @ 4-5g/L माती ओतणी — Fusarium साठी सर्वात प्रभावी जैव नियंत्रण | Pseudomonas fluorescens @ 2.5kg/हेक्टर | VAM मायकोरायझा @ 5kg/हेक्टर | Bacillus subtilis मुळात बुडवा",
        "cultural_hi": "4 साल का फसल चक्र बिना प्याज के | उठी क्यारियां | प्रमाणित रोग-मुक्त बीज | Fusarium वाली मिट्टी में बिना उपचार के रोपाई न करें",
        "cultural_mr": "4 वर्षांचे पीक फेरपालट कांद्याशिवाय | उंच वाफे | प्रमाणित रोगमुक्त बियाणे | Fusarium असलेल्या जमिनीत उपचाराशिवाय लावणी नाही",
        "fertilizer_hi": "फास्फोरस @ 80kg P2O5/हेक्टेयर | पोटेशियम @ 100kg K2O/हेक्टेयर | जिंक सल्फेट 20kg/हेक्टेयर | अधिक नाइट्रोजन नहीं",
        "fertilizer_mr": "स्फुरद @ 80kg P2O5/हेक्टर | पोटॅशियम @ 100kg K2O/हेक्टर | झिंक सल्फेट 20kg/हेक्टर | जास्त नत्र नाही",
        "irrigation_hi": "केवल ड्रिप। अत्यधिक पानी न दें — Fusarium जलभराव में पनपता है | 60% क्षेत्र क्षमता | कटाई से 15 दिन पहले बंद",
        "irrigation_mr": "फक्त ठिबक. अतिपाणी नको — Fusarium पाणी साचलेल्या जमिनीत वाढतो | 60% क्षेत्र क्षमता | काढणीपूर्वी 15 दिवस बंद",
        "recovery_hi": "संक्रमित पौधे ठीक नहीं होते। 15-25 दिन में प्रसार स्थिर होता है।",
        "recovery_mr": "बाधित झाडे बरे होत नाहीत. 15-25 दिवसांत प्रसार स्थिर होतो.",
        "prevention_hi": "स्थानिक क्षेत्रों में मिट्टी सौरीकरण (6 सप्ताह पॉलीथिन शीट) | बीज उपचार अनिवार्य | हर फसल चक्र में Trichoderma मिट्टी उपचार",
        "prevention_mr": "स्थानिक भागात माती सौरीकरण (6 आठवडे पॉलिथिन शीट) | बीज प्रक्रिया अनिवार्य | प्रत्येक पीक चक्रात Trichoderma माती उपचार",
        "organic_hi": "नीम केक @ 400kg/हेक्टेयर रोपाई से पहले | लकड़ी की राख @ 500kg/हेक्टेयर | अदरक अर्क 5% मिट्टी ड्रेंच",
        "organic_mr": "निंबोळी पेंड @ 400kg/हेक्टर लावणीपूर्वी | लाकडाची राख @ 500kg/हेक्टर | आले अर्क 5% माती ओतणी",
        "cost_hi": "₹1,200-2,000/एकड़ (मिट्टी उपचार आवश्यक)",
        "cost_mr": "₹1,200-2,000/एकर (माती उपचार आवश्यक)",
    },
    "iris_yellow_virus": {
        "severity": "CRITICAL", "pathogen": "Iris Yellow Spot Virus (IYSV) — transmitted by Thrips tabaci; causes diamond/straw-coloured lesions on leaves & scapes; NO CURE — vector control is the only management strategy.",
        "immediate": ["Remove ALL symptomatic plants within 24h — no exceptions","Apply thrips insecticide within 24h of detection","Install blue sticky traps @ 10/acre immediately","Do NOT use cuttings/sets from infected crop","Notify neighbouring farmers — spreads via wind-blown thrips"],
        "chemical": "THRIPS: Spinosad 45SC @ 0.3ml/L OR Fipronil 5SC @ 1.5ml/L | ALTERNATE: Imidacloprid 17.8SL @ 0.5ml/L | OPTION 3: Thiamethoxam 25WG @ 0.3g/L | ROTATE all insecticide groups — thrips resistance develops very fast",
        "biological": "Beauveria bassiana @ 5g/L foliar (evening) | Metarhizium anisopliae @ 5g/L | Release Orius insidiosus — natural thrips predator | Neem oil 3% every 5 days during flush",
        "cultural": "Burn all virus-infected plants — no composting | No onion near garlic/leek/other alliums | Remove weed hosts from field borders | Reflective silver mulch throughout entire field",
        "fertilizer": "Silicon @ 150kg/ha — hardens leaf surface against thrips feeding | Potassium silicate @ 5ml/L foliar every 14 days | Avoid excess Nitrogen",
        "irrigation": "Overhead irrigation 2x/week physically dislodges thrips | Keep adequate soil moisture — drought worsens thrips damage",
        "recovery": "Virus: NO recovery ever. Thrips elimination: 5-7 days. New healthy growth: 14-21 days after vector control.",
        "prevention": "Mineral oil 1% on seedlings — blocks thrips settling | Mustard trap crop around border | Seed TX: Imidacloprid 70WS @ 7g/kg | Silver reflective mulch all season — reduces IYSV by 60-70%",
        "organic": "Garlic + chilli + soap spray 5% every 5 days | Neem oil 2% every 5 days | Tobacco decoction 3% spray | Marigold border rows",
        "cost": "Rs.1,200-2,000/acre (intensive vector management)",
        "pathogen_hi": "Iris Yellow Spot Virus (IYSV) — Thrips tabaci द्वारा फैलाया जाता है; पत्तियों पर हीरे के आकार के पुआल रंग के धब्बे; कोई इलाज नहीं — केवल वेक्टर नियंत्रण।",
        "pathogen_mr": "Iris Yellow Spot Virus (IYSV) — Thrips tabaci द्वारे पसरतो; पानांवर हिऱ्याच्या आकाराचे पेंढ्याच्या रंगाचे ठिपके; कोणताही उपचार नाही — फक्त वेक्टर नियंत्रण.",
        "immediate_hi": ["24 घंटे में सभी लक्षण वाले पौधे हटाएं", "24 घंटे में थ्रिप्स कीटनाशक डालें", "नीले चिपचिपे ट्रैप @ 10/एकड़ तुरंत लगाएं", "संक्रमित फसल की कटिंग/सेट का उपयोग न करें", "पड़ोसी किसानों को सूचित करें — थ्रिप्स हवा से उड़ते हैं"],
        "immediate_mr": ["24 तासांत सर्व लक्षण असलेली झाडे काढा", "24 तासांत थ्रिप्स कीटकनाशक फवारा", "निळे चिकट सापळे @ 10/एकर ताबडतोब लावा", "बाधित पिकाच्या कटिंग/सेट वापरू नका", "शेजारच्या शेतकऱ्यांना कळवा — थ्रिप्स वाऱ्याने उडतात"],
        "chemical_hi": "थ्रिप्स: Spinosad 45SC @ 0.3ml/L या Fipronil 5SC @ 1.5ml/L | वैकल्पिक: Imidacloprid 17.8SL @ 0.5ml/L | विकल्प 3: Thiamethoxam 25WG @ 0.3g/L | सभी कीटनाशक समूह बदलें — थ्रिप्स में प्रतिरोध जल्दी आता है",
        "chemical_mr": "थ्रिप्स: Spinosad 45SC @ 0.3ml/L किंवा Fipronil 5SC @ 1.5ml/L | पर्याय: Imidacloprid 17.8SL @ 0.5ml/L | पर्याय 3: Thiamethoxam 25WG @ 0.3g/L | सर्व कीटकनाशक गट फिरवा — थ्रिप्समध्ये प्रतिकार लवकर येतो",
        "biological_hi": "Beauveria bassiana @ 5g/L शाम को | Metarhizium anisopliae @ 5g/L | Orius insidiosus (थ्रिप्स शिकारी) छोड़ें | नीम तेल 3% हर 5 दिन",
        "biological_mr": "Beauveria bassiana @ 5g/L संध्याकाळी | Metarhizium anisopliae @ 5g/L | Orius insidiosus (थ्रिप्स शिकारी) सोडा | निंबाचे तेल 3% दर 5 दिवस",
        "cultural_hi": "वायरस संक्रमित पौधे जलाएं | लहसुन/लीक के पास प्याज न लगाएं | मेड पर खरपतवार हटाएं | पूरे खेत में चांदी के परावर्तक मल्च",
        "cultural_mr": "विषाणू बाधित झाडे जाळा | लसूण/लीकजवळ कांदा नाही | बांधावरील तण काढा | संपूर्ण शेतात चांदीचे परावर्तक आच्छादन",
        "fertilizer_hi": "सिलिकॉन @ 150kg/हेक्टेयर — थ्रिप्स के खिलाफ पत्ती की सतह कठोर बनाता है | पोटेशियम सिलिकेट @ 5ml/L हर 14 दिन | अधिक नाइट्रोजन नहीं",
        "fertilizer_mr": "सिलिकॉन @ 150kg/हेक्टर — थ्रिप्स विरुद्ध पानांची पृष्ठभाग कडक करते | पोटॅशियम सिलिकेट @ 5ml/L दर 14 दिवस | जास्त नत्र नाही",
        "irrigation_hi": "सप्ताह में 2 बार ओवरहेड सिंचाई थ्रिप्स को शारीरिक रूप से हटाती है | पर्याप्त मिट्टी की नमी बनाए रखें — सूखे में थ्रिप्स का नुकसान अधिक",
        "irrigation_mr": "आठवड्यातून 2 वेळा ओव्हरहेड सिंचन थ्रिप्सला शारीरिकरित्या काढते | पुरेशी जमिनीची ओलावा राखा — दुष्काळात थ्रिप्सचे नुकसान जास्त",
        "recovery_hi": "वायरस: कोई रिकवरी नहीं। थ्रिप्स 5-7 दिन में खत्म। नई स्वस्थ वृद्धि 14-21 दिन।",
        "recovery_mr": "विषाणू: कोणतीही पुनर्प्राप्ती नाही. थ्रिप्स 5-7 दिवसांत संपतात. नवीन निरोगी वाढ 14-21 दिवस.",
        "prevention_hi": "रोपों पर खनिज तेल 1% — थ्रिप्स को बसने से रोकता है | सरसों की फसल बॉर्डर पर | बीज उपचार: Imidacloprid 70WS @ 7g/kg | पूरे मौसम चांदी परावर्तक मल्च — IYSV 60-70% कम करता है",
        "prevention_mr": "रोपांवर खनिज तेल 1% — थ्रिप्सला बसण्यापासून रोखते | मोहरी सीमा पीक | बीज प्रक्रिया: Imidacloprid 70WS @ 7g/kg | संपूर्ण हंगामात चांदीचे परावर्तक आच्छादन — IYSV 60-70% कमी करते",
        "organic_hi": "लहसुन + मिर्च + साबुन 5% हर 5 दिन | नीम तेल 2% हर 5 दिन | तंबाकू काढ़ा 3% | गेंदा बॉर्डर पंक्तियां",
        "organic_mr": "लसूण + मिरची + साबण 5% दर 5 दिवस | निंबाचे तेल 2% दर 5 दिवस | तंबाखू काढा 3% | झेंडूच्या सीमा ओळी",
        "cost_hi": "₹1,200-2,000/एकड़ (गहन वेक्टर प्रबंधन)",
        "cost_mr": "₹1,200-2,000/एकर (सखोल वेक्टर व्यवस्थापन)",
    },
    "purple_blotch": {
        "severity": "HIGH", "pathogen": "Alternaria porri (fungus) — purple/brown lesions with yellow margins; 25-30C humid conditions; thrips wounds create fungal entry points.",
        "immediate": ["Apply fungicide within 24h of spotting purple lesions — do not wait","Remove & burn all heavily infected leaves","Control thrips simultaneously — they create entry wounds for the fungus","Stop overhead irrigation immediately","Thin dense areas to improve airflow and reduce humidity"],
        "chemical": "PRIMARY: Iprodione 50WP @ 2g/L OR Mancozeb 75WP @ 2.5g/L | ALTERNATE: Propiconazole 25EC @ 1ml/L | COMBO: Mancozeb + Carbendazim @ 2g/L each | INTERVAL: Every 7-10 days; minimum 3 sprays",
        "biological": "Trichoderma harzianum @ 4g/L soil drench | Bacillus subtilis @ 2ml/L foliar | Neem oil 3% + soap 0.5% every 14 days | Pseudomonas fluorescens @ 2.5kg/ha soil",
        "cultural": "3-year rotation without alliums | Seed TX: Thiram 75WP @ 3g/kg | Remove all crop debris post-harvest | 15x10 cm spacing minimum | Control thrips early",
        "fertilizer": "Reduce Nitrogen during infection | Potassium @ 80kg K2O/ha | Foliar Calcium 0.3% every 10 days | Zinc sulfate 0.5% foliar monthly",
        "irrigation": "Drip ONLY — keep foliage dry | Morning irrigation only (before 8am)",
        "recovery": "14-21 days with Iprodione or Propiconazole treatment.",
        "prevention": "Preventive Mancozeb before humid periods | Scout from day 30 twice/week | Control thrips continuously — direct link between thrips and purple blotch",
        "organic": "Dashaparni Ark 3% every 10 days | Cow urine 5% + turmeric 1% spray every 7 days | Garlic extract 5% every 7 days",
        "cost": "Rs.800-1,200/acre per spray cycle",
        "pathogen_hi": "Alternaria porri (कवक) — पीले हाशिए वाले बैंगनी/भूरे धब्बे; 25-30°C नमी; थ्रिप्स के घाव कवक के लिए प्रवेश द्वार बनाते हैं।",
        "pathogen_mr": "Alternaria porri (बुरशी) — पिवळ्या कडेसह जांभळे/तपकिरी ठिपके; 25-30°C दमट हवामान; थ्रिप्सच्या जखमा बुरशीसाठी प्रवेशद्वार तयार करतात.",
        "immediate_hi": ["24 घंटे में बैंगनी धब्बे दिखते ही कवकनाशी डालें", "भारी संक्रमित पत्तियां जलाएं", "थ्रिप्स को एक साथ नियंत्रित करें — वे कवक के लिए घाव बनाते हैं", "ओवरहेड सिंचाई बंद करें", "घने क्षेत्रों को पतला करें"],
        "immediate_mr": ["24 तासांत जांभळे ठिपके दिसताच बुरशीनाशक फवारा", "जड बाधित पाने जाळा", "थ्रिप्स एकाच वेळी नियंत्रित करा — ते बुरशीसाठी जखमा तयार करतात", "ओव्हरहेड सिंचन बंद करा", "दाट भागांची विरळणी करा"],
        "chemical_hi": "प्राथमिक: Iprodione 50WP @ 2g/L या Mancozeb 75WP @ 2.5g/L | वैकल्पिक: Propiconazole 25EC @ 1ml/L | संयोजन: Mancozeb + Carbendazim @ 2g/L प्रत्येक | अंतराल: 7-10 दिन; न्यूनतम 3 छिड़काव",
        "chemical_mr": "प्राथमिक: Iprodione 50WP @ 2g/L किंवा Mancozeb 75WP @ 2.5g/L | पर्याय: Propiconazole 25EC @ 1ml/L | संयोजन: Mancozeb + Carbendazim @ 2g/L प्रत्येकी | अंतर: 7-10 दिवस; किमान 3 फवारण्या",
        "biological_hi": "Trichoderma harzianum @ 4g/L मिट्टी में | Bacillus subtilis @ 2ml/L पत्तेदार | नीम तेल 3% + साबुन 0.5% हर 14 दिन | Pseudomonas fluorescens @ 2.5kg/हेक्टेयर",
        "biological_mr": "Trichoderma harzianum @ 4g/L जमिनीत | Bacillus subtilis @ 2ml/L पर्णीय | निंबाचे तेल 3% + साबण 0.5% दर 14 दिवस | Pseudomonas fluorescens @ 2.5kg/हेक्टर",
        "cultural_hi": "3 साल का फसल चक्र | बीज उपचार: Thiram 75WP @ 3g/kg | कटाई के बाद मलबा हटाएं | 15x10 सेमी दूरी | थ्रिप्स जल्दी नियंत्रित करें",
        "cultural_mr": "3 वर्षांचे पीक फेरपालट | बीज प्रक्रिया: Thiram 75WP @ 3g/kg | काढणीनंतर अवशेष काढा | 15x10 सेमी अंतर | थ्रिप्स लवकर नियंत्रित करा",
        "fertilizer_hi": "संक्रमण के दौरान नाइट्रोजन कम करें | पोटेशियम @ 80kg K2O/हेक्टेयर | फोलियर कैल्शियम 0.3% हर 10 दिन | जिंक सल्फेट 0.5% मासिक",
        "fertilizer_mr": "संसर्गादरम्यान नत्र कमी करा | पोटॅशियम @ 80kg K2O/हेक्टर | पर्णीय कॅल्शियम 0.3% दर 10 दिवस | झिंक सल्फेट 0.5% मासिक",
        "irrigation_hi": "केवल ड्रिप — पत्तियां सूखी रखें | केवल सुबह सिंचाई",
        "irrigation_mr": "फक्त ठिबक — पाने कोरडी ठेवा | फक्त सकाळी सिंचन",
        "recovery_hi": "Iprodione या Propiconazole से 14-21 दिन।",
        "recovery_mr": "Iprodione किंवा Propiconazole ने 14-21 दिवस.",
        "prevention_hi": "आर्द्र मौसम से पहले निवारक Mancozeb | दिन 30 से सप्ताह में दो बार निगरानी | थ्रिप्स लगातार नियंत्रित करें",
        "prevention_mr": "दमट हवामानापूर्वी प्रतिबंधक Mancozeb | दिन 30 पासून आठवड्यातून दोनदा निरीक्षण | थ्रिप्स सतत नियंत्रित करा",
        "organic_hi": "दशपर्णी अर्क 3% हर 10 दिन | गोमूत्र 5% + हल्दी 1% हर 7 दिन | लहसुन अर्क 5% हर 7 दिन",
        "organic_mr": "दशपर्णी अर्क 3% दर 10 दिवस | गोमूत्र 5% + हळद 1% दर 7 दिवस | लसूण अर्क 5% दर 7 दिवस",
        "cost_hi": "₹800-1,200/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹800-1,200/एकर प्रति फवारणी चक्र",
    },
    "rust": {
        "severity": "MEDIUM", "pathogen": "Puccinia allii (fungus) — orange/brown powdery pustules; spreads via wind spores; warm days (20-25C) + cool humid nights; alternate hosts: garlic, leek, chives.",
        "immediate": ["Spray sulfur fungicide within 48h of detecting orange pustules","Remove & burn all pustule-bearing leaves","Do NOT disturb infected plants mechanically — movement spreads spores","Disinfect field equipment before moving to other fields","Remove garlic/leek/chives from nearby areas — they are alternate hosts"],
        "chemical": "PRIMARY: Sulfur 80WP @ 3g/L — direct rust killer | ALTERNATE: Propiconazole 25EC @ 1ml/L | ADVANCED: Triadimefon 25WP @ 1g/L | COMBO: Tebuconazole + Trifloxystrobin @ 0.5ml/L | INTERVAL: Every 10-14 days; 3 applications minimum",
        "biological": "Sulfur-based Thiovit @ 3g/L every 10 days | Neem oil 3% every 10 days | Bacillus subtilis @ 3ml/L preventively | Milk spray 10% — proven rust suppression via fatty acids",
        "cultural": "Avoid dense planting — rust thrives in humid canopy micro-climates | No onion near leek/garlic/chives | Early planting avoids peak rust season",
        "fertilizer": "Potassium silicate @ 5ml/L foliar — toughens cell walls | Sulfur @ 20kg/ha soil | Reduce Nitrogen — excess N increases severity | Zinc + Manganese 0.2% each monthly",
        "irrigation": "No overhead irrigation — spreads spores via water droplets | Drip ONLY | Morning irrigation only (before 8am)",
        "recovery": "10-14 days with sulfur fungicide in moderate temperatures.",
        "prevention": "Rust-tolerant varieties where available | Monitor for orange pustules from week 4 | Preventive sulfur spray in endemic areas | Remove all debris after harvest",
        "organic": "Milk + baking soda 0.5% spray every 7 days | Sulphur dust @ 25kg/ha early morning | Garlic extract + ginger extract 5% combined spray",
        "cost": "Rs.400-800/acre per spray cycle",
        "pathogen_hi": "Puccinia allii (कवक) — नारंगी/भूरे पाउडर जैसे फफोले; हवा के बीजाणुओं से फैलता है; गर्म दिन (20-25°C) + ठंडी नमी वाली रात; वैकल्पिक मेज़बान: लहसुन, लीक, चाइव्स।",
        "pathogen_mr": "Puccinia allii (बुरशी) — नारंगी/तपकिरी पावडरसारखे फोड; वाऱ्यातील बीजाणूंनी पसरते; उष्ण दिवस (20-25°C) + थंड दमट रात्र; पर्यायी यजमान: लसूण, लीक, चाइव्ज.",
        "immediate_hi": ["48 घंटे में नारंगी फफोले मिलते ही सल्फर कवकनाशी छिड़कें", "फफोलेदार सभी पत्तियां हटाएं और जलाएं", "संक्रमित पौधों को यांत्रिक रूप से न हिलाएं — बीजाणु फैलते हैं", "दूसरे खेतों में जाने से पहले उपकरण साफ करें", "लहसुन/लीक/चाइव्स को आसपास से हटाएं"],
        "immediate_mr": ["48 तासांत नारंगी फोड दिसताच सल्फर बुरशीनाशक फवारा", "फोड असलेली सर्व पाने काढा व जाळा", "बाधित झाडे यांत्रिकरित्या हलवू नका — बीजाणू पसरतात", "दुसऱ्या शेतात जाण्यापूर्वी अवजारे स्वच्छ करा", "लसूण/लीक/चाइव्ज आजूबाजूने काढा"],
        "chemical_hi": "प्राथमिक: Sulfur 80WP @ 3g/L — सीधा रस्ट नाशक | वैकल्पिक: Propiconazole 25EC @ 1ml/L | उन्नत: Triadimefon 25WP @ 1g/L | संयोजन: Tebuconazole + Trifloxystrobin @ 0.5ml/L | अंतराल: 10-14 दिन; 3 छिड़काव",
        "chemical_mr": "प्राथमिक: Sulfur 80WP @ 3g/L — थेट गंज नाशक | पर्याय: Propiconazole 25EC @ 1ml/L | प्रगत: Triadimefon 25WP @ 1g/L | संयोजन: Tebuconazole + Trifloxystrobin @ 0.5ml/L | अंतर: 10-14 दिवस; 3 फवारण्या",
        "biological_hi": "Thiovit (सल्फर-आधारित) @ 3g/L हर 10 दिन | नीम तेल 3% हर 10 दिन | Bacillus subtilis @ 3ml/L निवारक | दूध का छिड़काव 10% — फैटी एसिड से रस्ट दमन",
        "biological_mr": "Thiovit (सल्फर-आधारित) @ 3g/L दर 10 दिवस | निंबाचे तेल 3% दर 10 दिवस | Bacillus subtilis @ 3ml/L प्रतिबंधक | दूध फवारणी 10% — फॅटी ॲसिडने गंज दमन",
        "cultural_hi": "घनी बुआई न करें — रस्ट नमी में पनपता है | लीक/लहसुन/चाइव्स के पास प्याज न लगाएं | जल्दी बुआई से रस्ट के मौसम से बचें",
        "cultural_mr": "दाट लावणी नाही — गंज दमट सूक्ष्म हवामानात वाढतो | लीक/लसूण/चाइव्जच्या शेजारी कांदा नाही | लवकर लागवड गंजाच्या हंगामापासून वाचवते",
        "fertilizer_hi": "पोटेशियम सिलिकेट @ 5ml/L पत्तेदार — कोशिका भित्ति मजबूत करता है | Sulfur @ 20kg/हेक्टेयर मिट्टी में | नाइट्रोजन कम करें | जिंक + मैंगनीज 0.2% मासिक",
        "fertilizer_mr": "पोटॅशियम सिलिकेट @ 5ml/L पर्णीय — पेशी भिंती मजबूत करते | Sulfur @ 20kg/हेक्टर जमिनीत | नत्र कमी करा | झिंक + मॅंगनीज 0.2% मासिक",
        "irrigation_hi": "ओवरहेड सिंचाई नहीं — बीजाणु पानी से फैलते हैं | केवल ड्रिप | केवल सुबह सिंचाई (8 बजे से पहले)",
        "irrigation_mr": "ओव्हरहेड सिंचन नाही — बीजाणू पाण्याने पसरतात | फक्त ठिबक | फक्त सकाळी सिंचन (सकाळी 8 पूर्वी)",
        "recovery_hi": "सामान्य तापमान में सल्फर कवकनाशी से 10-14 दिन।",
        "recovery_mr": "सामान्य तापमानात सल्फर बुरशीनाशकाने 10-14 दिवस.",
        "prevention_hi": "जहां उपलब्ध हो रस्ट-सहिष्णु किस्में | सप्ताह 4 से नारंगी फफोलों की निगरानी | स्थानिक क्षेत्रों में निवारक सल्फर | कटाई के बाद खेत साफ करें",
        "prevention_mr": "उपलब्ध असल्यास गंज-सहनशील वाण | आठवडा 4 पासून नारंगी फोडांचे निरीक्षण | स्थानिक भागात प्रतिबंधक सल्फर | काढणीनंतर शेत स्वच्छ करा",
        "organic_hi": "दूध + बेकिंग सोडा 0.5% हर 7 दिन | सल्फर धूल @ 25kg/हेक्टेयर सुबह | लहसुन + अदरक अर्क 5% संयुक्त छिड़काव",
        "organic_mr": "दूध + बेकिंग सोडा 0.5% दर 7 दिवस | सल्फर धूळ @ 25kg/हेक्टर सकाळी | लसूण + आले अर्क 5% एकत्र फवारणी",
        "cost_hi": "₹400-800/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹400-800/एकर प्रति फवारणी चक्र",
    },
    "stemphylium_blight": {
        "severity": "HIGH", "pathogen": "Stemphylium vesicarium (fungus) — tan spindle-shaped lesions with purple border; follows Botrytis/mechanical injury; peaks in waterlogged warm humid conditions (25-28C).",
        "immediate": ["Apply Iprodione 50WP @ 2g/L within 24h — fastest curative option","Remove & destroy all blighted tissue — do not compost","Increase plant spacing to improve canopy drying","Inspect adjacent rows for tan spindle-shaped lesions","Repair any leaking drippers causing localised wet spots"],
        "chemical": "PRIMARY: Iprodione 50WP @ 2g/L | ALTERNATE: Boscalid + Pyraclostrobin @ 0.75ml/L | CURATIVE: Fluazinam 50SC @ 1ml/L | TANK-MIX: Add Mancozeb 75WP @ 2.5g/L as protectant | ROTATE Group 2, 7 & 11 fungicides — resistance develops easily",
        "biological": "Bacillus subtilis QST 713 @ 3ml/L every 10 days | Trichoderma harzianum @ 4g/L soil + foliar | Neem oil 3% + neem cake @ 250kg/ha | Ampelomyces quisqualis biocontrol spray",
        "cultural": "Improve drainage — Stemphylium peaks in waterlogged conditions | No late-evening irrigation | Destroy all infected material — no composting | 2-year allium-free rotation minimum",
        "fertilizer": "Calcium @ 200kg/ha (CaNO3) — ESSENTIAL; Stemphylium exploits Ca-deficient tissue | Potassium @ 100kg K2O/ha | Reduce Nitrogen 20% during infection | Magnesium sulfate @ 15g/L foliar monthly",
        "irrigation": "Stop overhead irrigation permanently when disease is active | Drip only @ 65% field capacity | Irrigate 6-8am — canopy dry by mid-morning | Fix leaking drippers causing wet spots",
        "recovery": "14-20 days for healthy regrowth with Iprodione treatment.",
        "prevention": "Seed TX: Iprodione @ 3g/kg | Preventive spray at 30 and 50 days crop age | Scout from week 5 | Plant rows perpendicular to prevailing wind for better canopy drying",
        "organic": "Copper sulfate 0.5% spray weekly | Neem leaf decoction 5% foliar every 10 days | Cow urine 5% + turmeric 1% every 7 days",
        "cost": "Rs.700-1,200/acre per spray cycle",
        "pathogen_hi": "Stemphylium vesicarium (कवक) — बैंगनी किनारे वाले हल्के धुरिया धब्बे; Botrytis के बाद या यांत्रिक चोट के बाद आता है; जलभराव में गर्म नमी (25-28°C) में चरम पर।",
        "pathogen_mr": "Stemphylium vesicarium (बुरशी) — जांभळ्या कडेसह फिकट धुरकट ठिपके; Botrytis नंतर किंवा यांत्रिक दुखापतीनंतर येते; पाणी साचलेल्या उष्ण दमट हवामानात (25-28°C) कळस.",
        "immediate_hi": ["24 घंटे में Iprodione 50WP @ 2g/L — सबसे तेज़ उपचार", "सभी झुलसी पत्तियां हटाएं — खाद न बनाएं", "पौधों की दूरी बढ़ाएं", "पड़ोसी पंक्तियों में हल्के धुरिया धब्बे जांचें", "टपकने वाले ड्रिपर ठीक करें"],
        "immediate_mr": ["24 तासांत Iprodione 50WP @ 2g/L — सर्वात जलद उपचार", "सर्व करपलेली पाने काढा — खत करू नका", "झाडांचे अंतर वाढवा", "शेजारील ओळींमध्ये फिकट ठिपके तपासा", "गळणारे ड्रिपर दुरुस्त करा"],
        "chemical_hi": "प्राथमिक: Iprodione 50WP @ 2g/L | वैकल्पिक: Boscalid + Pyraclostrobin @ 0.75ml/L | उपचारात्मक: Fluazinam 50SC @ 1ml/L | टैंक मिक्स: Mancozeb 75WP @ 2.5g/L रक्षक के रूप में | Group 2, 7 और 11 कवकनाशी बदलते रहें",
        "chemical_mr": "प्राथमिक: Iprodione 50WP @ 2g/L | पर्याय: Boscalid + Pyraclostrobin @ 0.75ml/L | उपचारात्मक: Fluazinam 50SC @ 1ml/L | टँक मिक्स: Mancozeb 75WP @ 2.5g/L संरक्षक म्हणून | Group 2, 7 आणि 11 बुरशीनाशके फिरवा",
        "biological_hi": "Bacillus subtilis QST 713 @ 3ml/L हर 10 दिन | Trichoderma harzianum @ 4g/L मिट्टी + पत्तेदार | नीम तेल 3% + नीम केक @ 250kg/हेक्टेयर | Ampelomyces quisqualis जैव नियंत्रण",
        "biological_mr": "Bacillus subtilis QST 713 @ 3ml/L दर 10 दिवस | Trichoderma harzianum @ 4g/L माती + पर्णीय | निंबाचे तेल 3% + निंबोळी पेंड @ 250kg/हेक्टर | Ampelomyces quisqualis जैव नियंत्रण",
        "cultural_hi": "जल निकासी सुधारें — जलभराव में Stemphylium चरम पर | शाम को सिंचाई नहीं | सभी संक्रमित पौधे नष्ट करें | 2 साल का प्याज-मुक्त फसल चक्र",
        "cultural_mr": "निचरा सुधारा — पाणी साचल्यावर Stemphylium शिखरावर | संध्याकाळी सिंचन नाही | सर्व बाधित झाडे नष्ट करा | 2 वर्षांचे कांदा-मुक्त पीक फेरपालट",
        "fertilizer_hi": "कैल्शियम @ 200kg/हेक्टेयर (CaNO3) — अनिवार्य; Stemphylium Ca की कमी में बढ़ता है | पोटेशियम @ 100kg K2O/हेक्टेयर | संक्रमण में नाइट्रोजन 20% कम करें | मैग्नीशियम सल्फेट @ 15g/L मासिक",
        "fertilizer_mr": "कॅल्शियम @ 200kg/हेक्टर (CaNO3) — अनिवार्य; Stemphylium Ca च्या कमतरतेत वाढते | पोटॅशियम @ 100kg K2O/हेक्टर | संसर्गात नत्र 20% कमी करा | मॅग्नेशियम सल्फेट @ 15g/L मासिक",
        "irrigation_hi": "सक्रिय रोग में ओवरहेड सिंचाई बंद | केवल ड्रिप @ 65% क्षेत्र क्षमता | सुबह 6-8 बजे सिंचाई | टपकने वाले ड्रिपर तुरंत ठीक करें",
        "irrigation_mr": "सक्रिय रोगात ओव्हरहेड सिंचन कायमचे बंद | फक्त ठिबक @ 65% क्षेत्र क्षमता | सकाळी 6-8 वाजता सिंचन | गळणारे ड्रिपर ताबडतोब दुरुस्त करा",
        "recovery_hi": "Iprodione से 14-20 दिन में स्वस्थ वृद्धि।",
        "recovery_mr": "Iprodione ने 14-20 दिवसांत निरोगी वाढ.",
        "prevention_hi": "बीज उपचार: Iprodione @ 3g/kg | दिन 30 और 50 पर निवारक छिड़काव | सप्ताह 5 से धुरिया धब्बे जांचें | हवा की दिशा के विपरीत पंक्तियां लगाएं",
        "prevention_mr": "बीज प्रक्रिया: Iprodione @ 3g/kg | दिन 30 आणि 50 वर प्रतिबंधक फवारणी | आठवडा 5 पासून ठिपके तपासा | वाऱ्याच्या दिशेला लंब ओळी लावा",
        "organic_hi": "कॉपर सल्फेट 0.5% साप्ताहिक छिड़काव | नीम पत्ता काढ़ा 5% हर 10 दिन | गोमूत्र 5% + हल्दी 1% हर 7 दिन",
        "organic_mr": "कॉपर सल्फेट 0.5% साप्ताहिक फवारणी | कडुनिंबाची पाने काढा 5% दर 10 दिवस | गोमूत्र 5% + हळद 1% दर 7 दिवस",
        "cost_hi": "₹700-1,200/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹700-1,200/एकर प्रति फवारणी चक्र",
    },
    "virosis": {
        "severity": "CRITICAL", "pathogen": "Onion Yellow Dwarf Potyvirus (OYDV) — transmitted by Myzus persicae (aphid); causes yellow striping, leaf twisting & stunting; NO CURE — vector control is the only strategy.",
        "immediate": ["Remove ALL plants with yellow stripes or stunting IMMEDIATELY within 24h","Apply aphid systemic insecticide within 24h","Install yellow sticky traps @ 10/acre immediately","Do NOT use cuttings or sets from infected crop — ever","Alert neighbouring farmers — virus spreads via flying winged aphids"],
        "chemical": "APHID: Imidacloprid 17.8SL @ 0.5ml/L | ALTERNATE: Thiamethoxam 25WG @ 0.3g/L | SYSTEMIC: Fluonicamid 50WG @ 0.3g/L | OPTION 4: Pymetrozine 50WG @ 0.6g/L | ROTATE ALL insecticide groups every spray — aphid resistance is very high",
        "biological": "Lacewing (Chrysoperla carnea) @ 50,000 eggs/ha | Aphidoletes aphidimyza release | Beauveria bassiana @ 5g/L foliar — kills aphids | Neem oil 3% every 5 days during flush",
        "cultural": "No cure for infected plants — remove & burn only | Certified virus-indexed transplants only | No onion near garlic/leek | Rogue symptomatic plants within 48h — do not wait",
        "fertilizer": "Silicon @ 150kg/ha hardens leaf surface against aphid probing | Potassium silicate @ 5ml/L foliar every 14 days | No excess Nitrogen — soft growth attracts more aphids",
        "irrigation": "Overhead irrigation 2x/week physically dislodges aphids | Maintain good soil moisture — drought worsens aphid damage | Morning irrigation washes aphids off foliage",
        "recovery": "Virus: NO recovery possible. Aphid elimination: 5-7 days. New healthy growth: 14-21 days after vector control.",
        "prevention": "Mineral oil 1% on seedlings — blocks aphid settling | Mustard trap crop around border | Seed TX: Imidacloprid 70WS @ 7g/kg | Silver reflective mulch all season — reduces OYDV by 50-60%",
        "organic": "Garlic + chilli + soap spray 5% every 5 days | Neem oil 2% every 5 days | Tobacco decoction 3% spray | Marigold border rows",
        "cost": "Rs.1,500-2,500/acre (intensive vector management)",
        "pathogen_hi": "Onion Yellow Dwarf Potyvirus (OYDV) — Myzus persicae (एफिड) द्वारा फैलाया जाता है; पीली धारियां, पत्ती मुड़ना और बौनापन; कोई इलाज नहीं — केवल वेक्टर नियंत्रण।",
        "pathogen_mr": "Onion Yellow Dwarf Potyvirus (OYDV) — Myzus persicae (माव) द्वारे पसरतो; पिवळे पट्टे, पाने वळणे आणि खुजेपणा; कोणताही उपचार नाही — फक्त वेक्टर नियंत्रण.",
        "immediate_hi": ["24 घंटे में पीली धारी या बौने पौधे हटाएं", "24 घंटे में एफिड प्रणालीगत कीटनाशक डालें", "पीले चिपचिपे ट्रैप @ 10/एकड़ तुरंत लगाएं", "संक्रमित फसल से कटिंग/सेट का उपयोग कभी न करें", "पड़ोसी किसानों को सूचित करें — पंखदार एफिड से फैलता है"],
        "immediate_mr": ["24 तासांत पिवळे पट्टे किंवा खुजी झाडे काढा", "24 तासांत माव प्रणालीगत कीटकनाशक फवारा", "पिवळे चिकट सापळे @ 10/एकर ताबडतोब लावा", "बाधित पिकातील कटिंग/सेट कधीही वापरू नका", "शेजारच्या शेतकऱ्यांना कळवा — पंख असलेल्या माव्याने पसरतो"],
        "chemical_hi": "एफिड: Imidacloprid 17.8SL @ 0.5ml/L | वैकल्पिक: Thiamethoxam 25WG @ 0.3g/L | प्रणालीगत: Fluonicamid 50WG @ 0.3g/L | विकल्प 4: Pymetrozine 50WG @ 0.6g/L | सभी कीटनाशक समूह बदलें — एफिड में प्रतिरोध बहुत अधिक",
        "chemical_mr": "माव: Imidacloprid 17.8SL @ 0.5ml/L | पर्याय: Thiamethoxam 25WG @ 0.3g/L | प्रणालीगत: Fluonicamid 50WG @ 0.3g/L | पर्याय 4: Pymetrozine 50WG @ 0.6g/L | सर्व कीटकनाशक गट फिरवा — माव्यात प्रतिकार खूप जास्त",
        "biological_hi": "लेसविंग (Chrysoperla carnea) @ 50,000 अंडे/हेक्टेयर | Aphidoletes aphidimyza छोड़ें | Beauveria bassiana @ 5g/L — एफिड मारता है | नीम तेल 3% हर 5 दिन",
        "biological_mr": "लेसविंग (Chrysoperla carnea) @ 50,000 अंडी/हेक्टर | Aphidoletes aphidimyza सोडा | Beauveria bassiana @ 5g/L — माव मारते | निंबाचे तेल 3% दर 5 दिवस",
        "cultural_hi": "संक्रमित पौधे हटाएं और जलाएं | प्रमाणित वायरस-मुक्त रोपे | लहसुन/लीक के पास प्याज न लगाएं | 48 घंटे में लक्षण दिखते ही पौधे हटाएं",
        "cultural_mr": "बाधित झाडे काढा व जाळा | प्रमाणित विषाणूमुक्त रोपे | लसूण/लीकजवळ कांदा नाही | 48 तासांत लक्षणे दिसताच झाडे काढा",
        "fertilizer_hi": "सिलिकॉन @ 150kg/हेक्टेयर — एफिड के खिलाफ पत्ती की सतह कठोर | पोटेशियम सिलिकेट @ 5ml/L हर 14 दिन | अधिक नाइट्रोजन नहीं — मुलायम वृद्धि एफिड को आकर्षित करती है",
        "fertilizer_mr": "सिलिकॉन @ 150kg/हेक्टर — माव्याविरुद्ध पानांची पृष्ठभाग कडक | पोटॅशियम सिलिकेट @ 5ml/L दर 14 दिवस | जास्त नत्र नाही — मऊ वाढ माव्याला आकर्षित करते",
        "irrigation_hi": "सप्ताह में 2 बार ओवरहेड सिंचाई एफिड को शारीरिक रूप से हटाती है | मिट्टी की पर्याप्त नमी बनाए रखें — सूखे में एफिड का नुकसान अधिक | एफिड धोने के लिए सुबह सिंचाई",
        "irrigation_mr": "आठवड्यातून 2 वेळा ओव्हरहेड सिंचन माव्याला शारीरिकरित्या काढते | पुरेशी जमिनीची ओलावा राखा — दुष्काळात माव्याचे नुकसान जास्त | माव धुण्यासाठी सकाळी सिंचन",
        "recovery_hi": "वायरस: कोई रिकवरी नहीं। एफिड 5-7 दिन में खत्म। नई वृद्धि 14-21 दिन बाद।",
        "recovery_mr": "विषाणू: कोणतीही पुनर्प्राप्ती नाही. माव 5-7 दिवसांत संपतात. नवीन वाढ 14-21 दिवसांनंतर.",
        "prevention_hi": "रोपों पर खनिज तेल 1% — एफिड को पत्ती पर बसने से रोकता है | बॉर्डर पर सरसों की फसल | बीज उपचार: Imidacloprid 70WS @ 7g/kg | चांदी परावर्तक मल्च — OYDV 50-60% कम करता है",
        "prevention_mr": "रोपांवर खनिज तेल 1% — माव्याला पानावर बसण्यापासून रोखते | सीमेवर मोहरी पीक | बीज प्रक्रिया: Imidacloprid 70WS @ 7g/kg | चांदीचे परावर्तक आच्छादन — OYDV 50-60% कमी करते",
        "organic_hi": "लहसुन + मिर्च + साबुन 5% हर 5 दिन | नीम तेल 2% हर 5 दिन | तंबाकू काढ़ा 3% | गेंदा बॉर्डर पंक्तियां",
        "organic_mr": "लसूण + मिरची + साबण 5% दर 5 दिवस | निंबाचे तेल 2% दर 5 दिवस | तंबाखू काढा 3% | झेंडूच्या सीमा ओळी",
        "cost_hi": "₹1,500-2,500/एकड़ (गहन वेक्टर प्रबंधन)",
        "cost_mr": "₹1,500-2,500/एकर (सखोल वेक्टर व्यवस्थापन)",
    },
    "xanthomonas_blight": {
        "severity": "HIGH", "pathogen": "Xanthomonas axonopodis pv. allii (bacterium) — water-soaked streaks turning yellow/tan; spreads through water splash & tools; enters via wounds; thrives in warm wet conditions (25-30C).",
        "immediate": ["Spray copper oxychloride 50WP @ 3g/L IMMEDIATELY on detection","Stop ALL overhead irrigation — water directly spreads bacterial cells","Do not work in field when leaves are wet","Disinfect ALL tools with 10% bleach after each plant touched","Do not move plant material from affected to healthy areas"],
        "chemical": "PRIMARY: Copper oxychloride 50WP @ 3g/L | ALTERNATE: Copper hydroxide 77WP @ 2.5g/L | COMBO: Streptomycin sulfate @ 0.5g/L + Copper oxychloride | SYSTEMIC: Kasugamycin 3SL @ 2ml/L | INTERVAL: Every 5-7 days in wet conditions; 4-5 applications",
        "biological": "Bacillus subtilis @ 3ml/L (produces natural antibiotics against bacteria) | Pseudomonas fluorescens @ 5g/L foliar | Lysobacter enzymogenes biocontrol | Bacteriophage spray (experimental — highly effective)",
        "cultural": "Remove & burn ALL infected leaf material | No onion in waterlogged/poorly-drained areas | 2-year rotation without alliums | Avoid mechanical wounding — bacteria enter through wounds",
        "fertilizer": "Calcium @ 200kg/ha — strengthens cell walls against bacterial entry | Potassium @ 100kg K2O/ha for systemic resistance | No excess Nitrogen | Silica @ 100kg/ha",
        "irrigation": "Drip ONLY — overhead irrigation directly spreads bacterial cells | Irrigate early morning ONLY (before 8am) | Minimise irrigation 5 days after rain | Ensure free-draining field",
        "recovery": "10-15 days with copper bactericide under dry weather conditions.",
        "prevention": "Copper oxychloride seed TX @ 3g/kg | Avoid wounding during transplanting | Monitor for water-soaked leaf streaks from week 3 | Certified disease-free planting material only",
        "organic": "Bordeaux mixture 0.5% spray every 7 days | Buttermilk 10% spray every 5 days — lactic acid suppresses bacteria | Cow urine 5% + copper sulfate 0.1% combined spray",
        "cost": "Rs.800-1,500/acre per spray cycle",
        "pathogen_hi": "Xanthomonas axonopodis pv. allii (जीवाणु) — पानी में भीगे धब्बे पीले/हल्के भूरे हो जाते हैं; पानी के छींटे और औजारों से फैलता है; गर्म गीले मौसम (25-30°C) में पनपता है।",
        "pathogen_mr": "Xanthomonas axonopodis pv. allii (जीवाणू) — पाण्याने भिजलेले ठिपके पिवळे/फिकट तपकिरी होतात; पाण्याचे शिडकावे आणि अवजारांनी पसरते; उष्ण ओल्या हवामानात (25-30°C) वाढते.",
        "immediate_hi": ["तुरंत Copper oxychloride 50WP @ 3g/L का छिड़काव करें", "ओवरहेड सिंचाई तुरंत बंद करें — पानी सीधे जीवाणु फैलाता है", "गीली पत्तियों में खेत में काम न करें", "हर पौधे को छूने के बाद उपकरण 10% ब्लीच से साफ करें", "प्रभावित से स्वस्थ क्षेत्रों में पौधे सामग्री न ले जाएं"],
        "immediate_mr": ["ताबडतोब Copper oxychloride 50WP @ 3g/L फवारा", "ओव्हरहेड सिंचन ताबडतोब बंद करा — पाणी थेट जीवाणू पसरवते", "ओल्या पानांत शेतात काम करू नका", "प्रत्येक झाड स्पर्श केल्यानंतर अवजारे 10% ब्लीचने स्वच्छ करा", "बाधित ते निरोगी भागात रोपे सामग्री नेऊ नका"],
        "chemical_hi": "प्राथमिक: Copper oxychloride 50WP @ 3g/L | वैकल्पिक: Copper hydroxide 77WP @ 2.5g/L | संयोजन: Streptomycin sulfate @ 0.5g/L + Copper oxychloride | प्रणालीगत: Kasugamycin 3SL @ 2ml/L | अंतराल: गीले में 5-7 दिन; 4-5 छिड़काव",
        "chemical_mr": "प्राथमिक: Copper oxychloride 50WP @ 3g/L | पर्याय: Copper hydroxide 77WP @ 2.5g/L | संयोजन: Streptomycin sulfate @ 0.5g/L + Copper oxychloride | प्रणालीगत: Kasugamycin 3SL @ 2ml/L | अंतर: ओल्या हवामानात 5-7 दिवस; 4-5 फवारण्या",
        "biological_hi": "Bacillus subtilis @ 3ml/L (जीवाणु-रोधी एंटीबायोटिक बनाता है) | Pseudomonas fluorescens @ 5g/L पत्तेदार | Lysobacter enzymogenes जैव नियंत्रण | बैक्टीरियोफेज स्प्रे (प्रयोगात्मक — बहुत प्रभावी)",
        "biological_mr": "Bacillus subtilis @ 3ml/L (जीवाणू-विरोधी प्रतिजैविक तयार करते) | Pseudomonas fluorescens @ 5g/L पर्णीय | Lysobacter enzymogenes जैव नियंत्रण | बॅक्टेरियोफेज फवारणी (प्रायोगिक — अत्यंत प्रभावी)",
        "cultural_hi": "सभी संक्रमित पत्तियां जलाएं | जलभराव वाले क्षेत्रों में न लगाएं | 2 साल का फसल चक्र | रोपाई में घाव से बचें — जीवाणु घावों से प्रवेश करते हैं",
        "cultural_mr": "सर्व बाधित पाने जाळा | पाणी साचणाऱ्या भागात लावणी नाही | 2 वर्षांचे पीक फेरपालट | लावणीत जखम टाळा — जीवाणू जखमांद्वारे प्रवेश करतात",
        "fertilizer_hi": "कैल्शियम @ 200kg/हेक्टेयर — जीवाणु प्रवेश से सुरक्षा | पोटेशियम @ 100kg K2O/हेक्टेयर | अधिक नाइट्रोजन नहीं | सिलिका @ 100kg/हेक्टेयर",
        "fertilizer_mr": "कॅल्शियम @ 200kg/हेक्टर — जीवाणू प्रवेशापासून संरक्षण | पोटॅशियम @ 100kg K2O/हेक्टर | जास्त नत्र नाही | सिलिका @ 100kg/हेक्टर",
        "irrigation_hi": "केवल ड्रिप — ओवरहेड सिंचाई सीधे जीवाणु फैलाती है | केवल सुबह 8 बजे से पहले | बारिश के बाद 5 दिन सिंचाई कम करें | जल निकासी सुनिश्चित करें",
        "irrigation_mr": "फक्त ठिबक — ओव्हरहेड सिंचन थेट जीवाणू पसरवते | फक्त सकाळी 8 पूर्वी | पावसानंतर 5 दिवस सिंचन कमी करा | निचरा सुनिश्चित करा",
        "recovery_hi": "कोरे मौसम में Copper bactericide से 10-15 दिन।",
        "recovery_mr": "कोरड्या हवामानात Copper bactericide ने 10-15 दिवस.",
        "prevention_hi": "बीज उपचार: Copper oxychloride @ 3g/kg | रोपाई में घाव से बचें | सप्ताह 3 से पत्तियों पर पानी जैसे धब्बे जांचें | प्रमाणित रोग-मुक्त रोपण सामग्री",
        "prevention_mr": "बीज प्रक्रिया: Copper oxychloride @ 3g/kg | लावणीत जखम टाळा | आठवडा 3 पासून पानांवर पाणीदार ठिपके तपासा | प्रमाणित रोगमुक्त लागवड साहित्य",
        "organic_hi": "बोर्डो मिश्रण 0.5% हर 7 दिन | छाछ 10% छिड़काव हर 5 दिन — लैक्टिक एसिड जीवाणु रोकता है | गोमूत्र 5% + कॉपर सल्फेट 0.1% संयुक्त छिड़काव",
        "organic_mr": "बोर्डो मिश्रण 0.5% दर 7 दिवस | ताक 10% फवारणी दर 5 दिवस — लॅक्टिक ॲसिड जीवाणू रोखते | गोमूत्र 5% + कॉपर सल्फेट 0.1% एकत्र फवारणी",
        "cost_hi": "₹800-1,500/एकड़ प्रति छिड़काव चक्र",
        "cost_mr": "₹800-1,500/एकर प्रति फवारणी चक्र",
    },
}

# Alias keys for alternate CNN class name formats
_dk_aliases = {
    "alternaria_d": "alternaria",
    "botrytis_leaf_blight": "botrytis_blight",
    "bulb_blight_d": "bulb_blight",
    "bulb_blight-d": "bulb_blight",
    "caterpillar_p": "caterpillar",
    "caterpillar-p": "caterpillar",
    "iris_yellow_spot_virus": "iris_yellow_virus",
    "stemphylium_leaf_blight": "stemphylium_blight",
    "virosis_d": "virosis",
    "virosis-d": "virosis",
    "xanthomonas_leaf_blight": "xanthomonas_blight",
}
def _get_disease_advice(name: str, lang: str = "en"):
    k = name.lower().replace(" ", "_")
    k = _dk_aliases.get(k, k)
    base = _DISEASE_ADVICE.get(k, None)
    if base is None:
        return None
    # Build a language-resolved copy: prefer _hi/_mr keys, fall back to English
    suffix = "_hi" if lang == "hi" else "_mr" if lang == "mr" else ""
    if not suffix:
        return base
    resolved = {}
    for field in ("pathogen","immediate","chemical","biological","cultural",
                  "fertilizer","irrigation","recovery","prevention","organic","cost","severity"):
        translated_key = field + suffix
        resolved[field] = base.get(translated_key, base.get(field, "—"))
    return resolved


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

@st.cache_data(ttl=1800)  # cache 30 min — weather doesn't change that fast
def get_weather_forecast(lat: float = 20.0, lon: float = 74.0):
    """
    Fetch 7-day weather forecast from Open-Meteo (free, no API key).
    Defaults to Nashik region (major onion belt).
    Returns DataFrame with: day, temp_c, temp_min_c, rain_pct, humidity,
                             wind_kmh, condition, condition_code
    Falls back to synthetic data if network is unavailable.
    """
    try:
        import urllib.request, json as _json
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=weathercode,temperature_2m_max,temperature_2m_min,"
            f"precipitation_probability_max,windspeed_10m_max,relative_humidity_2m_max"
            f"&timezone=Asia%2FKolkata&forecast_days=7"
        )
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = _json.loads(resp.read())
        daily = data["daily"]

        # WMO weather code → emoji + short label
        def _wmo_icon(code):
            if code == 0:   return "☀️", "Clear"
            if code <= 2:   return "🌤", "Partly Cloudy"
            if code == 3:   return "☁️", "Overcast"
            if code <= 49:  return "🌫", "Foggy"
            if code <= 59:  return "🌦", "Drizzle"
            if code <= 69:  return "🌧", "Rain"
            if code <= 79:  return "🌨", "Snow"
            if code <= 84:  return "🌦", "Showers"
            if code <= 99:  return "⛈️", "Thunderstorm"
            return "🌡", "Unknown"

        rows = []
        for i, dt in enumerate(daily["time"]):
            d = datetime.strptime(dt, "%Y-%m-%d")
            code = daily["weathercode"][i] or 0
            icon, label = _wmo_icon(int(code))
            rows.append({
                "day":       d.strftime("%a %d %b"),
                "temp_c":    round(daily["temperature_2m_max"][i] or 0, 1),
                "temp_min_c":round(daily["temperature_2m_min"][i] or 0, 1),
                "rain_pct":  round(daily["precipitation_probability_max"][i] or 0, 1),
                "humidity":  int(daily["relative_humidity_2m_max"][i] or 0),
                "wind_kmh":  round(daily["windspeed_10m_max"][i] or 0, 1),
                "condition": f"{icon} {label}",
                "icon":      icon,
            })
        return pd.DataFrame(rows)

    except Exception:
        # Graceful fallback — synthetic but deterministic per day
        np.random.seed(datetime.now().timetuple().tm_yday)
        rows = []
        for i in range(7):
            d = datetime.now() + timedelta(days=i)
            rows.append({
                "day":       d.strftime("%a %d %b"),
                "temp_c":    round(np.random.uniform(24, 36), 1),
                "temp_min_c":round(np.random.uniform(16, 24), 1),
                "rain_pct":  round(np.random.uniform(0, 80), 1),
                "humidity":  int(np.random.uniform(40, 90)),
                "wind_kmh":  round(np.random.uniform(5, 30), 1),
                "condition": "🌤 Partly Cloudy",
                "icon":      "🌤",
            })
        return pd.DataFrame(rows)

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
    st_autorefresh(interval=120000, key="news_refresh")  # refresh every 2 min for live news

    # ── Load resources ────────────────────────────────────────────────────
    model, features = load_model_and_features()
    datasets = load_all_datasets()
    monthly_df = datasets["monthly"]

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
            options=["Admin", "Farmer", "Disease Prevention"],
            format_func=lambda x: T["admin"] if x == "Admin" else T["farmer"] if x == "Farmer" else T["disease_prevention"],
            label_visibility="collapsed",
            key="role_select"
        )
    with ctrl_c3:
        current_month = datetime.now().month
        current_year = datetime.now().year

    T = TRANSLATIONS[lang]

    # ── LOCATION / WEATHER ────────────────────────────────────────────────
    _sel_loc = "Nashik"
    _lat, _lon = 20.00, 73.78
    weather_df = get_weather_forecast(_lat, _lon)

    # ── HEADER ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ric-header">
        <div class="corner-mark">AGRI · INTEL · v2.0</div>
        <h1>{T['app_title']}</h1>
        <p>{T['app_subtitle']}</p>
    </div>`
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SIDEBAR — WEATHER
    # ════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(f"""
        <div class="panel-title">☁ {T['weather_forecast']}
            <span style="font-size:0.55rem;color:var(--sage);margin-left:8px;">
                {_sel_loc} · Open-Meteo
            </span>
        </div>
        """, unsafe_allow_html=True)

        for _, row in weather_df.iterrows():
            rain_color = "#4A7C59" if row['rain_pct'] < 40 else "#E67E22" if row['rain_pct'] < 70 else "#E74C3C"
            st.markdown(f"""
            <div class="weather-row" style="flex-direction:column;align-items:flex-start;gap:2px;padding:6px 0;border-bottom:1px solid rgba(139,115,60,0.15);">
                <div style="display:flex;justify-content:space-between;width:100%;align-items:center;">
                    <span class="weather-day">{row['condition']} &nbsp;<b>{row['day']}</b></span>
                </div>
                <div style="display:flex;gap:10px;font-size:0.72rem;margin-top:3px;flex-wrap:wrap;">
                    <span style="color:var(--harvest);">🌡 {row['temp_min_c']}–{row['temp_c']}°C</span>
                    <span style="color:{rain_color};">🌧 {row['rain_pct']:.0f}%</span>
                    <span style="color:var(--straw);">💧 {row['humidity']}%</span>
                    <span style="color:var(--straw);">💨 {row['wind_kmh']} km/h</span>
                </div>
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
            f"📰 {'कृषि समाचार' if lang=='hi' else 'कृषी बातम्या' if lang=='mr' else 'Agri News'}",
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

        # TAB 6: AGRI NEWS FEED — live Google RSS
        # ═══════════════════════════════════════════════════════════════
        with tab5:
            st.markdown(f"""<div class="panel-title">📰 {'कृषि समाचार' if lang=='hi' else 'कृषी बातम्या' if lang=='mr' else 'Agricultural News'}
                <span style="font-size:0.6rem;color:var(--sage);margin-left:12px;">
                    ↻ {'दर 2 मिनिटांनी अपडेट' if lang=='mr' else 'हर 2 मिनट में अपडेट' if lang=='hi' else 'Live · updates every 2 min'}
                </span>
            </div>""", unsafe_allow_html=True)

            _farmer_news = get_agri_news(lang)

            if not _farmer_news:
                st.markdown(f"""
                <div style="text-align:center;padding:32px;border:1px dashed var(--bark);border-radius:8px;color:var(--straw);">
                    <div style="font-size:2rem;margin-bottom:8px;">📰</div>
                    <div style="font-family:'Cinzel',serif;font-size:0.82rem;color:var(--harvest);">
                        {'सध्या बातम्या उपलब्ध नाहीत. इंटरनेट कनेक्शन तपासा.' if lang=='mr'
                         else 'अभी समाचार उपलब्ध नहीं। इंटरनेट कनेक्शन जांचें।' if lang=='hi'
                         else 'No news available right now. Check your internet connection.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                _fnc1, _fnc2 = st.columns(2)
                for _fi, _fitem in enumerate(_farmer_news):
                    _fcol = _fnc1 if _fi % 2 == 0 else _fnc2
                    with _fcol:
                        _fpub = _fitem['published'][:16] if _fitem['published'] else ""
                        st.markdown(f"""
                        <div class="news-card">
                            <div class="news-title">{_fitem['title']}</div>
                            <div class="news-body" style="margin-top:6px;font-size:0.78rem;color:var(--straw);
                                                          display:-webkit-box;-webkit-line-clamp:3;
                                                          -webkit-box-orient:vertical;overflow:hidden;">
                                {_fitem['body']}
                            </div>
                            <div style="margin-top:8px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;">
                                <span class="news-tag">{_fitem['tag']}</span>
                                <span style="font-size:0.65rem;color:var(--bark);">{_fpub}</span>
                            </div>
                            <div style="margin-top:8px;">
                                <a href="{_fitem['link']}" target="_blank"
                                   style="font-size:0.72rem;color:var(--sage);text-decoration:none;font-family:'Cinzel',serif;letter-spacing:0.05em;">
                                    {'अधिक वाचा →' if lang=='mr' else 'और पढ़ें →' if lang=='hi' else 'Read More →'}
                                </a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Contact info footer
        st.markdown("<hr>", unsafe_allow_html=True)
        st.info("""
**📞 Need Help? / मदद चाहिए? / मदत हवी आहे?**

Contact your local Agricultural Extension Officer · अपने स्थानीय कृषि विस्तार अधिकारी से संपर्क करें · स्थानिक कृषी विस्तार अधिकाऱ्यांशी संपर्क साधा

📞 Toll-Free: 1800-180-1551 (Kisan Call Centre)
        """)

        return


    # ════════════════════════════════════════════════════════════════════
    # DISEASE PREVENTION VIEW
    # ════════════════════════════════════════════════════════════════════
    elif role == "Disease Prevention":
        st.markdown(f"""
        <div class="ric-header" style="margin-bottom:16px;">
            <h1 style="font-size:1.4rem;">🛡 {'रोग रोकथाम' if lang=='hi' else 'रोग प्रतिबंध' if lang=='mr' else 'Disease Prevention'}</h1>
            <p>{'CNN फसल स्कैन | रोग पूर्वानुमान | पूर्ण मौसम अनुमान' if lang=='hi' else 'CNN पीक स्कॅन | रोग अंदाज | संपूर्ण हंगाम पूर्वसूचना' if lang=='mr' else 'CNN Crop Scan | Disease Risk Forecast | Full-Season Prediction'}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<div class="panel-title">🛡 {'पूर्व रोग रोकथाम' if lang=='hi' else 'पूर्व रोग प्रतिबंध' if lang=='mr' else 'Predictive Disease Prevention'}</div>""",
                    unsafe_allow_html=True)

        # ── SECTION 1: MULTI-IMAGE UPLOAD & CNN SCAN ──────────────
        st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
            color:var(--straw);margin:0 0 8px;text-transform:uppercase;">
            🔬 {'फसल छवि से रोग पहचान' if lang=='hi' else 'पीक प्रतिमेतून रोग ओळख' if lang=='mr' else 'Crop Image Disease Scan'}
        </div>""", unsafe_allow_html=True)

        _dis_tl_path = os.path.join(BASE_DIR, "onion_disease_timeline.csv")
        _dis_timeline = _load_onion_disease_timeline(_dis_tl_path)

        _scan_model_path = (
            os.path.join(BASE_DIR, "best_disease_model.h5")
            if os.path.exists(os.path.join(BASE_DIR, "best_disease_model.h5"))
            else os.path.join(BASE_DIR, "onion_disease_model.h5")
        )
        _scan_cls_path = os.path.join(BASE_DIR, "class_indices.pkl")
        _scan_model_ready = _TF_AVAILABLE and _PIL_AVAILABLE and os.path.exists(_scan_model_path)

        # ── Plant image validator ─────────────────────────────────────────────
        # Strategy: check for biological pixel content (green/brown/white plant
        # tissue) in HSV space.  Onion images may be mostly green leaves OR
        # pale/white/purple bulbs — both are accepted.  Pure non-plant images
        # (selfies, screenshots, landscapes without crops) will fail.
        #
        # Two complementary checks:
        #   1. Green pixel ratio  ≥ 3%   (leaves, stems)
        #   2. Plant-tissue ratio ≥ 12%  (bulbs: pale, white, cream, purple hues)
        # Either check passing = image is treated as a plant photo.
        # Model confidence must still be ≥ 45% (softer — the content filter does
        # the heavy lifting; the confidence gate only catches total model confusion).

        _SCAN_CONF_THRESHOLD = 45.0   # minimum model confidence to accept result

        def _is_plant_image(pil_img: "_PILImage.Image") -> "tuple[bool, str]":
            """
            Returns (is_plant: bool, reason: str).
            Uses HSV colour analysis to detect green/plant-tissue pixels.
            Works for both leafy (green) and bulb (pale/purple/white) onion images.
            """
            import colorsys
            img_rgb = pil_img.convert("RGB").resize((100, 100))  # fast downsample
            pixels  = list(img_rgb.getdata())
            total   = len(pixels)

            green_count  = 0   # leaf-green pixels
            tissue_count = 0   # bulb/tissue pixels (pale, cream, white, light-purple)

            for r, g, b in pixels:
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                h_deg = h * 360

                # Green leaf pixels: hue 60–165°, reasonable saturation & brightness
                if 60 <= h_deg <= 165 and s > 0.15 and v > 0.15:
                    green_count += 1

                # Bulb / plant-tissue pixels:
                #   • Pale/cream/white   : low saturation, medium-high brightness
                #   • Light purple/pink  : hue 270–340°
                #   • Yellowish-brown    : hue 20–60°, low saturation (dry outer skin)
                elif (s < 0.35 and v > 0.40):                        # pale/white tissue
                    tissue_count += 1
                elif (270 <= h_deg <= 340 and s > 0.10 and v > 0.25):  # purple/pink bulb
                    tissue_count += 1
                elif (20 <= h_deg < 60 and s < 0.55 and v > 0.25):    # tan/brown skin
                    tissue_count += 1

            green_ratio  = green_count  / total * 100
            tissue_ratio = tissue_count / total * 100

            if green_ratio >= 3.0:
                return True, f"green={green_ratio:.1f}%"
            if tissue_ratio >= 12.0:
                return True, f"tissue={tissue_ratio:.1f}%"
            return False, f"green={green_ratio:.1f}%, tissue={tissue_ratio:.1f}% (too low)"

        if not _scan_model_ready:
            st.warning(f"""⚠️ {'CNN मॉडल उपलब्ध नहीं। "मॉडल प्रशिक्षण" टैब में ट्रेन करें।' if lang=='hi'
                          else 'CNN मॉडेल उपलब्ध नाही. "मॉडेल प्रशिक्षण" टॅबमधून प्रशिक्षण द्या.' if lang=='mr'
                          else 'CNN model not found. Train it first using the 🏋️ Train Disease Model tab.'}""")
        else:
            _scan_model = _load_cnn_disease_model(_scan_model_path)
            if os.path.exists(_scan_cls_path):
                with open(_scan_cls_path, "rb") as _scf:
                    _scan_cls_map = {v: k for k, v in pickle.load(_scf).items()}
            else:
                _scan_cls_map = {i: c for i, c in enumerate(_DISEASE_CLASSES)}

            st.markdown(f"""
            <div style="padding:10px 14px;background:rgba(74,124,89,0.08);border:1px solid var(--leaf);
                        border-radius:4px;font-size:0.84rem;color:var(--cream);margin-bottom:10px;">
                {'कैमरे से तस्वीर लें या गैलरी से अपलोड करें — AI रोग पहचानेगा।' if lang=='hi'
                 else 'कॅमेऱ्याने फोटो काढा किंवा गॅलरीतून अपलोड करा — AI रोग ओळखेल.' if lang=='mr'
                 else '📷 Take a photo with your camera or upload from gallery — AI detects disease instantly.'}
            </div>
            """, unsafe_allow_html=True)

            # ── INPUT MODE: Camera or Upload ──────────────────────────
            _input_mode = st.radio(
                f"{'इनपुट विधि' if lang=='hi' else 'इनपुट पद्धत' if lang=='mr' else 'Input Method'}",
                options=["📷 Camera", "🖼️ Upload"],
                horizontal=True,
                key="disease_input_mode",
                label_visibility="collapsed"
            )

            _all_images = []  # list of (PIL.Image, name) tuples to scan

            if _input_mode == "📷 Camera":
                st.markdown(f"""
                <div style="font-size:0.8rem;color:var(--straw);margin-bottom:6px;">
                    {'📱 अपने फोन/लैपटॉप का कैमरा खुलेगा — पत्ती या बल्ब पर फोकस करें और फोटो लें।' if lang=='hi'
                     else '📱 तुमचा फोन/लॅपटॉप कॅमेरा उघडेल — पान किंवा बल्बवर फोकस करा आणि फोटो काढा.' if lang=='mr'
                     else '📱 Your device camera will activate — point at the leaf or bulb and capture.'}
                </div>
                """, unsafe_allow_html=True)
                _cam_img = st.camera_input(
                    f"{'फोटो लें' if lang=='hi' else 'फोटो काढा' if lang=='mr' else 'Take Photo'}",
                    key="disease_camera_input",
                    label_visibility="collapsed"
                )
                if _cam_img:
                    _cam_pil = _PILImage.open(_cam_img)
                    _c1, _c2 = st.columns([1, 2])
                    with _c1:
                        st.image(_cam_pil, width=220)
                    with _c2:
                        st.markdown(f"""
                        <div style="background:rgba(231,76,60,0.12);border-left:5px solid #e74c3c;
                                    padding:16px 20px;border-radius:8px;margin-top:8px;">
                            <div style="font-family:'Cinzel',serif;color:#e74c3c;font-size:0.95rem;margin-bottom:8px;">
                                ❌ {'अमान्य प्रतिमा' if lang=='mr' else 'अमान्य छवि' if lang=='hi' else 'Invalid Image'}
                            </div>
                            <div style="font-size:0.84rem;color:var(--cream);margin-bottom:8px;">
                                {'ही प्रतिमा ओळखता येत नाही. कृपया कांद्याचे पान किंवा बल्बचा स्पष्ट फोटो काढा.' if lang=='mr'
                                 else 'यह छवि पहचानने योग्य नहीं है। कृपया प्याज की पत्ती या बल्ब का स्पष्ट फोटो लें।' if lang=='hi'
                                 else 'This image is not recognisable. Please take a clear photo of an onion leaf or bulb.'}
                            </div>
                            <div style="font-size:0.78rem;color:var(--straw);">
                                {'💡 टिप: पान किंवा बल्ब चांगल्या प्रकाशात कॅमेऱ्याच्या जवळ धरा.' if lang=='mr'
                                 else '💡 टिप: पत्ती या बल्ब को अच्छी रोशनी में कैमरे के करीब रखें।' if lang=='hi'
                                 else '💡 Tip: Hold the leaf or bulb close to the camera in good light.'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            else:
                _uploaded_imgs = st.file_uploader(
                    f"{'छवियां अपलोड करें' if lang=='hi' else 'प्रतिमा अपलोड करा' if lang=='mr' else 'Upload Crop Images'}",
                    type=["jpg","jpeg","png","bmp","tiff","webp"],
                    accept_multiple_files=True,
                    key="tab5_img_uploader",
                    label_visibility="collapsed"
                )
                if _uploaded_imgs:
                    _all_images = [(_PILImage.open(f), f.name) for f in _uploaded_imgs]

            if _all_images:
                st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                    color:var(--straw);margin:12px 0 8px;text-transform:uppercase;">
                    🔬 {'विश्लेषण परिणाम — ' + str(len(_all_images)) + ' छवि' if lang=='hi'
                         else 'विश्लेषण निकाल — ' + str(len(_all_images)) + ' प्रतिमा' if lang=='mr'
                         else f'Scan Results — {len(_all_images)} Image{"s" if len(_all_images)>1 else ""}'}
                </div>""", unsafe_allow_html=True)

                for _pil, _img_name in _all_images:
                    _inp = _preprocess_disease_image(_pil)
                    _preds = _scan_model.predict(_inp, verbose=0)[0]
                    _pred_idx = int(np.argmax(_preds))
                    _conf = float(_preds[_pred_idx]) * 100
                    _disease_name = _scan_cls_map.get(_pred_idx, f"class_{_pred_idx}")
                    _top3_idx = np.argsort(_preds)[::-1][:3]
                    _top3 = [(_scan_cls_map.get(i, f"class_{i}").replace("_"," ").title(), float(_preds[i])*100) for i in _top3_idx]

                    _top2_conf = float(np.sort(_preds)[-2]) * 100
                    _is_plant, _plant_reason = _is_plant_image(_pil)
                    _rejected = (not _is_plant) or (_conf < _SCAN_CONF_THRESHOLD)

                    _ic1, _ic2 = st.columns([1, 2])
                    with _ic1:
                        st.image(_pil, caption=_img_name, width=220)
                    with _ic2:
                        if _rejected:
                            # Reject — not a recognisable crop image
                            st.markdown(f"""
                            <div style="background:rgba(243,156,18,0.12);border-left:5px solid #f39c12;
                                        padding:14px 18px;border-radius:8px;">
                                <div style="font-family:'Cinzel',serif;color:#f39c12;font-size:0.9rem;margin-bottom:6px;">
                                    ⚠️ {'वैध फसल छवि नहीं' if lang=='hi' else 'वैध पीक प्रतिमा नाही' if lang=='mr' else 'Not a Recognisable Crop Image'}
                                </div>
                                <div style="font-size:0.82rem;color:var(--cream);margin-bottom:6px;">
                                    {'अधिकतम विश्वास' if lang in ['hi','mr'] else 'Highest confidence'}:
                                    <b style="color:#f39c12;">{_conf:.1f}%</b>
                                    {' — पौधे की छवि नहीं लगती (हरे/ऊतक पिक्सेल नहीं मिले)' if lang=='hi'
                                     else ' — वनस्पती प्रतिमा नाही (हिरवे/ऊतक पिक्सेल आढळले नाहीत)' if lang=='mr'
                                     else f' — no plant pixels detected ({_plant_reason})'}
                                </div>
                                <div style="font-size:0.8rem;color:var(--straw);">
                                    {'कृपया कांदा पत्ती या बल्ब की स्पष्ट फोटो अपलोड करें।' if lang=='hi'
                                     else 'कृपया कांद्याचे पान किंवा बल्बचा स्पष्ट फोटो अपलोड करा.' if lang=='mr'
                                     else 'Please upload a clear photo of an onion leaf or bulb.'}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            _adv = _get_disease_advice(_disease_name, lang)
                            _det_color = "#2ecc71" if "healthy" in _disease_name.lower() else "#e74c3c"
                            _sev_pal = {"NONE":"#2ecc71","LOW":"#27ae60","MEDIUM":"#f39c12","HIGH":"#e67e22","CRITICAL":"#e74c3c"}

                            if _adv:
                                _sev = _adv.get("severity","—")
                                _sc  = _sev_pal.get(_sev, "#aaa")
                                # ── HEADER ──────────────────────────────
                                st.markdown(f"""
                                <div style="background:{_det_color}12;border-left:5px solid {_det_color};padding:12px 16px;border-radius:8px;margin-bottom:8px;">
                                  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
                                    <div>
                                      <div style="font-family:'Cinzel',serif;color:{_det_color};font-size:0.92rem;margin-bottom:3px;">
                                        🔬 {'पहचाना गया' if lang=='hi' else 'आढळले' if lang=='mr' else 'Detected'}: <b>{_disease_name.replace('_',' ').title()}</b>
                                      </div>
                                      <div style="font-size:0.74rem;color:var(--straw);">{_adv.get('pathogen','')}</div>
                                    </div>
                                    <div style="display:flex;gap:10px;align-items:center;">
                                      <div style="text-align:center;">
                                        <div style="font-size:0.58rem;color:var(--straw);text-transform:uppercase;">{'विश्वास' if lang in ['hi','mr'] else 'Confidence'}</div>
                                        <div style="color:{_det_color};font-family:'Cinzel',serif;font-size:1.05rem;font-weight:bold;">{_conf:.1f}%</div>
                                      </div>
                                      <div style="text-align:center;">
                                        <div style="font-size:0.58rem;color:var(--straw);text-transform:uppercase;">{'गंभीरता' if lang in ['hi','mr'] else 'Severity'}</div>
                                        <div style="background:{_sc};color:#fff;font-size:0.68rem;padding:2px 9px;border-radius:20px;font-weight:bold;">{_sev}</div>
                                      </div>
                                    </div>
                                  </div>
                                </div>""", unsafe_allow_html=True)
                                # ── IMMEDIATE ACTIONS ────────────────────
                                _steps_html = "".join([f"<div style='padding:3px 0;font-size:0.79rem;color:var(--cream);border-bottom:1px solid rgba(255,255,255,0.05);'><span style='color:#e74c3c;margin-right:5px;'>▶</span>{s}</div>" for s in _adv.get('immediate',[])])
                                st.markdown(f"""
                                <div style="background:rgba(231,76,60,0.07);border:1px solid rgba(231,76,60,0.28);border-radius:6px;padding:10px 13px;margin-bottom:8px;">
                                  <div style="font-family:'Cinzel',serif;font-size:0.68rem;color:#e74c3c;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:7px;">
                                    🚨 {'तुरंत करें — पहले 24 घंटे' if lang=='hi' else 'तातडीने करा — पहिले 24 तास' if lang=='mr' else 'Immediate Actions — First 24 Hours'}
                                  </div>{_steps_html}
                                </div>""", unsafe_allow_html=True)
                                # ── TREATMENT TABS ───────────────────────
                                _tb1,_tb2,_tb3 = st.tabs([
                                    "💊 "+("रासायनिक" if lang in ["hi","mr"] else "Chemical"),
                                    "🌱 "+("जैविक" if lang in ["hi","mr"] else "Biological"),
                                    "🌾 "+("सांस्कृतिक" if lang in ["hi","mr"] else "Cultural"),
                                ])
                                with _tb1:
                                    st.markdown(f"<div style='padding:8px;font-size:0.8rem;color:var(--cream);line-height:1.75;white-space:pre-line;'>{_adv.get('chemical','—')}</div>",unsafe_allow_html=True)
                                with _tb2:
                                    st.markdown(f"<div style='padding:8px;font-size:0.8rem;color:var(--cream);line-height:1.75;'>{_adv.get('biological','—')}</div>",unsafe_allow_html=True)
                                with _tb3:
                                    st.markdown(f"<div style='padding:8px;font-size:0.8rem;color:var(--cream);line-height:1.75;'>{_adv.get('cultural','—')}</div>",unsafe_allow_html=True)
                                # ── DETAIL ROW ───────────────────────────
                                _c1,_c2,_c3 = st.columns(3)
                                with _c1:
                                    st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid var(--bark);border-radius:5px;padding:9px 11px;">
                                      <div style="font-size:0.6rem;color:var(--sage);text-transform:uppercase;margin-bottom:5px;">💧 {'सिंचाई' if lang=='hi' else 'सिंचन' if lang=='mr' else 'Irrigation'}</div>
                                      <div style="font-size:0.77rem;color:var(--cream);line-height:1.55;">{_adv.get('irrigation','—')}</div></div>""",unsafe_allow_html=True)
                                with _c2:
                                    st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid var(--bark);border-radius:5px;padding:9px 11px;">
                                      <div style="font-size:0.6rem;color:var(--sage);text-transform:uppercase;margin-bottom:5px;">🌿 {'उर्वरक' if lang=='hi' else 'खत' if lang=='mr' else 'Fertilizer'}</div>
                                      <div style="font-size:0.77rem;color:var(--cream);line-height:1.55;">{_adv.get('fertilizer','—')}</div></div>""",unsafe_allow_html=True)
                                with _c3:
                                    st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid var(--bark);border-radius:5px;padding:9px 11px;">
                                      <div style="font-size:0.6rem;color:var(--sage);text-transform:uppercase;margin-bottom:5px;">⏱ {'रिकवरी' if lang in ['hi','mr'] else 'Recovery'}</div>
                                      <div style="font-size:0.77rem;color:var(--cream);line-height:1.55;">{_adv.get('recovery','—')}</div></div>""",unsafe_allow_html=True)
                                # ── ORGANIC + COST ───────────────────────
                                _d1,_d2 = st.columns([2,1])
                                with _d1:
                                    st.markdown(f"""<div style="background:rgba(74,124,89,0.07);border:1px solid var(--leaf);border-radius:5px;padding:9px 11px;margin-top:7px;">
                                      <div style="font-size:0.6rem;color:var(--sage);text-transform:uppercase;margin-bottom:5px;">🇮🇳 {'भारतीय जैविक उपाय' if lang=='hi' else 'भारतीय सेंद्रिय उपाय' if lang=='mr' else 'India Organic Remedies'}</div>
                                      <div style="font-size:0.77rem;color:var(--cream);line-height:1.55;">{_adv.get('organic','—')}</div></div>""",unsafe_allow_html=True)
                                with _d2:
                                    st.markdown(f"""<div style="background:rgba(243,156,18,0.07);border:1px solid rgba(243,156,18,0.3);border-radius:5px;padding:9px 11px;margin-top:7px;text-align:center;">
                                      <div style="font-size:0.6rem;color:#f39c12;text-transform:uppercase;margin-bottom:5px;">💰 {'लागत' if lang in ['hi','mr'] else 'Est. Cost'}</div>
                                      <div style="font-size:0.85rem;color:var(--harvest);font-family:'Cinzel',serif;font-weight:bold;">{_adv.get('cost','—')}</div></div>""",unsafe_allow_html=True)
                                # ── PREVENTION ───────────────────────────
                                st.markdown(f"""
                                <div style="background:rgba(255,255,255,0.02);border:1px solid var(--bark);border-radius:5px;padding:9px 13px;margin-top:7px;">
                                  <div style="font-size:0.6rem;color:var(--straw);text-transform:uppercase;margin-bottom:4px;">🛡 {'रोकथाम' if lang=='hi' else 'प्रतिबंध' if lang=='mr' else 'Prevention Strategy'}</div>
                                  <div style="font-size:0.77rem;color:var(--cream);line-height:1.55;">{_adv.get('prevention','—')}</div>
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background:{_det_color}18;border-left:5px solid {_det_color};padding:12px 16px;border-radius:8px;">
                                  <div style="font-family:'Cinzel',serif;color:{_det_color};font-size:0.9rem;margin-bottom:5px;">
                                    🔬 {'पहचाना गया' if lang=='hi' else 'आढळले' if lang=='mr' else 'Detected'}: <b>{_disease_name.replace('_',' ').title()}</b>
                                  </div>
                                  <div style="font-size:0.82rem;color:var(--cream);">✅ Advisory: Consult an agronomist.</div>
                                </div>""", unsafe_allow_html=True)
                            # ── TOP 3 ────────────────────────────────────
                            st.markdown(f"""
                            <div style="margin-top:7px;font-size:0.74rem;color:var(--straw);">
                                <b>{'शीर्ष 3' if lang in ['hi','mr'] else 'Top 3'}:</b>
                                {'  ·  '.join([f"<span style='color:var(--cream);'>{n}</span> {p:.1f}%" for n,p in _top3])}
                            </div>""", unsafe_allow_html=True)

                    st.markdown("<hr style='margin:8px 0;border-color:var(--bark);'>", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style="text-align:center;padding:28px;border:1px dashed var(--bark);border-radius:8px;color:var(--straw);">
                    <div style="font-size:2rem;margin-bottom:8px;">🌿</div>
                    <div style="font-family:'Cinzel',serif;font-size:0.82rem;color:var(--harvest);">
                        {'कैमरे से फोटो लें या छवियां अपलोड करें' if lang=='hi'
                         else 'कॅमेऱ्याने फोटो काढा किंवा प्रतिमा अपलोड करा' if lang=='mr'
                         else 'Take a photo with your camera or upload images above'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='margin:14px 0;'>", unsafe_allow_html=True)

        # ── SECTION 2: SOWING DATE + FORECAST INPUTS ──────────────
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
            🌤️ {'वर्तमान मौसम' if lang=='hi' else 'सध्याचे हवामान' if lang=='mr' else 'Current Weather (from 7-day forecast)'}
        </div>""", unsafe_allow_html=True)

        _wc1, _wc2, _wc3, _wc4 = st.columns(4)
        with _wc1:
            _prev_avg_temp = float(weather_df['temp_c'].mean())
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">🌡️ {'तापमान' if lang in ['hi','mr'] else 'Avg Max Temp'}</div>
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
            _prev_avg_humid = float(weather_df['humidity'].mean())
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">💧 {'आर्द्रता' if lang in ['hi','mr'] else 'Humidity'}</div>
                <div class="metric-value" style="font-size:1.1rem;">{_prev_avg_humid:.0f}%</div>
                <div class="metric-sub">{'Open-Meteo लाइव' if lang=='hi' else 'Open-Meteo लाइव्ह' if lang=='mr' else 'Live · Open-Meteo'}</div>
            </div>""", unsafe_allow_html=True)
        with _wc4:
            _prev_avg_wind = float(weather_df['wind_kmh'].mean())
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">💨 {'हवा' if lang in ['hi','mr'] else 'Wind Speed'}</div>
                <div class="metric-value" style="font-size:1.1rem;">{_prev_avg_wind:.0f}</div>
                <div class="metric-sub">km/h {'औसत' if lang in ['hi','mr'] else 'avg'}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # ── RISK SCORING FUNCTION ─────────────────────────────────
        def _calc_risk_score_at(row, crop_age, avg_temp, avg_humid, avg_rain):
            base = 0.3
            h_w = (0.25 if (avg_humid > 70 and row["humidity_factor"] == "high")
                   else 0.10 if (avg_humid > 50 and row["humidity_factor"] in ["high","moderate"]) else 0.0)
            r_w = (0.20 if (avg_rain > 60 and row["rain_factor"] == "high")
                   else 0.10 if (avg_rain > 40 and row["rain_factor"] in ["high","moderate"]) else 0.0)
            t_w = (0.20 if (avg_temp > 32 and row["temp_factor"] in ["hot","warm"])
                   else 0.10 if (avg_temp > 25 and row["temp_factor"] == "warm")
                   else 0.05 if (avg_temp < 20 and row["temp_factor"] == "cool") else 0.0)
            _mid = (row["day_start"] + row["day_end"]) / 2
            _prox = max(0, 1 - abs(crop_age - _mid) / max((row["day_end"] - row["day_start"]) / 2, 1))
            return min(base + h_w + r_w + t_w + 0.15 * _prox, 1.0)

        if st.button(
            f"🔍 {'रोग जोखिम विश्लेषण + पूर्ण मौसम अनुमान' if lang=='hi' else 'रोग जोखीम विश्लेषण + संपूर्ण हंगाम अंदाज' if lang=='mr' else 'Analyse Disease Risk + Full Season Forecast'}",
            key="btn_farmer_disease_risk", use_container_width=True
        ):
            if _prev_crop_age < 0:
                st.error("❌ Sowing date cannot be in the future.")
            else:
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
                        ⚠️ <b>{'सक्रिय खिड़कियां' if lang in ['hi','mr'] else 'Active Windows Now'}:</b> {len(_active)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                    color:var(--straw);margin:4px 0 10px;text-transform:uppercase;">
                    🔴 {'आज के सक्रिय रोग' if lang=='hi' else 'आजचे सक्रिय रोग' if lang=='mr' else f'Active Diseases — Today (Day {_prev_crop_age})'}
                </div>""", unsafe_allow_html=True)

                if _active.empty:
                    st.success(f"✅ {'आज कोई सक्रिय रोग नहीं।' if lang=='hi' else 'आज कोणताही सक्रिय रोग नाही.' if lang=='mr' else f'No active disease windows at day {_prev_crop_age}. Crop is in a safe zone.'}")
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
                        if _prev_avg_humid > 70 and _drow["humidity_factor"] == "high": _dreasons.append("High humidity matches trigger")
                        if _prev_avg_rain > 60 and _drow["rain_factor"] == "high": _dreasons.append("High rainfall probability")
                        if _prev_avg_temp > 32 and _drow["temp_factor"] in ["hot","warm"]: _dreasons.append("Temperature in risk zone")
                        _dmid = (_drow["day_start"] + _drow["day_end"]) / 2
                        if abs(_prev_crop_age - _dmid) < 10: _dreasons.append("At peak risk window")
                        _dreason_text = " · ".join(_dreasons) if _dreasons else "Crop age within disease window"
                        st.markdown(f"""
                        <div style="background:{_dcolor}18;border-left:5px solid {_dcolor};
                                    padding:14px 18px;border-radius:8px;margin:8px 0;">
                            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                                <h4 style="margin:0;color:{_dcolor};font-family:'Cinzel',serif;">{_demoji} {_dname}</h4>
                                <span style="background:{_dcolor};color:white;padding:3px 10px;border-radius:10px;font-weight:bold;font-size:12px;">{_dlevel} — {_score*100:.0f}%</span>
                            </div>
                            <p style="margin:6px 0 3px;font-size:0.75rem;color:var(--straw);">
                                <b>📆 {'विंडो' if lang in ['hi','mr'] else 'Window'}:</b> Day {int(_drow['day_start'])}–{int(_drow['day_end'])}
                                &nbsp;|&nbsp; <b>🔎 {'कारण' if lang in ['hi','mr'] else 'Reason'}:</b> {_dreason_text}
                            </p>
                            <p style="margin:3px 0 0;font-size:0.86rem;color:var(--cream);">
                                <b>✅ {'उपाय' if lang in ['hi','mr'] else 'Action'}:</b> {_drow['preventive_advice']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

                # ── FULL SEASON FORECAST CHART ─────────────────────
                st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                    color:var(--straw);margin:4px 0 12px;text-transform:uppercase;">
                    📅 {'पूर्ण मौसम रोग पूर्वानुमान' if lang=='hi' else 'संपूर्ण हंगाम रोग अंदाज' if lang=='mr' else f'Full-Season Forecast — Next {_forecast_days} Days'}
                </div>""", unsafe_allow_html=True)

                _forecast_checkpoints = list(range(_prev_crop_age + 1, _prev_crop_age + _forecast_days + 1, 10)) or [_prev_crop_age + 10]
                _chart_rows, _timeline_cards = [], []

                for _chk_day in _forecast_checkpoints:
                    _chk_date = _prev_sowing + timedelta(days=_chk_day)
                    _chk_diseases = _dis_timeline[
                        (_dis_timeline["day_start"] <= _chk_day) &
                        (_dis_timeline["day_end"] >= _chk_day)
                    ].copy()
                    if _chk_diseases.empty:
                        _chart_rows.append({"Day": _chk_day, "Date": _chk_date.strftime("%d %b"), "Top Disease": "Safe Zone", "Risk %": 0, "Count": 0})
                    else:
                        _chk_diseases["risk_score"] = _chk_diseases.apply(
                            lambda r: _calc_risk_score_at(r, _chk_day, _prev_avg_temp, _prev_avg_humid, _prev_avg_rain), axis=1
                        )
                        _chk_diseases = _chk_diseases.sort_values("risk_score", ascending=False)
                        _top = _chk_diseases.iloc[0]
                        _chart_rows.append({"Day": _chk_day, "Date": _chk_date.strftime("%d %b"),
                                            "Top Disease": str(_top["disease"]).replace("_"," ").title(),
                                            "Risk %": round(_top["risk_score"]*100, 1), "Count": len(_chk_diseases)})
                        _timeline_cards.append((_chk_day, _chk_date, _chk_diseases))

                _chart_df = pd.DataFrame(_chart_rows)
                _bar_colors = ["#27ae60" if r==0 else "#2ecc71" if r<40 else "#f39c12" if r<60 else "#e67e22" if r<75 else "#e74c3c"
                               for r in _chart_df["Risk %"]]

                _fig_tl = go.Figure()
                _fig_tl.add_trace(go.Bar(
                    x=[f"Day {r['Day']}<br>({r['Date']})" for _, r in _chart_df.iterrows()],
                    y=_chart_df["Risk %"], marker_color=_bar_colors,
                    text=[f"{r['Top Disease']}<br>{r['Risk %']}%" for _, r in _chart_df.iterrows()],
                    textposition="outside", textfont=dict(size=9, color="#D4C5A9"),
                    hovertemplate="<b>Day %{customdata[0]}</b><br>Top: %{customdata[1]}<br>Risk: %{y:.0f}%<br>Active: %{customdata[2]}<extra></extra>",
                    customdata=[[r["Day"], r["Top Disease"], r["Count"]] for _, r in _chart_df.iterrows()]
                ))
                _fig_tl.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#D4C5A9", size=10, family="Georgia"),
                    yaxis=dict(title="Peak Risk %", range=[0,110], gridcolor="rgba(255,255,255,0.06)"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9)),
                    margin=dict(t=30, b=20, l=40, r=20), height=280, showlegend=False,
                    title=dict(text=f"{'रोग जोखिम पूर्वानुमान' if lang=='hi' else 'रोग जोखीम अंदाज' if lang=='mr' else 'Disease Risk Forecast — Next '+str(_forecast_days)+' Days'}",
                               font=dict(family="Cinzel", size=12, color="#E8C875"), x=0.02)
                )
                st.plotly_chart(_fig_tl, use_container_width=True, key="disease_timeline_chart")

                # ── EXPANDABLE DAY CARDS ───────────────────────────
                if _timeline_cards:
                    st.markdown(f"""<div style="font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.2em;
                        color:var(--straw);margin:8px 0 10px;text-transform:uppercase;">
                        🗓️ {'दिन-वार पूर्वानुमान' if lang in ['hi','mr'] else 'Day-by-Day Detailed Forecast'}
                    </div>""", unsafe_allow_html=True)

                    for _chk_day, _chk_date, _chk_dis in _timeline_cards:
                        _days_from_now = _chk_day - _prev_crop_age
                        _top_row = _chk_dis.iloc[0]
                        _top_score = _top_row["risk_score"]
                        if _top_score >= 0.7: _fc_color, _fc_emoji = "#e74c3c", "🔴"
                        elif _top_score >= 0.4: _fc_color, _fc_emoji = "#f39c12", "🟡"
                        else: _fc_color, _fc_emoji = "#27ae60", "🟢"
                        _top_name = str(_top_row["disease"]).replace("_"," ").title()

                        with st.expander(f"{_fc_emoji} Day {_chk_day} ({_chk_date.strftime('%d %b')}) · +{_days_from_now} days · {len(_chk_dis)} disease(s) · Top: {_top_name} ({_top_score*100:.0f}%)"):
                            for _, _fr in _chk_dis.iterrows():
                                _rs = _fr["risk_score"]
                                if _rs >= 0.7: _rc2, _re2 = "#e74c3c", "🔴"
                                elif _rs >= 0.4: _rc2, _re2 = "#f39c12", "🟡"
                                else: _rc2, _re2 = "#27ae60", "🟢"
                                _dn2 = str(_fr["disease"]).replace("_"," ").title()
                                st.markdown(f"""
                                <div style="background:{_rc2}12;border-left:4px solid {_rc2};padding:10px 14px;border-radius:6px;margin:6px 0;">
                                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
                                        <b style="color:{_rc2};font-family:'Cinzel',serif;font-size:0.85rem;">{_re2} {_dn2}</b>
                                        <span style="background:{_rc2};color:white;padding:2px 8px;border-radius:8px;font-size:11px;">{_rs*100:.0f}% risk</span>
                                    </div>
                                    <div style="font-size:0.75rem;color:var(--straw);margin:5px 0 3px;">
                                        📆 Day {int(_fr['day_start'])}–{int(_fr['day_end'])} &nbsp;|&nbsp;
                                        💧 {_fr['humidity_factor']} &nbsp;|&nbsp; 🌡️ {_fr['temp_factor']} &nbsp;|&nbsp; 🌧️ {_fr['rain_factor']}
                                    </div>
                                    <div style="font-size:0.84rem;color:var(--cream);">
                                        ✅ <b>{'उपाय' if lang in ['hi','mr'] else 'Action'}:</b> {_fr['preventive_advice']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            if _scan_model_ready:
                                st.markdown(f"""<span style="font-size:0.7rem;color:var(--sage);">
                                    🤖 CNN model ready — upload a crop photo above on day {_chk_day} to confirm with AI
                                </span>""", unsafe_allow_html=True)
                else:
                    st.success(f"✅ {'अगले ' + str(_forecast_days) + ' दिनों में कोई रोग खतरा नहीं!' if lang=='hi' else f'पुढील {_forecast_days} दिवसांत कोणतेही रोग नाही!' if lang=='mr' else f'No disease risks forecast in the next {_forecast_days} days!'}")

                _ml_path = os.path.join(BASE_DIR, "onion_disease_risk_model.pkl")
                if os.path.exists(_ml_path):
                    try:
                        with open(_ml_path, "rb") as _f: _risk_ml = pickle.load(_f)
                        _feats = np.array([[_prev_crop_age, _prev_avg_temp, _prev_avg_humid, _prev_avg_rain]])
                        _ml_proba = _risk_ml.predict_proba(_feats)[0]
                        _ml_cls = _risk_ml.classes_[np.argmax(_ml_proba)]
                        st.markdown(f"""<div style="margin-top:10px;padding:8px 14px;background:rgba(74,124,89,0.1);
                            border:1px solid var(--leaf);border-radius:4px;font-size:0.8rem;">
                            🤖 <b>Risk ML Model:</b> <code>{_ml_cls}</code> ({max(_ml_proba)*100:.1f}% confidence)
                        </div>""", unsafe_allow_html=True)
                    except Exception:
                        pass

    # ═══════════════════════════════════════════════════════════════
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
    tab1, tab3, tab4 = st.tabs([
        f"⟲ {T['crisis_rewind']}",
        f"◉ {T['market_analysis']}",
        f"◎ {T['news_feed']}",
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

        st.markdown(f"""<div class="panel-title">◎ {T['news_feed']}
            <span style="font-size:0.6rem;color:var(--sage);margin-left:12px;">
                ↻ {'दर 2 मिनिटांनी अपडेट' if lang=='mr' else 'हर 2 मिनट में अपडेट' if lang=='hi' else 'Live · updates every 2 min'}
            </span>
        </div>""", unsafe_allow_html=True)

        news_items = get_agri_news(lang)

        if not news_items:
            st.markdown(f"""
            <div style="text-align:center;padding:32px;border:1px dashed var(--bark);border-radius:8px;color:var(--straw);">
                <div style="font-size:2rem;margin-bottom:8px;">📰</div>
                <div style="font-family:'Cinzel',serif;font-size:0.82rem;color:var(--harvest);">
                    {'सध्या बातम्या उपलब्ध नाहीत. इंटरनेट कनेक्शन तपासा.' if lang=='mr'
                     else 'अभी समाचार उपलब्ध नहीं। इंटरनेट कनेक्शन जांचें।' if lang=='hi'
                     else 'No news available right now. Check your internet connection.'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            nc1, nc2 = st.columns(2)
            for i, item in enumerate(news_items):
                col = nc1 if i % 2 == 0 else nc2
                with col:
                    _pub = item['published'][:16] if item['published'] else ""
                    st.markdown(f"""
                    <div class="news-card">
                        <div class="news-title">{item['title']}</div>
                        <div class="news-body" style="margin-top:6px;font-size:0.78rem;color:var(--straw);
                                                      display:-webkit-box;-webkit-line-clamp:3;
                                                      -webkit-box-orient:vertical;overflow:hidden;">
                            {item['body']}
                        </div>
                        <div style="margin-top:8px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;">
                            <span class="news-tag">{item['tag']}</span>
                            <span style="font-size:0.65rem;color:var(--bark);">{_pub}</span>
                        </div>
                        <div style="margin-top:8px;">
                            <a href="{item['link']}" target="_blank"
                               style="font-size:0.72rem;color:var(--sage);text-decoration:none;font-family:'Cinzel',serif;letter-spacing:0.05em;">
                                {'अधिक वाचा →' if lang=='mr' else 'और पढ़ें →' if lang=='hi' else 'Read More →'}
                            </a>
                        </div>
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




if __name__ == "__main__":
    main()

# An-AI-Based-Framework-for-Early-Detection-of-Crop-Loss-and-Overproduction-Risk
This project uses AI and computer vision to detect onion crop diseases from leaf images and also analyzes production patterns to identify overproduction risk. It provides early warnings and preventive recommendations to reduce crop losses and support better yield and market planning

⚙️ Features Provided

The system provides the following key features:

🌱 1. Crop Disease Detection (Computer Vision)
Upload or input a leaf image
Classifies the crop as healthy or diseased
Supports early disease diagnosis using AI

📈 2. Overproduction Risk Analysis

Uses historical production and market data
Predicts risk of overproduction as:
HIGH
MEDIUM
LOW
Helps prevent market crashes due to excess supply

🔍 3. Analog Year Matching

Compares the current season with historical seasons
Identifies similar years based on rainfall, production, and arrivals
Improves reliability of risk prediction

📊 4. Trend & Pattern Analysis

Detects:
Long-term production trends
Seasonal fluctuations
Price instability
Helps in understanding crop behavior over time

⚠️ 5. Early Warning System

Provides alerts for:
Disease presence
Overproduction risk
Enables farmers and planners to take preventive actions

🧠 6. Decision Support for Farmers & Planners

Supports:
Crop planning decisions
Storage and market timing
Yield optimization
Helps reduce crop loss and financial risk

This project integrates custom-generated agricultural datasets along with a public crop disease image dataset from Kaggle to perform early detection of crop loss and overproduction risk.

🔹 1. Generated Agricultural Datasets (6 Datasets)
We created six structured datasets to model historical production trends, market behavior, and environmental conditions related to onion crops:

Onion Price Dataset – Historical price variations over multiple years.
Onion Arrivals Dataset – Daily/seasonal market arrival quantities.
Onion Production Dataset – Year-wise crop production values.
Monthly Onion Data (1960–2025) – Long-term seasonal and trend analysis.
Enhanced Price Dataset – Includes derived features such as moving averages and volatility.
Enhanced Arrivals Dataset – Includes normalized arrivals and trend indicators.

These datasets are used to:
Identify patterns of overproduction
Detect abnormal production trends
Estimate market risk levels (HIGH / MEDIUM / LOW)
Support analog-year comparison for prediction

2. Crop Disease Image Dataset (Kaggle)

A publicly available crop disease dataset from Kaggle is used for training the computer vision model for onion leaf disease detection.

The dataset contains labeled images belonging to the following classes:

Alternaria_D
Botrytis Leaf Blight
Bulb Rot
Bulb_blight-D
Caterpillar-P
Downy mildew
Fusarium-D
Healthy leaves
Iris yellow virus_augment
onion1

This dataset enables:

Automated classification of onion leaf diseases
Differentiation between healthy and infected plants
Early detection of fungal, viral, and pest-related infections
Reduction of crop loss through timely disease identification and treatment
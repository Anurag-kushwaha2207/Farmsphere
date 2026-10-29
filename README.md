# 🌾 FarmSphere AI — Level 3 Complete Smart Agriculture Platform

> **An AI-powered Smart Agriculture System for Crop Recommendation, Soil Fertility Analysis, Crop Suitability Assessment, Yield Prediction, Precision Irrigation, and IoT Sensor Integration.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML%20Models-orange.svg)](https://scikit-learn.org/)
[![Firebase](https://img.shields.io/badge/Cloud-Firebase%20Only-FFCA28.svg)](https://firebase.google.com/)
[![IoT](https://img.shields.io/badge/IoT-ESP32%20Firmware-green.svg)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 🌟 1. System Overview

**FarmSphere AI** combines agricultural datasets, machine learning models, farmer-provided information, and future IoT sensor integration to provide intelligent agricultural decision support.

The platform uses **Firebase as its primary cloud infrastructure** for authentication, database management, file storage, backend services, and notifications (no AWS or Azure).

### Core Data Collection Modes:
1. **Manual Data Collection Mode**: Farmers enter soil test and crop data via web forms.
2. **IoT-Based Automatic Data Collection Mode**: When registered IoT devices (ESP32 with soil moisture, DHT22, pH, NPK probes) are verified and active, **manual measurement entry is strictly disabled for farmers** by backend policy, while historical records remain available for comparison.

---

## 📁 2. Project Architecture & Modular Structure

All modules are strictly separated into dedicated directories:

```
FarmSphere/
├── datasets/                            # Agricultural Datasets
│   ├── raw/                             # Independent raw datasets (Soil, Weather, Crops, Yield, Farmers)
│   ├── processed/                       # Cleaned and integrated datasets ready for ML
│   └── generate_datasets.py             # ICAR/CIMMYT/FAO benchmark dataset generator
│
├── ml_models/                           # Machine Learning & Agricultural AI
│   ├── preprocessing/
│   │   ├── data_cleaner.py              # Outlier detection, range validation & imputation
│   │   └── dataset_integrator.py        # Multi-dataset relational integration pipeline
│   ├── training/
│   │   ├── train_crop_recommendation.py # Crop Recommendation Model (Random Forest Classifier)
│   │   ├── train_crop_suitability.py    # Wheat & Rice Field Suitability Engine
│   │   ├── train_yield_prediction.py    # Crop Yield Regression Model (tonnes/ha & quintals/acre)
│   │   ├── train_irrigation_advisor.py  # FAO-56 Penman-Monteith Evapotranspiration Engine
│   │   └── train_fertilizer_engine.py   # ICAR STCR Precision Fertilizer Engine
│   ├── saved_models/                    # Trained .joblib binaries and evaluation JSONs
│   ├── evaluation/
│   │   ├── model_evaluator.py           # Comprehensive evaluation suite
│   │   └── evaluation_summary.json      # Cross-validation and test scores
│   ├── explainability/
│   │   └── explainable_ai.py            # Feature attribution & agronomic interpretation
│   ├── agronomy/
│   │   └── engines.py                   # Core calculation classes
│   └── predict.py                       # Unified prediction CLI & API interface
│
├── backend/                             # Firebase Backend & Hardware Ingestion
│   ├── firebase/
│   │   └── firestore.rules              # Strict security rules enforcing IoT mode lock
│   ├── server.js                        # Express REST API & telemetry ingestion
│   └── package.json
│
├── iot_firmware/                        # Smart Agriculture Hardware
│   ├── esp32_farmsphere_node.ino        # ESP32 C++ Arduino firmware (Capacitive Moisture, DHT22, pH, NPK)
│   └── iot_telemetry_simulator.py       # Virtual hardware telemetry generator
│
└── frontend/                            # React + Vite Web Application (isolated for UI phase)
```

---

## 🧪 3. Machine Learning Models & Evaluation Metrics

| Model Module | Algorithm | Evaluation Metric | Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Crop Recommendation** | Random Forest Classifier (100 Trees) | **Accuracy**<br>**F1-Score** | **98.62%**<br>**98.63%** | ✅ PASSED |
| **Wheat Suitability** | Dual Random Forest (Regressor + Classifier) | **R² Score**<br>**Class Accuracy** | **0.9654**<br>**95.55%** | ✅ PASSED |
| **Rice Suitability** | Dual Random Forest (Regressor + Classifier) | **R² Score**<br>**Class Accuracy** | **0.9512**<br>**97.55%** | ✅ PASSED |
| **Yield Prediction** | One-Hot Pipeline + RF Regressor | **R² Score**<br>**MAE** | **0.9954**<br>**0.26 tonnes/ha** | ✅ PASSED |
| **Irrigation Advisor** | FAO-56 Penman-Monteith ($ET_c = ET_o \times K_c$) | Deficit (mm)<br>Pump Runtime | Calculated dynamically | ✅ PASSED |
| **Fertilizer Engine** | ICAR Soil Test Crop Response (STCR) | Bags (50kg)<br>3-Stage Splits | Exact kg Urea, DAP, MOP | ✅ PASSED |

---

## 🚀 4. How to Run & Test

### Run Complete Field Intelligence Prediction
```bash
python ml_models/predict.py
```

### Run Multi-Model Testing & Evaluation Suite
```bash
python ml_models/evaluation/model_evaluator.py
```

### Run Data Preprocessing & Multi-Dataset Integration
```bash
python datasets/generate_datasets.py
python ml_models/preprocessing/data_cleaner.py
python ml_models/preprocessing/dataset_integrator.py
```

### Run IoT Telemetry Simulation
```bash
python iot_firmware/iot_telemetry_simulator.py
```

---

## 🛡️ 5. Mode Switching Security (Manual vs IoT Mode)

- When an authorized administrator or device manager activates **IoT Mode**:
  - The backend and Firestore rules **immediately reject manual farmer measurements (HTTP 403 Forbidden)**.
  - Live sensor telemetry becomes the active data source.
  - Historical manual data is preserved for auditing and comparative analysis.

---

## 👨‍💻 Author
**Anurag Singh (Kushwaha)**  
GitHub: [@Anurag-kushwaha2207](https://github.com/Anurag-kushwaha2207)

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
│   ├── raw/                             # Raw datasets (Soil, Weather, Crops, Yield, Farmers, Kaggle)
│   ├── processed/                       # Cleaned and integrated datasets ready for ML
│   ├── generate_datasets.py             # Calibrated agricultural dataset generator [SIMULATED]
│   └── DATA_CARD.md                     # Data Card: sources, DES 2022-23 citations & limitations
│
├── ml_models/                           # Machine Learning & Agricultural AI
│   ├── preprocessing/
│   │   ├── data_cleaner.py              # Outlier detection, range validation & mg/kg->kg/ha conversion
│   │   └── dataset_integrator.py        # Multi-dataset relational integration pipeline (Seeded)
│   ├── training/
│   │   ├── train_crop_recommendation.py # Crop Recommendation Model (Real Kaggle 22-crop dataset)
│   │   └── train_yield_prediction.py    # Group-Aware Crop Yield Regressor (Field split by field_id)
│   ├── agronomy/
│   │   ├── suitability.py               # Deterministic Agronomic Rule Engine (Limiting factors)
│   │   ├── suitability_config.json      # Biological thresholds across all 7 master crops
│   │   ├── yield_predictor.py           # Yield predictor with RF tree spread uncertainty
│   │   └── engines.py                   # Hargreaves-Samani ETo & Nutrient-Balance engines
│   ├── explainability/
│   │   └── explainable_ai.py            # Real marginal feature contributions & biological benchmarks
│   ├── evaluation/
│   │   ├── split_utils.py               # Shared group split helpers & naive crop-mean baselines
│   │   ├── model_evaluator.py           # Multi-model evaluation suite on held-out test data
│   │   └── evaluation_summary.json      # Certified held-out test evaluation metrics
│   ├── saved_models/                    # Trained models (joblib gitignored) & metric JSONs
│   ├── model_card_yield.md              # Model Card: Yield predictor architecture, bounds & limits
│   ├── predict.py                       # Unified prediction CLI (JSON file/stdin/interactive)
│   └── run_all.py                       # One-command full pipeline rebuild orchestrator
│
├── tests/                               # Comprehensive Pytest Verification Suite
│   ├── test_reproducibility.py          # Determinism test (identical CSV checksums on re-run)
│   ├── test_validation.py               # Input validation, OOD detection & rejection of absurd inputs
│   ├── test_evaluator.py                # Split disjointness & verification that train rows are never in test
│   ├── test_suitability.py              # Rule engine verification & limiting factor checks
│   ├── test_yield.py                    # Yield uncertainty range & unsupported crop error checks
│   └── test_agronomy.py                 # Hand-calculated expected values for irrigation and fertilizer
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

> **Data Provenance & Evaluation Integrity Disclosure:**  
> - **Simulation:** Field harvest, soil testing, and weather CSVs are **simulated**, calibrated against the Directorate of Economics and Statistics (DES 2022–23) Indian national crop averages and ICAR guidelines (see [`datasets/DATA_CARD.md`](datasets/DATA_CARD.md)).
> - **Real Data:** The crop recommendation classifier is trained on the real Kaggle 22-crop agricultural dataset (`Crop_recommendation.csv`).
> - **Evaluation Protocol:** Every reported metric is computed strictly on **unseen, held-out test splits** (stratified split for recommendation, `GroupShuffleSplit` by `field_id` for yield to prevent field-level leakage). Training set scores are never presented as test metrics.

| Model Module | Algorithm & Method | Held-Out Evaluation Metric | Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Crop Recommendation** | Random Forest Classifier (22 Crops, Stratified 80/20) | **Test Accuracy**<br>**Macro-F1** | **99.55%**<br>**99.54%** | ✅ PASSED |
| **Crop Suitability** | Deterministic Agronomic Rule Engine (`suitability.py`) | **Limiting Factors**<br>**Coverage** | **Biological Score (0-100)**<br>**All 7 Master Crops** | ✅ VERIFIED |
| **Yield Prediction** | Random Forest Regressor (Group Split by `field_id`) | **Held-Out Test R²**<br>**Held-Out MAE**<br>**Within-Crop R²**<br>*(Naive Baseline R²)* | **0.9948**<br>**0.158 t/ha**<br>**0.9791**<br>*(0.6876)* | ✅ PASSED |
| **Irrigation Advisor** | Hargreaves-Samani ETo & Volumetric Soil Depletion | Water Deficit (mm)<br>Pump Runtime | Dynamically calculated | ✅ PASSED |
| **Fertilizer Engine** | Nutrient-Balance Estimate (1 mg/kg = 2.0 kg/ha furrow slice) | Commercial Bags (50kg)<br>Growth-Stage Splits | Exact Urea, DAP, MOP | ✅ PASSED |

---

## 🚀 4. How to Run & Test

### One-Command Full Pipeline Rebuild & Reproduction
Re-runs dataset generation, cleaning, multi-source integration, model training, evaluation, and README synchronization:
```bash
python -m ml_models.run_all
```

### Run Pytest Verification Suite
Runs all 23 unit tests across determinism, agronomic physics, validation, and evaluation integrity:
```bash
pytest
```

### Run Field Intelligence Prediction CLI
Accepts direct JSON payload via argument or standard input:
```bash
# 1. Run demonstration sample:
python -m ml_models.predict

# 2. Run with a custom JSON input file:
python -m ml_models.predict path/to/field_input.json

# 3. Stream JSON via stdin (for backend integration):
cat field_input.json | python -m ml_models.predict -
```

### Run Held-Out Model Evaluation Suite
```bash
python -m ml_models.evaluation.model_evaluator
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

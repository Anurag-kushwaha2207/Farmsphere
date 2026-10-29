# FarmSphere AI — Agricultural Dataset Documentation & Data Card

## 1. Overview & Dataset Provenance

This document details the origins, simulation methods, assumptions, and limitations of all datasets used within FarmSphere AI.

### Summary Table

| Dataset File | Nature | Source / Benchmark | Purpose |
| :--- | :--- | :--- | :--- |
| `raw/Crop_recommendation.csv` | **Real Public Dataset** | Kaggle Crop Recommendation Dataset (~2,200 rows, 22 crops) | Multi-class crop recommendation |
| `raw/soil_testing_data.csv` | **Simulated** | Statistically generated using typical Indian soil testing distributions (ICAR ranges) | Soil fertility screening |
| `raw/weather_climate_data.csv` | **Simulated** | Regional meteorological averages across 12 agro-climatic districts (2018–2025) | Seasonal climate modeling |
| `raw/crop_master_data.csv` | **Reference Standards** | Agronomic biological requirement ranges & DES 2022–23 national yield baselines | Agronomic rules & crop limits |
| `raw/farmer_field_records.csv` | **Simulated** | 300 fields across 150 farms with realistic rotation systems | Farm management registry |
| `raw/historical_yield_data.csv`| **Simulated** | Mechanistic yield response simulation (crop base x water stress x heat stress x soil factor) | Yield regression modeling |

---

## 2. Yield Citations & Baseline Reference Values

All base potential yields used in the simulation are calibrated against published statistics from the **Directorate of Economics & Statistics (DES), Ministry of Agriculture & Farmers Welfare, Government of India**:
- *Reference*: **Agricultural Statistics at a Glance (2022–2023)**, Directorate of Economics and Statistics, Department of Agriculture & Farmers Welfare, Ministry of Agriculture & Farmers Welfare, Government of India.
- *Online Access*: [agricoop.nic.in / eands.dacnet.nic.in](https://eands.dacnet.nic.in/)

### All-India Average Yields vs Simulation Base (DES 2022–23)

| Crop | DES All-India Avg Yield (2022–23) | Simulation Base Potential Yield | Irrigated High-Yield Potential Benchmark |
| :--- | :--- | :--- | :--- |
| **Wheat** | **3.53 t/ha** (35.3 q/ha) | **3.60 t/ha** | ~4.8 – 5.2 t/ha (Punjab/Haryana) |
| **Rice (Paddy)** | **2.81 t/ha** (28.1 q/ha) | **2.90 t/ha** | ~3.8 – 4.2 t/ha (Irrigated Andhra/Punjab) |
| **Maize** | **3.19 t/ha** (31.9 q/ha) | **3.20 t/ha** | ~4.5 – 5.5 t/ha (Rabi irrigated) |
| **Chickpea (Gram)** | **1.19 t/ha** (11.9 q/ha) | **1.20 t/ha** | ~1.6 – 2.0 t/ha |
| **Mustard** | **1.46 t/ha** (14.6 q/ha) | **1.45 t/ha** | ~1.8 – 2.2 t/ha |
| **Cotton (Seed Cotton)** | **1.45 t/ha** (~480 kg lint/ha) | **1.50 t/ha** | ~2.0 – 2.5 t/ha |
| **Potato** | **23.8 t/ha** (238 q/ha) | **23.50 t/ha** | ~28.0 – 32.0 t/ha |

*Note*: Sugarcane has been excluded from both master and yield datasets because it is a perennial crop with vastly distinct water/growth dynamics and is not present in the 22-crop recommendation dataset.

---

## 3. Simulation Methodology & Assumptions

### A. Rabi Rainfall vs Seasonal Water Requirement
- **Observed Agronomic Reality**: In Northern India, natural rainfall during the Rabi winter season (Nov–April) is only 50–100 mm (derived from Western Disturbances).
- **Resolution**: We model **total water availability** ($\text{water\_available} = \text{season\_rainfall} + \text{irrigation\_water}$), rather than comparing rainfall alone against crop water requirements. A field with tube well or drip irrigation receives 280–380 mm of supplemental water, satisfying the 350–500 mm requirement for Wheat.
- **Water Stress Equation**:
  $$\text{Factor}_{\text{water}} = \begin{cases} 1.0 + \min(0.08, \frac{W - W_{\text{opt}}}{W_{\text{opt}}} \times 0.1), & W \ge W_{\text{opt}} \\ 0.85 + \frac{W - W_{\text{min}}}{W_{\text{opt}} - W_{\text{min}}} \times 0.15, & W_{\text{min}} \le W < W_{\text{opt}} \\ \max(0.40, 0.85 - \frac{W_{\text{min}} - W}{W_{\text{min}}} \times 0.60), & W < W_{\text{min}} \end{cases}$$

### B. Temperature Stress Modeling
- **Terminal Heat Stress for Wheat**: In North India, temperatures exceeding 21°C during grain filling (March/April) lead to forced maturity and shriveled grains. A simulated penalty of 4% per °C above 21°C is applied.
- **Cold Stress for Rice**: Rice yields drop if average vegetative season temperatures fall below 25°C.

### C. Soil Fertility & Unit Conversions
- Soil lab tests report Nitrogen (N), Phosphorus (P), and Potassium (K) in **mg/kg (ppm)**.
- Standard agronomic conversion for a 15 cm furrow slice with bulk density ~1.33 g/cm³ ($2.24 \times 10^6$ kg soil/ha):
  $$\text{Nutrient (kg/ha)} \approx \text{Nutrient (mg/kg)} \times 2.24$$
- In `data_cleaner.py`, NPK categories are explicitly labeled with their conversion factor rather than claiming direct unadjusted ICAR kg/ha thresholds.

---

## 4. Known Limitations

1. **Synthetic Nature**: Soil test rows, weather entries, and harvest yield records are simulated using parametric models. While calibrated against published ICAR and DES state averages, they do not represent physically measured micro-plots.
2. **Geographic Scope**: Calibrated primarily to Indian agro-climatic zones (Indo-Gangetic Plain, Central Black Soils, Deccan Plateau).
3. **Pest & Pathogen Interactions**: Pest outbreaks, weeds, and severe hail damage are not dynamically simulated; natural stochastic variance is modeled via a 5% Gaussian noise parameter.

"""
FarmSphere AI - Agricultural Benchmark Datasets Generator
Generates realistic agricultural CSV datasets matching ICAR, CIMMYT, and FAO standards:
1. soil_testing_data.csv       - Independent soil lab testing reports
2. weather_climate_data.csv    - Meteorological & climate historical records
3. crop_master_data.csv        - Biological thresholds & nutrient requirements per crop
4. historical_yield_data.csv   - Actual field harvest yield records with management practices
5. farmer_field_records.csv    - Farmer & landholding registry
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

RAW_DIR = os.path.join(os.path.dirname(__file__), 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

print(f"Creating datasets in {RAW_DIR}...")

# -----------------------------------------------------------------------------
# 1. Soil Testing Dataset (Independent lab records)
# -----------------------------------------------------------------------------
DISTRICTS = [
    'Bareilly, UP', 'Varanasi, UP', 'Meerut, UP', 'Karnal, Haryana', 
    'Ludhiana, Punjab', 'Patna, Bihar', 'Samastipur, Bihar', 'Bhopal, MP', 
    'Indore, MP', 'Nagpur, Maharashtra', 'Ahmedabad, Gujarat', 'Guntur, AP'
]

SOIL_TYPES = ['Alluvial', 'Clay Loam', 'Sandy Loam', 'Black Soil', 'Red Loam']

n_soil_samples = 2200

soil_data = []
for i in range(1, n_soil_samples + 1):
    district = np.random.choice(DISTRICTS)
    stype = np.random.choice(SOIL_TYPES)
    
    # Realistic distributions based on Indian agricultural soil testing
    if 'Black' in stype:
        ph = np.clip(np.random.normal(7.6, 0.4), 6.5, 8.8)
        n = np.clip(np.random.normal(75, 25), 20, 160)
        p = np.clip(np.random.normal(35, 15), 10, 85)
        k = np.clip(np.random.normal(85, 25), 30, 150)
        oc = np.clip(np.random.normal(0.65, 0.18), 0.2, 1.2)
        ec = np.clip(np.random.normal(0.6, 0.25), 0.1, 1.8)
        moisture = np.clip(np.random.normal(48, 12), 15, 80)
    elif 'Sandy' in stype:
        ph = np.clip(np.random.normal(6.8, 0.5), 5.5, 8.2)
        n = np.clip(np.random.normal(55, 20), 15, 120)
        p = np.clip(np.random.normal(25, 12), 8, 65)
        k = np.clip(np.random.normal(40, 15), 15, 90)
        oc = np.clip(np.random.normal(0.40, 0.12), 0.15, 0.75)
        ec = np.clip(np.random.normal(0.4, 0.15), 0.1, 1.2)
        moisture = np.clip(np.random.normal(35, 10), 10, 60)
    elif 'Clay' in stype:
        ph = np.clip(np.random.normal(6.5, 0.6), 5.2, 7.8)
        n = np.clip(np.random.normal(95, 25), 35, 175)
        p = np.clip(np.random.normal(45, 18), 15, 95)
        k = np.clip(np.random.normal(70, 22), 25, 130)
        oc = np.clip(np.random.normal(0.72, 0.20), 0.3, 1.4)
        ec = np.clip(np.random.normal(0.7, 0.3), 0.2, 2.2)
        moisture = np.clip(np.random.normal(62, 14), 25, 92)
    else: # Alluvial / Red Loam
        ph = np.clip(np.random.normal(7.1, 0.5), 6.0, 8.3)
        n = np.clip(np.random.normal(85, 28), 25, 160)
        p = np.clip(np.random.normal(40, 16), 12, 90)
        k = np.clip(np.random.normal(60, 20), 20, 120)
        oc = np.clip(np.random.normal(0.58, 0.15), 0.25, 1.1)
        ec = np.clip(np.random.normal(0.5, 0.2), 0.15, 1.6)
        moisture = np.clip(np.random.normal(52, 12), 20, 85)
        
    soil_data.append({
        'sample_id': f'SOIL_{i:05d}',
        'district': district,
        'soil_type': stype,
        'soil_ph': round(float(ph), 2),
        'nitrogen_mg_kg': round(float(n), 1),
        'phosphorus_mg_kg': round(float(p), 1),
        'potassium_mg_kg': round(float(k), 1),
        'organic_carbon_pct': round(float(oc), 2),
        'electrical_conductivity_ds_m': round(float(ec), 2),
        'soil_moisture_pct': round(float(moisture), 1)
    })

df_soil = pd.DataFrame(soil_data)
df_soil.to_csv(os.path.join(RAW_DIR, 'soil_testing_data.csv'), index=False)
print(f" Saved soil_testing_data.csv ({len(df_soil)} rows)")

# -----------------------------------------------------------------------------
# 2. Weather & Climate Dataset (District & Season historical observations)
# -----------------------------------------------------------------------------
SEASONS = ['Kharif (Monsoon)', 'Rabi (Winter)', 'Zaid (Summer)']
weather_data = []

for district in DISTRICTS:
    for year in range(2018, 2026):
        for season in SEASONS:
            if 'Kharif' in season:
                temp_avg = np.random.normal(29.0, 1.8)
                humidity = np.random.normal(78.0, 7.0)
                rainfall = np.random.normal(850.0, 180.0)
                solar_rad = np.random.normal(18.5, 2.5)
            elif 'Rabi' in season:
                temp_avg = np.random.normal(18.5, 2.2)
                humidity = np.random.normal(60.0, 8.0)
                rainfall = np.random.normal(75.0, 35.0)
                solar_rad = np.random.normal(16.0, 2.0)
            else: # Zaid
                temp_avg = np.random.normal(34.0, 2.5)
                humidity = np.random.normal(45.0, 9.0)
                rainfall = np.random.normal(40.0, 25.0)
                solar_rad = np.random.normal(22.0, 2.8)
                
            weather_data.append({
                'weather_record_id': f'WTR_{district[:3].upper()}_{year}_{season[:3].upper()}',
                'district': district,
                'year': year,
                'season': season,
                'avg_temp_c': round(float(temp_avg), 1),
                'min_temp_c': round(float(temp_avg - np.random.uniform(5, 9)), 1),
                'max_temp_c': round(float(temp_avg + np.random.uniform(5, 9)), 1),
                'avg_humidity_pct': round(float(np.clip(humidity, 30, 98)), 1),
                'season_rainfall_mm': round(float(max(rainfall, 5.0)), 1),
                'solar_radiation_mj_m2': round(float(solar_rad), 1)
            })

df_weather = pd.DataFrame(weather_data)
df_weather.to_csv(os.path.join(RAW_DIR, 'weather_climate_data.csv'), index=False)
print(f" Saved weather_climate_data.csv ({len(df_weather)} rows)")

# -----------------------------------------------------------------------------
# 3. Crop Master Dataset (Agronomic biological requirements)
# -----------------------------------------------------------------------------
CROPS_MASTER = [
    {
        'crop_id': 'wheat',
        'crop_name': 'Wheat',
        'category': 'Cereal',
        'primary_season': 'Rabi (Winter)',
        'min_temp_c': 10, 'opt_temp_c': 20, 'max_temp_c': 30,
        'min_ph': 6.0, 'opt_ph': 6.8, 'max_ph': 7.8,
        'req_n_kg_ha': 120, 'req_p_kg_ha': 60, 'req_k_kg_ha': 40,
        'min_rainfall_mm': 300, 'opt_rainfall_mm': 500, 'max_rainfall_mm': 800,
        'preferred_soil': 'Alluvial, Loamy',
        'base_potential_yield_t_ha': 5.2
    },
    {
        'crop_id': 'rice',
        'crop_name': 'Rice (Paddy)',
        'category': 'Cereal',
        'primary_season': 'Kharif (Monsoon)',
        'min_temp_c': 20, 'opt_temp_c': 28, 'max_temp_c': 38,
        'min_ph': 5.5, 'opt_ph': 6.2, 'max_ph': 7.2,
        'req_n_kg_ha': 130, 'req_p_kg_ha': 50, 'req_k_kg_ha': 60,
        'min_rainfall_mm': 900, 'opt_rainfall_mm': 1400, 'max_rainfall_mm': 2500,
        'preferred_soil': 'Clayey, Clay Loam',
        'base_potential_yield_t_ha': 5.5
    },
    {
        'crop_id': 'maize',
        'crop_name': 'Maize',
        'category': 'Cereal',
        'primary_season': 'Kharif (Monsoon)',
        'min_temp_c': 18, 'opt_temp_c': 26, 'max_temp_c': 35,
        'min_ph': 5.8, 'opt_ph': 6.5, 'max_ph': 7.5,
        'req_n_kg_ha': 120, 'req_p_kg_ha': 60, 'req_k_kg_ha': 50,
        'min_rainfall_mm': 450, 'opt_rainfall_mm': 650, 'max_rainfall_mm': 1000,
        'preferred_soil': 'Sandy Loam, Loamy',
        'base_potential_yield_t_ha': 6.0
    },
    {
        'crop_id': 'cotton',
        'crop_name': 'Cotton',
        'category': 'Fibre / Cash',
        'primary_season': 'Kharif (Monsoon)',
        'min_temp_c': 21, 'opt_temp_c': 28, 'max_temp_c': 37,
        'min_ph': 6.0, 'opt_ph': 7.2, 'max_ph': 8.2,
        'req_n_kg_ha': 100, 'req_p_kg_ha': 50, 'req_k_kg_ha': 50,
        'min_rainfall_mm': 550, 'opt_rainfall_mm': 750, 'max_rainfall_mm': 1200,
        'preferred_soil': 'Black Soil, Clay Loam',
        'base_potential_yield_t_ha': 2.8
    },
    {
        'crop_id': 'chickpea',
        'crop_name': 'Chickpea',
        'category': 'Pulse / Legume',
        'primary_season': 'Rabi (Winter)',
        'min_temp_c': 12, 'opt_temp_c': 20, 'max_temp_c': 28,
        'min_ph': 6.0, 'opt_ph': 7.0, 'max_ph': 8.0,
        'req_n_kg_ha': 25, 'req_p_kg_ha': 60, 'req_k_kg_ha': 30,
        'min_rainfall_mm': 250, 'opt_rainfall_mm': 400, 'max_rainfall_mm': 650,
        'preferred_soil': 'Sandy Loam, Alluvial',
        'base_potential_yield_t_ha': 2.2
    },
    {
        'crop_id': 'potato',
        'crop_name': 'Potato',
        'category': 'Tuber',
        'primary_season': 'Rabi (Winter)',
        'min_temp_c': 14, 'opt_temp_c': 19, 'max_temp_c': 25,
        'min_ph': 5.0, 'opt_ph': 5.8, 'max_ph': 6.8,
        'req_n_kg_ha': 150, 'req_p_kg_ha': 80, 'req_k_kg_ha': 120,
        'min_rainfall_mm': 400, 'opt_rainfall_mm': 550, 'max_rainfall_mm': 800,
        'preferred_soil': 'Sandy Loam, Loose Alluvial',
        'base_potential_yield_t_ha': 28.0
    },
    {
        'crop_id': 'mustard',
        'crop_name': 'Mustard',
        'category': 'Oilseed',
        'primary_season': 'Rabi (Winter)',
        'min_temp_c': 10, 'opt_temp_c': 18, 'max_temp_c': 27,
        'min_ph': 6.0, 'opt_ph': 7.0, 'max_ph': 8.2,
        'req_n_kg_ha': 80, 'req_p_kg_ha': 40, 'req_k_kg_ha': 40,
        'min_rainfall_mm': 250, 'opt_rainfall_mm': 380, 'max_rainfall_mm': 550,
        'preferred_soil': 'Sandy Loam, Loam',
        'base_potential_yield_t_ha': 1.9
    },
    {
        'crop_id': 'sugarcane',
        'crop_name': 'Sugarcane',
        'category': 'Commercial',
        'primary_season': 'Annual',
        'min_temp_c': 20, 'opt_temp_c': 30, 'max_temp_c': 40,
        'min_ph': 6.0, 'opt_ph': 6.8, 'max_ph': 7.8,
        'req_n_kg_ha': 200, 'req_p_kg_ha': 80, 'req_k_kg_ha': 100,
        'min_rainfall_mm': 1200, 'opt_rainfall_mm': 1800, 'max_rainfall_mm': 2600,
        'preferred_soil': 'Deep Alluvial, Clay Loam',
        'base_potential_yield_t_ha': 85.0
    }
]

df_crop_master = pd.DataFrame(CROPS_MASTER)
df_crop_master.to_csv(os.path.join(RAW_DIR, 'crop_master_data.csv'), index=False)
print(f" Saved crop_master_data.csv ({len(df_crop_master)} crops)")

# -----------------------------------------------------------------------------
# 4. Farmer & Farm Registry Dataset
# -----------------------------------------------------------------------------
farmers_data = []
farm_field_records = []
field_idx = 1

for f_id in range(1, 151):
    district = np.random.choice(DISTRICTS)
    farm_id = f'FARM_{f_id:04d}'
    farmer_name = f'Farmer_{f_id}'
    num_fields = np.random.randint(1, 4)
    
    for fld in range(1, num_fields + 1):
        fld_id = f'FLD_{field_idx:05d}'
        field_idx += 1
        area_ha = round(float(np.random.uniform(0.8, 4.5)), 2)
        stype = np.random.choice(SOIL_TYPES)
        irrig = np.random.choice(['Tube Well (Canal / Groundwater)', 'Drip Irrigation', 'Sprinkler', 'Rainfed (No source)'], p=[0.55, 0.20, 0.15, 0.10])
        
        farm_field_records.append({
            'farmer_id': f'FMR_{f_id:04d}',
            'farm_id': farm_id,
            'field_id': fld_id,
            'district': district,
            'area_hectares': area_ha,
            'soil_type': stype,
            'irrigation_source': irrig,
            'active_data_mode': 'manual' # Initially manual mode for all fields
        })

df_farmers = pd.DataFrame(farm_field_records)
df_farmers.to_csv(os.path.join(RAW_DIR, 'farmer_field_records.csv'), index=False)
print(f" Saved farmer_field_records.csv ({len(df_farmers)} fields across 150 farms)")

# -----------------------------------------------------------------------------
# 5. Historical Yield Observations Dataset (Real harvest records CIMMYT/ICAR style)
# -----------------------------------------------------------------------------
yield_records = []
rec_id = 1

for idx, field in df_farmers.iterrows():
    fld_id = field['field_id']
    district = field['district']
    stype = field['soil_type']
    irrig = field['irrigation_source']
    area = field['area_hectares']
    
    # 3-5 years of history for each field
    for yr in range(2021, 2026):
        for crop_info in CROPS_MASTER:
            # Check season & climate suitability
            crop_name = crop_info['crop_name']
            if yr % 2 == 0 and crop_info['crop_id'] in ['wheat', 'chickpea', 'mustard', 'potato']:
                season = 'Rabi (Winter)'
            elif yr % 2 == 1 and crop_info['crop_id'] in ['rice', 'maize', 'cotton']:
                season = 'Kharif (Monsoon)'
            else:
                continue
                
            # Base potential yield
            base_yield = crop_info['base_potential_yield_t_ha']
            
            # Irrigation impact factor
            if 'Drip' in irrig:
                irrig_factor = 1.15
            elif 'Tube Well' in irrig:
                irrig_factor = 1.02
            elif 'Sprinkler' in irrig:
                irrig_factor = 1.08
            else: # Rainfed
                irrig_factor = 0.72
                
            # Soil type affinity
            soil_factor = 1.05 if stype in crop_info['preferred_soil'] else 0.88
            
            # Management & weather noise
            noise = np.random.normal(0.98, 0.08)
            
            # Actual harvest yield
            actual_yield = round(float(base_yield * irrig_factor * soil_factor * noise), 2)
            total_production = round(float(actual_yield * area), 2)
            
            yield_records.append({
                'yield_record_id': f'YLD_{rec_id:06d}',
                'field_id': fld_id,
                'district': district,
                'crop_id': crop_info['crop_id'],
                'crop_name': crop_name,
                'year': yr,
                'season': season,
                'soil_type': stype,
                'irrigation_source': irrig,
                'field_area_ha': area,
                'actual_yield_tonnes_per_ha': actual_yield,
                'total_production_tonnes': total_production
            })
            rec_id += 1

df_yield = pd.DataFrame(yield_records)
df_yield.to_csv(os.path.join(RAW_DIR, 'historical_yield_data.csv'), index=False)
print(f" Saved historical_yield_data.csv ({len(df_yield)} harvest yield records)")

print("\nAll raw agricultural datasets successfully generated in datasets/raw/")

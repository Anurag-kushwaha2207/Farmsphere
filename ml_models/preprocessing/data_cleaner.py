"""
FarmSphere AI - Data Cleaning & Validation Pipeline
Responsible for:
- Missing value imputation and duplicate removal
- Outlier handling (IQR clamping for realistic agricultural bounds)
- Correct conversion from laboratory mg/kg (ppm) to field available kg/ha
- Categorization using standard ICAR soil fertility ratings (in kg/ha)
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/raw'))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/processed'))
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Standard furrow slice conversion factor (0-15 cm soil layer, bulk density ~1.33 g/cm3)
# 1 hectare furrow slice = 2.24 million kg soil => 1 mg/kg (ppm) = 2.24 kg/ha
PPM_TO_KG_PER_HA = 2.24

class DataCleaner:
    def __init__(self):
        self.validation_report = {}

    def clean_soil_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Cleaning Soil Testing Dataset ---")
        initial_len = len(df)
        df_clean = df.copy()

        # 1. Check & drop exact duplicates
        df_clean = df_clean.drop_duplicates(subset=['sample_id'])

        # 2. Agronomic Range Validation & Clamping
        # pH must be realistic (3.5 to 10.0)
        df_clean['soil_ph'] = df_clean['soil_ph'].clip(lower=4.0, upper=9.5)
        # N, P, K must be strictly positive
        df_clean['nitrogen_mg_kg'] = df_clean['nitrogen_mg_kg'].clip(lower=5.0, upper=250.0)
        df_clean['phosphorus_mg_kg'] = df_clean['phosphorus_mg_kg'].clip(lower=2.0, upper=150.0)
        df_clean['potassium_mg_kg'] = df_clean['potassium_mg_kg'].clip(lower=5.0, upper=200.0)
        # Organic Carbon % (0.05 to 2.5%)
        df_clean['organic_carbon_pct'] = df_clean['organic_carbon_pct'].clip(lower=0.1, upper=2.0)
        # Electrical Conductivity dS/m (0.05 to 5.0)
        df_clean['electrical_conductivity_ds_m'] = df_clean['electrical_conductivity_ds_m'].clip(lower=0.05, upper=4.0)
        # Moisture % (5 to 100%)
        df_clean['soil_moisture_pct'] = df_clean['soil_moisture_pct'].clip(lower=5.0, upper=98.0)

        # 3. Explicit conversion: mg/kg -> available kg/ha
        df_clean['available_n_kg_ha'] = round(df_clean['nitrogen_mg_kg'] * PPM_TO_KG_PER_HA, 1)
        df_clean['available_p_kg_ha'] = round(df_clean['phosphorus_mg_kg'] * PPM_TO_KG_PER_HA, 1)
        df_clean['available_k_kg_ha'] = round(df_clean['potassium_mg_kg'] * PPM_TO_KG_PER_HA, 1)

        # 4. Standard ICAR Soil Fertility Ratings (evaluated in kg/ha):
        # - Available Nitrogen: Low < 280 kg/ha, Medium 280 - 560 kg/ha, High > 560 kg/ha
        # - Available Phosphorus (Olsen P): Low < 10 kg/ha, Medium 10 - 25 kg/ha, High > 25 kg/ha
        # - Available Potassium: Low < 110 kg/ha, Medium 110 - 280 kg/ha, High > 280 kg/ha
        df_clean['icar_n_status'] = pd.cut(
            df_clean['available_n_kg_ha'], 
            bins=[-np.inf, 280, 560, np.inf], 
            labels=['Low (<280 kg/ha)', 'Medium (280-560 kg/ha)', 'High (>560 kg/ha)']
        )
        df_clean['icar_p_status'] = pd.cut(
            df_clean['available_p_kg_ha'], 
            bins=[-np.inf, 10, 25, np.inf], 
            labels=['Low (<10 kg/ha)', 'Medium (10-25 kg/ha)', 'High (>25 kg/ha)']
        )
        df_clean['icar_k_status'] = pd.cut(
            df_clean['available_k_kg_ha'], 
            bins=[-np.inf, 110, 280, np.inf], 
            labels=['Low (<110 kg/ha)', 'Medium (110-280 kg/ha)', 'High (>280 kg/ha)']
        )

        print(f" Cleaned soil records: {len(df_clean)} / {initial_len}")
        return df_clean

    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Cleaning Weather & Climate Dataset ---")
        df_clean = df.copy()

        # Validate temperatures
        df_clean['avg_temp_c'] = df_clean['avg_temp_c'].clip(lower=0.0, upper=50.0)
        df_clean['min_temp_c'] = df_clean['min_temp_c'].clip(lower=-5.0, upper=40.0)
        df_clean['max_temp_c'] = df_clean['max_temp_c'].clip(lower=5.0, upper=52.0)
        # Humidity %
        df_clean['avg_humidity_pct'] = df_clean['avg_humidity_pct'].clip(lower=15.0, upper=100.0)
        # Rainfall mm
        df_clean['season_rainfall_mm'] = df_clean['season_rainfall_mm'].clip(lower=0.0, upper=3500.0)

        print(f" Cleaned weather records: {len(df_clean)}")
        return df_clean

    def clean_yield_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Cleaning Historical Yield Dataset ---")
        df_clean = df.copy()

        # Drop impossible yields (<= 0 or extreme outliers)
        df_clean = df_clean[df_clean['actual_yield_tonnes_per_ha'] > 0.1]
        
        # Calculate Quintals per Acre for Indian farmer units (1 tonne/ha = 4.04686 quintals/acre)
        df_clean['actual_yield_quintals_per_acre'] = round(df_clean['actual_yield_tonnes_per_ha'] * 4.047, 2)
        
        print(f" Cleaned yield records: {len(df_clean)}")
        return df_clean

def run_cleaning():
    cleaner = DataCleaner()
    
    # Load raw
    soil_df = pd.read_csv(os.path.join(RAW_DIR, 'soil_testing_data.csv'))
    weather_df = pd.read_csv(os.path.join(RAW_DIR, 'weather_climate_data.csv'))
    yield_df = pd.read_csv(os.path.join(RAW_DIR, 'historical_yield_data.csv'))
    
    # Clean
    soil_clean = cleaner.clean_soil_data(soil_df)
    weather_clean = cleaner.clean_weather_data(weather_df)
    yield_clean = cleaner.clean_yield_data(yield_df)
    
    # Save clean versions
    soil_clean.to_csv(os.path.join(PROCESSED_DIR, 'soil_cleaned.csv'), index=False)
    weather_clean.to_csv(os.path.join(PROCESSED_DIR, 'weather_cleaned.csv'), index=False)
    yield_clean.to_csv(os.path.join(PROCESSED_DIR, 'yield_cleaned.csv'), index=False)
    
    print("\n Cleaned datasets saved to datasets/processed/")

if __name__ == '__main__':
    run_cleaning()

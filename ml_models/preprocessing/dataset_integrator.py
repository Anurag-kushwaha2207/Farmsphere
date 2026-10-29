"""
FarmSphere AI - Multiple Dataset Integration Engine
Integrates independent agricultural data sources:
1. Soil testing records
2. Weather climate records
3. Crop master reference standards
4. Farmer field registry
5. Historical harvest yield records
6. Real Kaggle Crop Recommendation dataset (N, P, K, temp, humidity, ph, rainfall)
"""

import os
import pandas as pd
import numpy as np

# Seed for 100% reproducibility across runs
np.random.seed(42)

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/raw'))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/processed'))
os.makedirs(PROCESSED_DIR, exist_ok=True)

class DatasetIntegrator:
    def __init__(self):
        print("Initializing Agricultural Dataset Integrator (seed=42)...")
        self.soil_df = pd.read_csv(os.path.join(RAW_DIR, 'soil_testing_data.csv'))
        self.weather_df = pd.read_csv(os.path.join(RAW_DIR, 'weather_climate_data.csv'))
        self.crop_master_df = pd.read_csv(os.path.join(RAW_DIR, 'crop_master_data.csv'))
        self.yield_df = pd.read_csv(os.path.join(RAW_DIR, 'historical_yield_data.csv'))
        self.farmers_df = pd.read_csv(os.path.join(RAW_DIR, 'farmer_field_records.csv'))

    def build_crop_recommendation_dataset(self) -> pd.DataFrame:
        """
        Integrates the real Kaggle Crop Recommendation dataset.
        Columns: N, P, K, temperature, humidity, ph, rainfall, label (22 crops, ~2200 rows)
        Strictly requires datasets/raw/Crop_recommendation.csv. No silent fallbacks.
        """
        print("\n[1/3] Preparing Real Kaggle Crop Recommendation Dataset...")
        kaggle_raw_path = os.path.join(RAW_DIR, 'Crop_recommendation.csv')
        
        if not os.path.exists(kaggle_raw_path):
            error_msg = (
                "\n" + "=" * 70 + "\n"
                "CRITICAL ERROR: Real Kaggle 'Crop_recommendation.csv' not found!\n"
                f"Expected location: {kaggle_raw_path}\n\n"
                "DOWNLOAD INSTRUCTIONS:\n"
                "1. Download the dataset from Kaggle:\n"
                "   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n"
                "2. Place the uncompressed CSV file at:\n"
                f"   {kaggle_raw_path}\n"
                "3. Re-run this pipeline.\n"
                "Silent fallback to synthetic data is prohibited by project ground rules.\n"
                + "=" * 70
            )
            print(error_msg)
            raise FileNotFoundError(f"Missing mandatory dataset: {kaggle_raw_path}")
            
        df_rec = pd.read_csv(kaggle_raw_path)
        required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        for col in required_cols:
            if col not in df_rec.columns:
                raise ValueError(f"Crop_recommendation.csv missing required column: {col}")
                
        # Drop duplicates and clean whitespace
        df_rec = df_rec.drop_duplicates().copy()
        df_rec['label'] = df_rec['label'].str.strip()
        
        # Consistent reproducible shuffle
        df_rec = df_rec.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        out_path = os.path.join(PROCESSED_DIR, 'crop_recommendation_clean.csv')
        df_rec.to_csv(out_path, index=False)
        print(f" Saved {out_path} ({len(df_rec)} rows, {df_rec['label'].nunique()} crops)")
        return df_rec

    def build_crop_suitability_dataset(self) -> pd.DataFrame:
        """
        Builds benchmark suitability evaluation dataset across master crops.
        """
        print("\n[2/3] Building Crop Suitability Reference Dataset...")
        records = []
        
        # Test wheat and rice against all soil testing samples
        for _, soil in self.soil_df.iterrows():
            ph = soil['soil_ph']
            n = soil['nitrogen_mg_kg']
            p = soil['phosphorus_mg_kg']
            k = soil['potassium_mg_kg']
            oc = soil['organic_carbon_pct']
            ec = soil['electrical_conductivity_ds_m']
            stype = soil['soil_type']
            
            # --- Evaluate Wheat Suitability ---
            score_wheat = 100.0
            if ph < 6.0 or ph > 7.8: score_wheat -= 25.0
            elif ph < 6.5 or ph > 7.3: score_wheat -= 10.0
            if n < 60: score_wheat -= 18.0
            elif n < 80: score_wheat -= 8.0
            if p < 25: score_wheat -= 15.0
            if k < 25: score_wheat -= 12.0
            if ec > 1.5: score_wheat -= 20.0
            if stype in ['Alluvial', 'Clay Loam']: score_wheat += 5.0
            elif 'Sandy' in stype: score_wheat -= 15.0
            
            score_wheat = float(np.clip(score_wheat, 15.0, 98.0))
            if score_wheat >= 85: sclass_w = 'Highly Suitable'
            elif score_wheat >= 70: sclass_w = 'Moderately Suitable'
            elif score_wheat >= 50: sclass_w = 'Marginally Suitable'
            else: sclass_w = 'Unsuitable'
            
            records.append({
                'sample_id': soil['sample_id'],
                'district': soil['district'],
                'soil_type': stype,
                'target_crop': 'Wheat',
                'soil_ph': ph,
                'nitrogen': n,
                'phosphorus': p,
                'potassium': k,
                'organic_carbon': oc,
                'electrical_conductivity': ec,
                'suitability_score': round(score_wheat, 1),
                'suitability_class': sclass_w
            })

            # --- Evaluate Rice Suitability ---
            score_rice = 100.0
            if ph < 5.0 or ph > 7.5: score_rice -= 30.0
            elif ph > 7.0: score_rice -= 15.0
            if n < 80: score_rice -= 15.0
            if p < 25: score_rice -= 10.0
            if k < 35: score_rice -= 10.0
            if ec > 1.8: score_rice -= 25.0
            if 'Clay' in stype: score_rice += 10.0
            elif 'Sandy' in stype: score_rice -= 35.0
            
            score_rice = float(np.clip(score_rice, 10.0, 98.0))
            if score_rice >= 85: sclass_r = 'Highly Suitable'
            elif score_rice >= 70: sclass_r = 'Moderately Suitable'
            elif score_rice >= 50: sclass_r = 'Marginally Suitable'
            else: sclass_r = 'Unsuitable'
            
            records.append({
                'sample_id': soil['sample_id'],
                'district': soil['district'],
                'soil_type': stype,
                'target_crop': 'Rice (Paddy)',
                'soil_ph': ph,
                'nitrogen': n,
                'phosphorus': p,
                'potassium': k,
                'organic_carbon': oc,
                'electrical_conductivity': ec,
                'suitability_score': round(score_rice, 1),
                'suitability_class': sclass_r
            })

        df_suit = pd.DataFrame(records)
        out_path = os.path.join(PROCESSED_DIR, 'crop_suitability_clean.csv')
        df_suit.to_csv(out_path, index=False)
        print(f" Saved {out_path} ({len(df_suit)} evaluations)")
        return df_suit

    def build_crop_yield_dataset(self) -> pd.DataFrame:
        """
        Integrates Field Records + Weather Records + Yield Records into a 
        multi-feature regression dataset for Yield Prediction.
        Features: crop_name, soil_type, irrigation_source, field_area_ha, 
                  season, avg_temp_c, season_rainfall_mm, avg_humidity_pct
        Key for group-splitting: field_id
        """
        print("\n[3/3] Building Integrated Crop Yield Regression Dataset...")
        
        # Merge yield with weather via district, year, season
        merged = pd.merge(
            self.yield_df,
            self.weather_df,
            on=['district', 'year', 'season'],
            how='inner'
        )
        
        merged['actual_yield_quintals_per_acre'] = round(merged['actual_yield_tonnes_per_ha'] * 4.047, 2)
        
        # Preserve field_id for group splitting
        yield_final = merged[[
            'field_id', 'crop_name', 'soil_type', 'irrigation_source', 'field_area_ha',
            'season', 'avg_temp_c', 'season_rainfall_mm', 'avg_humidity_pct',
            'actual_yield_tonnes_per_ha', 'actual_yield_quintals_per_acre'
        ]].copy()
        
        # Ensure deterministic sort and index
        yield_final = yield_final.sort_values(by=['field_id', 'crop_name', 'season']).reset_index(drop=True)
        
        out_path = os.path.join(PROCESSED_DIR, 'yield_prediction_clean.csv')
        yield_final.to_csv(out_path, index=False)
        print(f" Saved {out_path} ({len(yield_final)} integrated harvest records)")
        return yield_final

def run_integration():
    integrator = DatasetIntegrator()
    integrator.build_crop_recommendation_dataset()
    integrator.build_crop_suitability_dataset()
    integrator.build_crop_yield_dataset()
    print("\n Multi-Dataset Integration Complete!")

if __name__ == '__main__':
    run_integration()

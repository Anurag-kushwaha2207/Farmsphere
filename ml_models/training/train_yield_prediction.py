"""
FarmSphere AI - Crop Yield Prediction Model Training Pipeline
Algorithm: Random Forest Regressor vs Gradient Boosting Regressor
Inputs: crop_name, soil_type, irrigation_source, field_area_ha, season, 
        avg_temp_c, season_rainfall_mm, avg_humidity_pct
Output: Estimated Crop Yield in Tonnes/Hectare & Quintals/Acre
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/processed/yield_prediction_clean.csv'))
SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def train_yield_predictor():
    print("=" * 60)
    print("   FARMSPHERE AI: TRAINING CROP YIELD PREDICTION MODEL")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} harvest yield records across {df['crop_name'].nunique()} crops.")
    
    categorical_cols = ['crop_name', 'soil_type', 'irrigation_source', 'season']
    numeric_cols = ['field_area_ha', 'avg_temp_c', 'season_rainfall_mm', 'avg_humidity_pct']
    feature_cols = categorical_cols + numeric_cols
    
    X = df[feature_cols]
    y = df['actual_yield_tonnes_per_ha']
    
    # Train / Test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )
    
    # Candidate Models
    models = {
        'Ridge Regression': Pipeline([
            ('prep', preprocessor),
            ('reg', Ridge(alpha=1.0))
        ]),
        'Random Forest Regressor': Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42, n_jobs=1))
        ]),
        'Gradient Boosting Regressor': Pipeline([
            ('prep', preprocessor),
            ('reg', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
        ])
    }
    
    print("\n--- 5-Fold Cross-Validation (R2 Score) ---")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    best_name = None
    best_score = -float('inf')
    best_pipe = None
    cv_scores_dict = {}
    
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2')
        mean_r2 = scores.mean()
        std_r2 = scores.std()
        cv_scores_dict[name] = {'mean_r2': round(float(mean_r2), 4), 'std': round(float(std_r2), 4)}
        print(f"  {name:30s}: Mean R2 = {mean_r2:.4f} (+/- {std_r2:.4f})")
        
        if mean_r2 > best_score:
            best_score = mean_r2
            best_name = name
            best_pipe = pipe
            
    print(f"\n>> Selected Best Model: {best_name} (CV R2: {best_score:.4f})")
    
    # Fit best model
    best_pipe.fit(X_train, y_train)
    
    # Test evaluation
    y_pred = best_pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\n--- Test Set Evaluation ---")
    print(f"Test R2 Score:                     {r2:.4f}")
    print(f"Mean Absolute Error (MAE):         {mae:.2f} tonnes/ha")
    print(f"Root Mean Squared Error (RMSE):    {rmse:.2f} tonnes/ha")
    print(f"Mean Absolute Percentage Error:    {mape:.2f}%")
    
    # Save Pipeline Artifact
    save_path = os.path.join(SAVED_MODELS_DIR, 'yield_prediction_model.joblib')
    joblib.dump(best_pipe, save_path)
    print(f"\n Model binary saved to: {save_path}")
    
    # Save Metrics JSON
    metrics = {
        'model_name': 'Crop Yield Prediction Regressor',
        'algorithm': best_name,
        'features': {
            'categorical': categorical_cols,
            'numeric': numeric_cols
        },
        'cross_validation': cv_scores_dict,
        'test_metrics': {
            'r2_score': round(float(r2), 4),
            'mae_tonnes_per_ha': round(float(mae), 3),
            'rmse_tonnes_per_ha': round(float(rmse), 3),
            'mape_pct': round(float(mape), 2)
        }
    }
    
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'yield_prediction_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f" Evaluation metrics saved to: {metrics_path}")
    
    # Sample Test Predictions
    print("\n--- Sample Test Predictions ---")
    test_samples = pd.DataFrame([
        {
            'crop_name': 'Wheat',
            'soil_type': 'Alluvial',
            'irrigation_source': 'Tube Well (Canal / Groundwater)',
            'field_area_ha': 2.5,
            'season': 'Rabi (Winter)',
            'avg_temp_c': 18.5,
            'season_rainfall_mm': 75.0,
            'avg_humidity_pct': 60.0
        },
        {
            'crop_name': 'Rice (Paddy)',
            'soil_type': 'Clay Loam',
            'irrigation_source': 'Drip Irrigation',
            'field_area_ha': 3.0,
            'season': 'Kharif (Monsoon)',
            'avg_temp_c': 28.5,
            'season_rainfall_mm': 920.0,
            'avg_humidity_pct': 82.0
        }
    ])
    
    sample_preds = best_pipe.predict(test_samples)
    for i, (_, row) in enumerate(test_samples.iterrows()):
        t_ha = sample_preds[i]
        q_acre = t_ha * 4.047
        tot_production = t_ha * row['field_area_ha']
        print(f"\nSample {i+1}: {row['crop_name']} on {row['field_area_ha']} ha ({row['soil_type']} soil, {row['irrigation_source']})")
        print(f"  -> Predicted Yield: {t_ha:.2f} tonnes/ha ({q_acre:.1f} quintals/acre)")
        print(f"  -> Total Estimated Harvest: {tot_production:.1f} tonnes across field")

if __name__ == '__main__':
    train_yield_predictor()

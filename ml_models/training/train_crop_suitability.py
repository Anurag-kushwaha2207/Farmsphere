"""
FarmSphere AI - Crop Suitability Model Training Pipeline
Evaluates field-specific soil & environmental compatibility for targeted crops (Wheat & Rice).
Inputs: soil_ph, nitrogen, phosphorus, potassium, organic_carbon, electrical_conductivity
Outputs: Suitability Score (0-100) + Suitability Class (Highly / Moderately / Marginally Suitable / Unsuitable)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/processed/crop_suitability_clean.csv'))
SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def train_crop_suitability():
    print("=" * 60)
    print("   FARMSPHERE AI: TRAINING CROP SUITABILITY MODEL")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} suitability evaluations.")
    
    # We train separate models or unified model conditioned on target crop
    feature_cols = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_carbon', 'electrical_conductivity']
    
    suitability_artifacts = {}
    
    for crop in ['Wheat', 'Rice (Paddy)']:
        print(f"\n>>> Training Suitability Engine for: {crop}")
        crop_df = df[df['target_crop'] == crop].copy()
        
        X = crop_df[feature_cols]
        y_score = crop_df['suitability_score']
        y_class = crop_df['suitability_class']
        
        # Train / Test split
        X_train, X_test, y_train_score, y_test_score, y_train_class, y_test_class = train_test_split(
            X, y_score, y_class, test_size=0.20, random_state=42
        )
        
        # 1. Regressor for Score
        reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
        reg.fit(X_train, y_train_score)
        score_preds = reg.predict(X_test)
        
        r2 = r2_score(y_test_score, score_preds)
        mae = mean_absolute_error(y_test_score, score_preds)
        rmse = np.sqrt(mean_squared_error(y_test_score, score_preds))
        
        print(f"  [Score Regressor] R2 Score: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        
        # 2. Classifier for Class
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        clf.fit(X_train, y_train_class)
        class_preds = clf.predict(X_test)
        
        acc = accuracy_score(y_test_class, class_preds)
        print(f"  [Class Classifier] Accuracy: {acc*100:.2f}%")
        
        # Feature importance
        feat_imp = dict(zip(feature_cols, [round(float(v), 4) for v in reg.feature_importances_]))
        
        suitability_artifacts[crop] = {
            'regressor': reg,
            'classifier': clf,
            'features': feature_cols,
            'classes': list(clf.classes_),
            'metrics': {
                'r2_score': round(float(r2), 4),
                'mae': round(float(mae), 2),
                'rmse': round(float(rmse), 2),
                'class_accuracy': round(float(acc), 4)
            },
            'feature_importances': feat_imp
        }
        
    # Save Model Artifact
    save_path = os.path.join(SAVED_MODELS_DIR, 'crop_suitability_model.joblib')
    joblib.dump(suitability_artifacts, save_path)
    print(f"\n Suitability model bundle saved to: {save_path}")
    
    # Save Metrics JSON
    clean_metrics = {
        crop: {
            'metrics': data['metrics'],
            'feature_importances': data['feature_importances'],
            'classes': data['classes']
        }
        for crop, data in suitability_artifacts.items()
    }
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'crop_suitability_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    print(f" Metrics report saved to: {metrics_path}")
    
    # Test sample evaluation for Wheat
    print("\n--- Test Suitability Evaluation for Wheat ---")
    test_soil = pd.DataFrame([{
        'soil_ph': 7.1,
        'nitrogen': 95.0,
        'phosphorus': 42.0,
        'potassium': 65.0,
        'organic_carbon': 0.65,
        'electrical_conductivity': 0.45
    }])
    wheat_models = suitability_artifacts['Wheat']
    pred_score = wheat_models['regressor'].predict(test_soil)[0]
    pred_class = wheat_models['classifier'].predict(test_soil)[0]
    print(f"Sample Field Soil: pH=7.1, N=95, P=42, K=65, OC=0.65%, EC=0.45 dS/m")
    print(f"Predicted Suitability Score: {pred_score:.1f} / 100")
    print(f"Predicted Classification:    {pred_class}")

if __name__ == '__main__':
    train_crop_suitability()

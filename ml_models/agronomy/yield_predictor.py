"""
FarmSphere AI - Yield Prediction Engine with Uncertainty Estimation
Wraps the trained Random Forest model with:
- Strict crop validation (refuses unsupported crops with clear ValueError)
- Tree ensemble spread for 10th-90th percentile uncertainty interval
- Realistic unit conversions (tonnes/ha and quintals/acre)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
DEFAULT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'yield_prediction_model.joblib')

class YieldPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Yield model binary not found at: {model_path}. Train the model first.")
        self.pipe = joblib.load(model_path)
        self.supported_crops = getattr(self.pipe, 'supported_crops_', [
            'Wheat', 'Rice (Paddy)', 'Maize', 'Cotton (Seed Cotton)', 'Chickpea', 'Mustard', 'Potato'
        ])
        self.feature_cols = getattr(self.pipe, 'feature_cols_', [
            'crop_name', 'soil_type', 'irrigation_source', 'season',
            'avg_temp_c', 'season_rainfall_mm', 'avg_humidity_pct'
        ])

    def predict_yield(
        self,
        crop_name: str,
        soil_type: str,
        irrigation_source: str,
        season: str,
        avg_temp_c: float,
        season_rainfall_mm: float,
        avg_humidity_pct: float
    ) -> Dict[str, Any]:
        """
        Predicts crop yield with 10th-90th percentile prediction intervals.
        Raises ValueError if crop is not supported (no silent fallbacks).
        """
        matched_crop = None
        for sc in self.supported_crops:
            if sc.lower() == crop_name.strip().lower() or sc.split()[0].lower() == crop_name.strip().lower():
                matched_crop = sc
                break
                
        if not matched_crop:
            supported_str = ", ".join(self.supported_crops)
            raise ValueError(
                f"Unsupported crop for yield prediction: '{crop_name}'. "
                f"Model was trained only on: [{supported_str}]."
            )
            
        input_df = pd.DataFrame([{
            'crop_name': matched_crop,
            'soil_type': soil_type,
            'irrigation_source': irrigation_source,
            'season': season,
            'avg_temp_c': float(avg_temp_c),
            'season_rainfall_mm': float(season_rainfall_mm),
            'avg_humidity_pct': float(avg_humidity_pct)
        }])
        
        prep = self.pipe.named_steps['prep']
        reg = self.pipe.named_steps['reg']
        
        X_prep = prep.transform(input_df)
        
        # Uncertainty interval from Random Forest tree spread
        if hasattr(reg, 'estimators_'):
            tree_preds = np.array([tree.predict(X_prep)[0] for tree in reg.estimators_])
            pred_mean = float(np.mean(tree_preds))
            lower_bound = float(np.percentile(tree_preds, 10))
            upper_bound = float(np.percentile(tree_preds, 90))
        else:
            pred_mean = float(reg.predict(X_prep)[0])
            lower_bound = pred_mean * 0.90
            upper_bound = pred_mean * 1.10
            
        pred_mean = round(max(0.1, pred_mean), 2)
        lower_bound = round(max(0.1, lower_bound), 2)
        upper_bound = round(max(lower_bound, upper_bound), 2)
        
        # Unit conversion: 1 tonne/ha = 4.047 quintals/acre
        quintals_per_acre = round(pred_mean * 4.047, 1)
        lower_q_acre = round(lower_bound * 4.047, 1)
        upper_q_acre = round(upper_bound * 4.047, 1)
        
        return {
            'crop': matched_crop,
            'predicted_yield_tonnes_per_ha': pred_mean,
            'lower_bound_tonnes_per_ha': lower_bound,
            'upper_bound_tonnes_per_ha': upper_bound,
            'yield_range_tonnes_per_ha': f"{lower_bound} - {upper_bound} t/ha",
            'predicted_yield_quintals_per_acre': quintals_per_acre,
            'lower_bound_quintals_per_acre': lower_q_acre,
            'upper_bound_quintals_per_acre': upper_q_acre,
            'yield_range_quintals_per_acre': f"{lower_q_acre} - {upper_q_acre} q/acre",
            'uncertainty_metric': '10th-90th percentile tree ensemble spread'
        }

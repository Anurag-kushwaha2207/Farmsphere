"""
FarmSphere AI - Explainable AI (XAI) Agronomic Explainer
Provides:
1. Feature Attribution: Marginal contribution of each input parameter to the model's decision.
2. Agronomic Interpretation: Benchmarked against transparent biological thresholds from suitability_config.json.
3. Limiting Factors & Drivers: Honest agronomic constraints without invented certainty scores.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../agronomy/suitability_config.json'))

class AgronomicXAIExplainer:
    def __init__(self):
        crop_model_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_model.joblib')
        if not os.path.exists(crop_model_path):
            raise FileNotFoundError(f"Missing crop recommendation model at: {crop_model_path}")
            
        artifact = joblib.load(crop_model_path)
        self.clf = artifact['model']
        self.scaler = artifact['scaler']
        self.crop_scaler = artifact['scaler']
        self.le = artifact['label_encoder']
        self.features = artifact['feature_names']
        self.classes = artifact['classes']
        
        # Load shared suitability config as source of truth for agronomic thresholds
        self.crop_benchmarks = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
                for crop, data in cfg.items():
                    norm_key = crop.lower().split()[0]
                    self.crop_benchmarks[norm_key] = {
                        'N': (data['nitrogen_mg_kg']['min'], data['nitrogen_mg_kg']['opt']),
                        'P': (data['phosphorus_mg_kg']['min'], data['phosphorus_mg_kg']['opt']),
                        'K': (data['potassium_mg_kg']['min'], data['potassium_mg_kg']['opt']),
                        'ph': (data['ph']['opt_min'], data['ph']['opt_max']),
                        'temperature': (data['temperature_c']['opt_min'], data['temperature_c']['opt_max']),
                        'humidity': (40.0, 85.0),
                        'rainfall': (data['water_available_mm']['min'], data['water_available_mm']['opt'])
                    }

    def explain_crop_recommendation(
        self,
        n: float,
        p: float,
        k: float,
        temp: float,
        humidity: float,
        ph: float,
        rainfall: float
    ) -> Dict[str, Any]:
        """
        Explains model prediction with real marginal feature contributions
        and consistent agronomic benchmarks.
        """
        import pandas as pd
        raw_input = pd.DataFrame([{
            'N': n, 'P': p, 'K': k, 'temperature': temp,
            'humidity': humidity, 'ph': ph, 'rainfall': rainfall
        }])[self.features]
        scaled_input = self.crop_scaler.transform(raw_input)
        
        # Predicted Class & Probabilities
        probs = self.clf.predict_proba(scaled_input)[0]
        pred_idx = np.argmax(probs)
        pred_crop = self.classes[pred_idx]
        confidence = float(probs[pred_idx] * 100.0)
        
        # Real Marginal Feature Contributions relative to standard scaler zero-mean baseline
        baseline_scaled = np.zeros_like(scaled_input)
        baseline_pred_prob = float(self.clf.predict_proba(baseline_scaled)[0, pred_idx])
        
        inputs_dict = {
            'N': n, 'P': p, 'K': k, 'ph': ph, 
            'temperature': temp, 'humidity': humidity, 'rainfall': rainfall
        }
        
        # Match benchmark
        norm_pred = pred_crop.lower().split()[0]
        crop_bench = self.crop_benchmarks.get(norm_pred, {
            'N': (20.0, 140.0), 'P': (20.0, 80.0), 'K': (20.0, 100.0),
            'ph': (6.0, 7.5), 'temperature': (18.0, 32.0),
            'humidity': (40.0, 80.0), 'rainfall': (300.0, 1200.0)
        })
        
        feature_breakdown = []
        optimal_factors = []
        limiting_factors = []
        
        for idx, feat in enumerate(self.features):
            val = inputs_dict[feat]
            
            # Marginal contribution: how much does activating this feature change the predicted class probability?
            feat_test = baseline_scaled.copy()
            feat_test[0, idx] = scaled_input[0, idx]
            feat_prob = float(self.clf.predict_proba(feat_test)[0, pred_idx])
            marginal_contrib = round((feat_prob - baseline_pred_prob) * 100.0, 2)
            
            min_opt, max_opt = crop_bench.get(feat, (0.0, 1000.0))
            
            if min_opt <= val <= max_opt:
                status = 'OPTIMAL'
                status_desc = f'Within ideal range [{min_opt} - {max_opt}]'
                optimal_factors.append(f"{feat} ({val:.1f}) is in the optimal range [{min_opt} - {max_opt}] for {pred_crop}.")
            elif val < min_opt:
                status = 'SUB_OPTIMAL_LOW'
                deficit = min_opt - val
                status_desc = f'Deficit of {deficit:.1f} below recommended minimum ({min_opt})'
                limiting_factors.append(f"{feat} ({val:.1f}) is below optimal minimum ({min_opt}) for {pred_crop}.")
            else:
                status = 'SUB_OPTIMAL_HIGH'
                surplus = val - max_opt
                status_desc = f'Surplus of {surplus:.1f} above recommended upper threshold ({max_opt})'
                limiting_factors.append(f"{feat} ({val:.1f}) exceeds optimal threshold ({max_opt}) for {pred_crop}.")
                
            feature_breakdown.append({
                'feature': feat,
                'input_value': val,
                'optimal_range': [min_opt, max_opt],
                'marginal_contribution_pct': marginal_contrib,
                'status': status,
                'status_description': status_desc
            })
            
        interpretation = (
            f"Model predicts {pred_crop} with {confidence:.1f}% statistical probability. "
            f"Primary supportive factors: {optimal_factors[0] if optimal_factors else 'moderate environmental compatibility'}. "
            f"Limiting conditions to monitor: {len(limiting_factors)} identified."
        )
        
        return {
            'recommended_crop': pred_crop,
            'statistical_confidence_pct': round(confidence, 1),
            'interpretation': interpretation,
            'optimal_factors': optimal_factors,
            'limiting_factors': limiting_factors,
            'feature_attributions': feature_breakdown
        }

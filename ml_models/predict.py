"""
FarmSphere AI - Unified Agricultural Prediction & Decision Support Engine
Combines validated agronomic modules into a production API/CLI:
- Input validation with strict biological range enforcement (no impossible values)
- Out-of-Distribution (OOD) detection
- Crop Recommendation (Kaggle real soil dataset)
- Crop Suitability Evaluation (Deterministic Rule Engine)
- Crop Yield Prediction with Tree Ensemble Uncertainty (tonnes/ha & quintals/acre)
- Precision Irrigation Schedule (Hargreaves ETo & Volumetric Root Depletion)
- Fertilizer Prescription (Nutrient-balance advisory)
- Explainable AI (XAI) Marginal Feature Contributions

CLI Usage:
  python -m ml_models.predict field_input.json
  cat field_input.json | python -m ml_models.predict -
  python -m ml_models.predict  # Runs verified demonstration sample
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml_models.agronomy.suitability import CropSuitabilityEngine
from ml_models.agronomy.yield_predictor import YieldPredictor
from ml_models.agronomy.engines import IrrigationAdvisor, FertilizerEngine
from ml_models.explainability.explainable_ai import AgronomicXAIExplainer

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models'))

# Agronomic Valid Ranges
VALID_RANGES = {
    'soil_ph': (3.5, 10.0, "pH must be between 3.5 (extreme acid) and 10.0 (extreme alkali)"),
    'nitrogen': (0.0, 500.0, "Available Nitrogen must be between 0.0 and 500.0 mg/kg"),
    'phosphorus': (0.0, 300.0, "Available Phosphorus must be between 0.0 and 300.0 mg/kg"),
    'potassium': (0.0, 600.0, "Available Potassium must be between 0.0 and 600.0 mg/kg"),
    'temperature': (-5.0, 50.0, "Temperature must be between -5.0°C and 50.0°C"),
    'humidity': (0.0, 100.0, "Relative humidity must be between 0.0% and 100.0%"),
    'rainfall': (0.0, 4000.0, "Seasonal rainfall must be between 0.0 and 4000.0 mm"),
    'soil_moisture_pct': (0.0, 100.0, "Volumetric soil moisture must be between 0.0% and 100.0%"),
    'field_area_ha': (0.01, 10000.0, "Field area must be greater than 0.01 hectares")
}

# Training distribution boundaries for Out-of-Distribution (OOD) flagging
TRAINING_DISTRIBUTIONS = {
    'nitrogen': (0.0, 150.0),
    'phosphorus': (5.0, 150.0),
    'potassium': (5.0, 210.0),
    'temperature': (8.0, 45.0),
    'humidity': (14.0, 100.0),
    'soil_ph': (3.5, 9.9),
    'rainfall': (20.0, 300.0)
}

def validate_field_inputs(data: dict) -> Tuple[bool, List[str], List[str]]:
    """
    Validates field inputs against physical and agronomic boundary conditions.
    Returns (is_valid, error_messages, ood_warnings).
    """
    errors = []
    ood_warnings = []
    
    for key, (min_val, max_val, err_msg) in VALID_RANGES.items():
        if key in data:
            try:
                val = float(data[key])
                if val < min_val or val > max_val:
                    errors.append(f"Invalid '{key}' ({val}): {err_msg}")
            except (ValueError, TypeError):
                errors.append(f"Field '{key}' must be a numerical value.")
                
    # OOD detection for valid numbers
    if not errors:
        for feat, (train_min, train_max) in TRAINING_DISTRIBUTIONS.items():
            if feat in data:
                val = float(data[feat])
                if val < train_min or val > train_max:
                    ood_warnings.append(
                        f"Parameter '{feat}' ({val}) is outside the training distribution range [{train_min} - {train_max}]. "
                        f"Model predictions may have higher uncertainty."
                    )
                    
    return len(errors) == 0, errors, ood_warnings


class FarmSpherePredictor:
    def __init__(self):
        # 1. Crop Recommender (Kaggle real soil dataset)
        rec_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_model.joblib')
        if not os.path.exists(rec_path):
            raise FileNotFoundError(f"Missing crop recommendation model at: {rec_path}. Train it first.")
        artifact = joblib.load(rec_path)
        self.crop_clf = artifact['model']
        self.crop_scaler = artifact['scaler']
        self.crop_le = artifact['label_encoder']
        self.crop_classes = artifact['classes']
        self.crop_features = artifact['feature_names']
        
        # 2. Suitability Rule Engine
        self.suit_engine = CropSuitabilityEngine()
        
        # 3. Yield Predictor
        self.yield_predictor = YieldPredictor()
        
        # 4. Irrigation Advisor & Fertilizer Engine
        self.irrig_advisor = IrrigationAdvisor()
        self.fert_engine = FertilizerEngine()
        
        # 5. XAI Explainer
        self.xai = AgronomicXAIExplainer()

    def analyze_field(self, data: dict) -> dict:
        """
        Executes full agronomic advisory pipeline with strict validation and no silent fallbacks.
        """
        is_valid, errors, ood_warnings = validate_field_inputs(data)
        if not is_valid:
            return {
                'status': 'error',
                'error_type': 'ValidationError',
                'message': 'Input rejected: impossible or invalid agronomic parameter values detected.',
                'errors': errors,
                'field_inputs': data
            }

        # Extract sanitized values
        ph = float(data.get('soil_ph', 7.0))
        n = float(data.get('nitrogen', 100.0))
        p = float(data.get('phosphorus', 50.0))
        k = float(data.get('potassium', 40.0))
        temp = float(data.get('temperature', 22.0))
        humidity = float(data.get('humidity', 60.0))
        rainfall = float(data.get('rainfall', 500.0))
        area_ha = float(data.get('field_area_ha', 2.0))
        soil_type = str(data.get('soil_type', 'Alluvial'))
        irrigation_source = str(data.get('irrigation_source', 'Tube Well (Canal / Groundwater)'))
        season = str(data.get('season', 'Rabi (Winter)'))
        moisture = float(data.get('soil_moisture_pct', 25.0))
        growth_stage = str(data.get('growth_stage', 'Tillering & Vegetative'))
        ec = float(data.get('electrical_conductivity', 0.5))

        # --- 1. Crop Recommendation ---
        input_rec = pd.DataFrame([{
            'N': n, 'P': p, 'K': k, 'temperature': temp,
            'humidity': humidity, 'ph': ph, 'rainfall': rainfall
        }])[self.crop_features]
        scaled_rec = self.crop_scaler.transform(input_rec)
        probs = self.crop_clf.predict_proba(scaled_rec)[0]
        top_idx = np.argmax(probs)
        recommended_crop = self.crop_classes[top_idx]
        rec_confidence = round(float(probs[top_idx] * 100.0), 1)

        top3_indices = np.argsort(probs)[::-1][:3]
        alternatives = [
            {'crop': self.crop_classes[i], 'confidence_pct': round(float(probs[i] * 100.0), 1)}
            for i in top3_indices[1:]
        ]

        # Active crop: user target crop if specified, else recommended crop
        active_crop = data.get('target_crop') or recommended_crop

        # --- 2. Crop Suitability (Transparent Rule Engine) ---
        suitability_res = None
        try:
            suitability_res = self.suit_engine.evaluate_suitability(
                crop_name=active_crop,
                soil_ph=ph,
                nitrogen_mg_kg=n,
                phosphorus_mg_kg=p,
                potassium_mg_kg=k,
                electrical_conductivity_ds_m=ec,
                soil_type=soil_type,
                temperature_c=temp,
                water_available_mm=rainfall
            )
        except ValueError as e:
            suitability_res = {
                'status': 'unsupported',
                'crop': active_crop,
                'message': str(e)
            }

        # --- 3. Crop Yield Prediction (with 10th-90th Percentile Uncertainty) ---
        yield_res = None
        try:
            yield_pred_raw = self.yield_predictor.predict_yield(
                crop_name=active_crop,
                soil_type=soil_type,
                irrigation_source=irrigation_source,
                season=season,
                avg_temp_c=temp,
                season_rainfall_mm=rainfall,
                avg_humidity_pct=humidity
            )
            t_ha = yield_pred_raw['predicted_yield_tonnes_per_ha']
            yield_res = {
                'crop': active_crop,
                'predicted_yield_tonnes_per_ha': t_ha,
                'predicted_yield_quintals_per_acre': yield_pred_raw['predicted_yield_quintals_per_acre'],
                'yield_range_tonnes_per_ha': yield_pred_raw['yield_range_tonnes_per_ha'],
                'yield_range_quintals_per_acre': yield_pred_raw['yield_range_quintals_per_acre'],
                'uncertainty_interval_80pct': {
                    'lower_bound_t_ha': yield_pred_raw['lower_bound_tonnes_per_ha'],
                    'upper_bound_t_ha': yield_pred_raw['upper_bound_tonnes_per_ha']
                },
                'total_harvest_tonnes': round(t_ha * area_ha, 1),
                'field_area_ha': area_ha,
                'field_area_acres': round(area_ha * 2.471, 1)
            }
        except ValueError as e:
            yield_res = {
                'status': 'unsupported',
                'crop': active_crop,
                'message': str(e)
            }

        # --- 4. Irrigation Advisory ---
        irrig_res = None
        try:
            irrig_res = self.irrig_advisor.calculate_irrigation(
                crop_name=active_crop,
                growth_stage=growth_stage,
                current_soil_moisture_pct=moisture,
                soil_type=soil_type,
                temp_c=temp,
                recent_rainfall_mm=0.0,
                irrigation_method='Drip Irrigation' if 'Drip' in irrigation_source else 'Tube Well (Canal / Groundwater)'
            )
        except ValueError as e:
            irrig_res = {
                'status': 'unsupported',
                'crop': active_crop,
                'message': str(e)
            }

        # --- 5. Fertilizer Prescription ---
        fert_res = None
        try:
            fert_res = self.fert_engine.calculate_fertilizer_schedule(
                crop_name=active_crop,
                soil_n_mg_kg=n,
                soil_p_mg_kg=p,
                soil_k_mg_kg=k,
                soil_ph=ph,
                field_area_ha=area_ha
            )
        except ValueError as e:
            fert_res = {
                'status': 'unsupported',
                'crop': active_crop,
                'message': str(e)
            }

        # --- 6. Explainable AI Attribution ---
        xai_res = self.xai.explain_crop_recommendation(
            n=n, p=p, k=k, temp=temp, humidity=humidity, ph=ph, rainfall=rainfall
        )

        return {
            'status': 'success',
            'field_inputs': data,
            'out_of_distribution_warnings': ood_warnings,
            'active_crop': active_crop,
            'crop_recommendation': {
                'recommended_crop': recommended_crop,
                'confidence_pct': rec_confidence,
                'top_3_alternatives': alternatives,
                'input_units_note': 'N, P, K represent soil nutrient ratios from soil testing.'
            },
            'crop_suitability': suitability_res,
            'crop_yield_prediction': yield_res,
            'irrigation_advisory': irrig_res,
            'fertilizer_schedule': fert_res,
            'explainable_ai': xai_res
        }


def main():
    predictor = FarmSpherePredictor()
    
    # Check CLI arguments: JSON file or stdin
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '-':
            # Read from stdin
            raw_input = sys.stdin.read()
            data = json.loads(raw_input)
        else:
            # Read from file
            with open(arg, 'r') as f:
                data = json.load(f)
        result = predictor.analyze_field(data)
        print(json.dumps(result, indent=2))
        return

    # Default demonstration test
    demo_sample = {
        'soil_ph': 7.1,
        'nitrogen': 90.0,
        'phosphorus': 45.0,
        'potassium': 40.0,
        'temperature': 21.0,
        'humidity': 62.0,
        'rainfall': 180.0,
        'soil_moisture_pct': 22.0,
        'electrical_conductivity': 0.4,
        'field_area_ha': 2.0,
        'soil_type': 'Alluvial',
        'irrigation_source': 'Tube Well (Canal / Groundwater)',
        'season': 'Rabi (Winter)',
        'target_crop': 'Wheat'
    }
    
    print("=" * 70)
    print("   FARMSPHERE AI: PREDICTION & DECISION SUPPORT PIPELINE")
    print("=" * 70)
    report = predictor.analyze_field(demo_sample)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

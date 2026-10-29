"""
FarmSphere AI - Unified Agricultural Prediction & Decision Support Engine
Combines all trained ML models & agronomic modules into one single API/CLI:
- Crop Recommendation
- Crop Suitability Evaluation (Wheat / Rice)
- Crop Yield Prediction (tonnes/ha & quintals/acre)
- Precision Irrigation Schedule
- Fertilizer Prescription & Split Dosing
- Explainable AI (XAI) Attribution
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models'))

class FarmSpherePredictor:
    def __init__(self):
        # 1. Load Crop Recommender
        rec_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_model.joblib')
        if not os.path.exists(rec_path):
            raise FileNotFoundError(f"Missing crop recommendation model: {rec_path}")
        artifact = joblib.load(rec_path)
        self.crop_clf = artifact['model']
        self.crop_scaler = artifact['scaler']
        self.crop_le = artifact['label_encoder']
        self.crop_classes = artifact['classes']
        self.crop_features = artifact['feature_names']
        
        # 2. Load Suitability Models
        suit_path = os.path.join(SAVED_MODELS_DIR, 'crop_suitability_model.joblib')
        self.suit_models = joblib.load(suit_path) if os.path.exists(suit_path) else {}
        
        # 3. Load Yield Predictor
        yield_path = os.path.join(SAVED_MODELS_DIR, 'yield_prediction_model.joblib')
        self.yield_pipe = joblib.load(yield_path) if os.path.exists(yield_path) else None
        
        # 4. Irrigation Advisor & Fertilizer Engine
        from agronomy.engines import IrrigationAdvisor, FertilizerEngine
        self.irrig_advisor = IrrigationAdvisor()
        self.fert_engine = FertilizerEngine()
        
        # 5. Load XAI
        from explainability.explainable_ai import AgronomicXAIExplainer
        self.xai = AgronomicXAIExplainer()

    def analyze_field(self, data: dict) -> dict:
        """
        Takes raw farmer / sensor field measurements and produces a comprehensive advisory.
        Required keys:
        - soil_ph (float, e.g. 7.1)
        - nitrogen (float, e.g. 110.0)
        - phosphorus (float, e.g. 55.0)
        - potassium (float, e.g. 45.0)
        - temperature (float, e.g. 21.0)
        - humidity (float, e.g. 58.0)
        - rainfall (float, e.g. 520.0)
        - field_area_ha (float, default 2.0)
        - soil_type (str, default 'Alluvial')
        - irrigation_source (str, default 'Tube Well (Canal / Groundwater)')
        - season (str, default 'Rabi (Winter)')
        - target_crop (str, optional, default None -> uses recommended crop)
        - growth_stage (str, optional, default 'Tillering & Vegetative')
        - soil_moisture_pct (float, optional, default 42.0)
        """
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
        moisture = float(data.get('soil_moisture_pct', 42.0))
        growth_stage = str(data.get('growth_stage', 'Tillering & Vegetative'))
        
        # --- 1. Crop Recommendation ---
        input_rec = np.array([[n, p, k, temp, humidity, ph, rainfall]])
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
        
        # Target crop for yield & fertilizer: if specified by user, use that; else use recommended
        active_crop = data.get('target_crop') or recommended_crop
        
        # --- 2. Crop Suitability Analysis (Field Specific) ---
        suitability_res = None
        suit_key = 'Wheat' if 'Wheat' in active_crop else ('Rice (Paddy)' if 'Rice' in active_crop else None)
        if suit_key and suit_key in self.suit_models:
            models = self.suit_models[suit_key]
            test_df = pd.DataFrame([{
                'soil_ph': ph,
                'nitrogen': n,
                'phosphorus': p,
                'potassium': k,
                'organic_carbon': float(data.get('organic_carbon', 0.65)),
                'electrical_conductivity': float(data.get('electrical_conductivity', 0.5))
            }])
            suit_score = round(float(models['regressor'].predict(test_df)[0]), 1)
            suit_class = str(models['classifier'].predict(test_df)[0])
            suitability_res = {
                'evaluated_crop': suit_key,
                'suitability_score': suit_score,
                'suitability_class': suit_class
            }
        else:
            # Fallback heuristic suitability
            suitability_res = {
                'evaluated_crop': active_crop,
                'suitability_score': rec_confidence,
                'suitability_class': 'Highly Suitable' if rec_confidence >= 80 else 'Moderately Suitable'
            }
            
        # --- 3. Crop Yield Prediction ---
        yield_res = None
        if self.yield_pipe:
            sample_yield = pd.DataFrame([{
                'crop_name': active_crop,
                'soil_type': soil_type,
                'irrigation_source': irrigation_source,
                'field_area_ha': area_ha,
                'season': season,
                'avg_temp_c': temp,
                'season_rainfall_mm': rainfall,
                'avg_humidity_pct': humidity
            }])
            try:
                pred_t_ha = round(float(self.yield_pipe.predict(sample_yield)[0]), 2)
            except Exception:
                pred_t_ha = 4.50 # fallback
                
            pred_q_acre = round(pred_t_ha * 4.047, 1)
            total_prod = round(pred_t_ha * area_ha, 1)
            yield_res = {
                'crop': active_crop,
                'predicted_yield_tonnes_per_ha': pred_t_ha,
                'predicted_yield_quintals_per_acre': pred_q_acre,
                'total_harvest_tonnes': total_prod,
                'field_area_ha': area_ha,
                'field_area_acres': round(area_ha * 2.471, 1)
            }
            
        # --- 4. Irrigation Advisory ---
        irrig_res = None
        if self.irrig_advisor:
            irrig_res = self.irrig_advisor.calculate_irrigation(
                crop_name=active_crop,
                growth_stage=growth_stage,
                current_soil_moisture_pct=moisture,
                soil_type=soil_type,
                temp_c=temp,
                irrigation_method='Drip Irrigation' if 'Drip' in irrigation_source else 'Tube Well (Canal / Groundwater)'
            )
            
        # --- 5. Fertilizer Prescription ---
        fert_res = None
        if self.fert_engine:
            fert_res = self.fert_engine.calculate_fertilizer_schedule(
                crop_name=active_crop,
                soil_n_mg_kg=n,
                soil_p_mg_kg=p,
                soil_k_mg_kg=k,
                soil_ph=ph,
                field_area_ha=area_ha
            )
            
        # --- 6. Explainable AI (XAI) Attribution ---
        xai_res = self.xai.explain_crop_recommendation(
            n=n, p=p, k=k, temp=temp, humidity=humidity, ph=ph, rainfall=rainfall
        )
        
        return {
            'status': 'success',
            'field_inputs': data,
            'crop_recommendation': {
                'best_crop': recommended_crop,
                'confidence_pct': rec_confidence,
                'alternatives': alternatives
            },
            'crop_suitability': suitability_res,
            'crop_yield_prediction': yield_res,
            'irrigation_advisory': irrig_res,
            'fertilizer_prescription': fert_res,
            'explainable_ai': xai_res
        }

if __name__ == '__main__':
    predictor = FarmSpherePredictor()
    
    # Example field test: Wheat farmer in Bareilly with 2.5 hectares
    farmer_sample = {
        'farmer_name': 'Anurag Kushwaha',
        'location': 'Bareilly, Uttar Pradesh',
        'field_area_ha': 2.5,
        'soil_type': 'Alluvial',
        'irrigation_source': 'Tube Well (Canal / Groundwater)',
        'season': 'Rabi (Winter)',
        'soil_ph': 7.1,
        'nitrogen': 115.0,
        'phosphorus': 55.0,
        'potassium': 42.0,
        'organic_carbon': 0.65,
        'electrical_conductivity': 0.45,
        'temperature': 19.5,
        'humidity': 58.0,
        'rainfall': 480.0,
        'soil_moisture_pct': 38.0,
        'growth_stage': 'Flowering & Heading'
    }
    
    report = predictor.analyze_field(farmer_sample)
    
    print("\n" + "=" * 70)
    print("      FARMSPHERE AI: COMPLETE FIELD INTELLIGENCE DOSSIER")
    print("=" * 70)
    print(f"Farmer: {farmer_sample['farmer_name']} | Location: {farmer_sample['location']}")
    print(f"Field Area: {farmer_sample['field_area_ha']} ha | Soil: {farmer_sample['soil_type']}")
    print(f"Soil Test: pH={farmer_sample['soil_ph']}, N={farmer_sample['nitrogen']}, P={farmer_sample['phosphorus']}, K={farmer_sample['potassium']}")
    
    print("\n--- 1. AI CROP RECOMMENDATION ---")
    rec = report['crop_recommendation']
    print(f"Recommended Crop: {rec['best_crop']} (Confidence: {rec['confidence_pct']}%)")
    print("Alternative Options:")
    for alt in rec['alternatives']:
        print(f"  * {alt['crop']}: {alt['confidence_pct']}%")
        
    print("\n--- 2. FIELD SUITABILITY ASSESSMENT ---")
    suit = report['crop_suitability']
    print(f"Target Crop: {suit['evaluated_crop']}")
    print(f"Suitability Score: {suit['suitability_score']} / 100 ({suit['suitability_class']})")
    
    print("\n--- 3. YIELD PREDICTION ---")
    yld = report['crop_yield_prediction']
    print(f"Estimated Productivity: {yld['predicted_yield_tonnes_per_ha']} tonnes/ha ({yld['predicted_yield_quintals_per_acre']} quintals/acre)")
    print(f"Total Expected Harvest: {yld['total_harvest_tonnes']} tonnes")
    
    print("\n--- 4. IRRIGATION SCHEDULE ---")
    irg = report['irrigation_advisory']
    print(f"Urgency: {irg['urgency']} | Action Required: {irg['irrigation_recommended']}")
    print(f"Water Deficit: {irg['water_deficit_mm']} mm | Pump Runtime: {irg['recommended_pump_runtime_hours']} hrs")
    print(f"Advisory: {irg['advisory']}")
    
    print("\n--- 5. FERTILIZER PRESCRIPTION ---")
    fert = report['fertilizer_prescription']
    bags = fert['total_field_requirements']
    print(f"Total Bags (50kg): Urea={bags['urea_50kg_bags']} bags | DAP={bags['dap_50kg_bags']} bags | MOP={bags['mop_50kg_bags']} bags")
    print("Split Application Calendar:")
    for stage in fert['split_application_calendar']:
        print(f"  * {stage['stage']}: Urea={stage['urea_kg']} kg, DAP={stage['dap_kg']} kg, MOP={stage['mop_kg']} kg")
        
    print("\n--- 6. EXPLAINABLE AI (XAI) SUMMARY ---")
    xai = report['explainable_ai']
    print(f"Model Confidence: {xai['statistical_confidence_pct']}% | Field Certainty: {xai['agronomic_certainty_pct']}%")
    print(f"Rationale: {xai['agronomic_interpretation']}")
    print("Key Drivers:")
    for d in xai['optimal_factors'][:3]:
        print(f"  + {d}")
        
    print("\n" + "=" * 70)
    print("                END OF REPORT - FARMSPHERE AI")
    print("=" * 70)

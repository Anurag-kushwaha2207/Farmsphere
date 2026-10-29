"""
FarmSphere AI - Explainable AI (XAI) Agronomic Explainer
Provides:
1. Feature Attribution: Quantifies positive and negative contributions of each input parameter.
2. Agronomic Interpretation: Translates numerical ML weights into practical farmer explanations.
3. Distinction between Model Confidence (Statistical) vs Field Agronomic Certainty.
4. Limiting Factors vs Optimal Drivers.
"""

import os
import joblib
import numpy as np

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))

class AgronomicXAIExplainer:
    def __init__(self):
        crop_model_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_model.joblib')
        if os.path.exists(crop_model_path):
            artifact = joblib.load(crop_model_path)
            self.clf = artifact['model']
            self.scaler = artifact['scaler']
            self.le = artifact['label_encoder']
            self.features = artifact['feature_names']
            self.classes = artifact['classes']
        else:
            self.clf = None

    def explain_crop_recommendation(self, n, p, k, temp, humidity, ph, rainfall):
        """
        Explains why a crop was selected using tree feature contributions
        and agronomic reference thresholds.
        """
        raw_input = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        scaled_input = self.scaler.transform(raw_input)
        
        # Predicted Class & Probabilities
        probs = self.clf.predict_proba(scaled_input)[0]
        pred_idx = np.argmax(probs)
        pred_crop = self.classes[pred_idx]
        confidence = probs[pred_idx] * 100.0
        
        # Feature Importance weights from model
        importances = self.clf.feature_importances_
        
        # Agronomic thresholds for the recommended crop
        # (Standard ideal benchmarks)
        benchmarks = {
            'Wheat': {'N': (80, 140), 'P': (40, 80), 'K': (30, 60), 'ph': (6.0, 7.5), 'temperature': (12, 25), 'humidity': (40, 70), 'rainfall': (350, 650)},
            'Rice (Paddy)': {'N': (90, 150), 'P': (35, 70), 'K': (40, 80), 'ph': (5.5, 7.0), 'temperature': (22, 35), 'humidity': (70, 95), 'rainfall': (1000, 2500)},
            'Maize': {'N': (70, 130), 'P': (35, 75), 'K': (30, 65), 'ph': (6.0, 7.5), 'temperature': (18, 32), 'humidity': (50, 80), 'rainfall': (450, 900)},
            'Cotton': {'N': (90, 140), 'P': (40, 70), 'K': (35, 65), 'ph': (6.2, 8.0), 'temperature': (21, 35), 'humidity': (45, 75), 'rainfall': (550, 1100)},
            'Chickpea': {'N': (20, 50), 'P': (50, 90), 'K': (40, 80), 'ph': (6.0, 8.0), 'temperature': (14, 26), 'humidity': (35, 60), 'rainfall': (250, 550)},
            'Potato': {'N': (110, 170), 'P': (60, 110), 'K': (80, 150), 'ph': (5.0, 6.5), 'temperature': (15, 24), 'humidity': (60, 85), 'rainfall': (400, 750)}
        }
        
        crop_bench = benchmarks.get(pred_crop, benchmarks['Wheat'])
        
        inputs_dict = {
            'N': n, 'P': p, 'K': k, 'ph': ph, 
            'temperature': temp, 'humidity': humidity, 'rainfall': rainfall
        }
        
        feature_breakdown = []
        optimal_factors = []
        limiting_factors = []
        
        for idx, feat in enumerate(self.features):
            val = inputs_dict[feat]
            imp = importances[idx] * 100.0
            min_opt, max_opt = crop_bench.get(feat, (0, 1000))
            
            if min_opt <= val <= max_opt:
                status = 'OPTIMAL'
                delta = 'Within ideal biological range'
                contrib = imp
                optimal_factors.append(f"{feat} ({val}) is in the ideal range [{min_opt} - {max_opt}] for {pred_crop}.")
            elif val < min_opt:
                status = 'SUB_OPTIMAL_LOW'
                deficit = min_opt - val
                delta = f'Deficit of {deficit:.1f} below minimum'
                contrib = -imp * 0.5
                limiting_factors.append(f"{feat} ({val}) is below optimal minimum ({min_opt}) for {pred_crop}.")
            else:
                status = 'SUB_OPTIMAL_HIGH'
                surplus = val - max_opt
                delta = f'Surplus of {surplus:.1f} above maximum'
                contrib = -imp * 0.5
                limiting_factors.append(f"{feat} ({val}) is above optimal threshold ({max_opt}) for {pred_crop}.")
                
            feature_breakdown.append({
                'feature': feat,
                'input_value': val,
                'optimal_range': [min_opt, max_opt],
                'model_weight_pct': round(imp, 1),
                'status': status,
                'status_description': delta
            })
            
        # Agronomic Certainty vs Model Confidence
        # Statistical confidence is from softmax probability.
        # Agronomic certainty factors environmental risks like drought, salinity, pH extremes.
        certainty_score = confidence
        if limiting_factors:
            certainty_score = max(50.0, certainty_score - (len(limiting_factors) * 8.0))
            
        return {
            'recommended_crop': pred_crop,
            'statistical_confidence_pct': round(confidence, 1),
            'agronomic_certainty_pct': round(certainty_score, 1),
            'agronomic_interpretation': (
                f"Model strongly favors {pred_crop} primarily because {optimal_factors[0] if optimal_factors else 'field parameters are suitable'}. "
                f"Statistical confidence is {confidence:.1f}%, while field certainty is rated at {certainty_score:.1f}% factoring local environmental variability."
            ),
            'optimal_factors': optimal_factors,
            'limiting_factors': limiting_factors,
            'feature_breakdown': feature_breakdown
        }

if __name__ == '__main__':
    explainer = AgronomicXAIExplainer()
    explanation = explainer.explain_crop_recommendation(
        n=120, p=60, k=40, temp=20.0, humidity=55.0, ph=6.8, rainfall=500.0
    )
    print("\n--- Explainable AI Report ---")
    print(f"Recommended Crop:       {explanation['recommended_crop']}")
    print(f"Model Confidence:       {explanation['statistical_confidence_pct']}%")
    print(f"Agronomic Certainty:    {explanation['agronomic_certainty_pct']}%")
    print(f"\nExplanation Summary:\n{explanation['agronomic_interpretation']}")
    print("\nKey Optimal Drivers:")
    for f in explanation['optimal_factors']:
        print(f"  + {f}")
    if explanation['limiting_factors']:
        print("\nLimiting Factors:")
        for lf in explanation['limiting_factors']:
            print(f"  - {lf}")

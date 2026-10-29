"""
FarmSphere AI - Transparent Crop Suitability Rule Engine
Replaces empirical ML approximations with an interpretable, agronomic rule engine
calibrated from published ICAR and FAO crop requirements.

Features:
- Score 0 to 100 based on biological thresholds.
- Exhaustive limiting factor breakdown with exact penalties and ideal ranges.
- Covers all supported crops (Wheat, Rice, Maize, Cotton, Chickpea, Mustard, Potato).
- Explicit ValueError on unsupported crops (no silent fallbacks).
"""

import os
import json
from typing import Dict, Any, List

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'suitability_config.json')

class CropSuitabilityEngine:
    def __init__(self, config_path: str = CONFIG_PATH):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing suitability config at: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
    def get_supported_crops(self) -> List[str]:
        return list(self.config.keys())

    def evaluate_suitability(
        self,
        crop_name: str,
        soil_ph: float,
        nitrogen_mg_kg: float,
        phosphorus_mg_kg: float,
        potassium_mg_kg: float,
        electrical_conductivity_ds_m: float = 0.5,
        soil_type: str = 'Alluvial',
        temperature_c: float = 22.0,
        water_available_mm: float = 500.0
    ) -> Dict[str, Any]:
        """
        Evaluates field suitability for the target crop.
        Raises ValueError if crop is unsupported (no silent fallback).
        """
        # Normalize crop name lookup
        matched_crop = None
        for c in self.config.keys():
            if c.lower() == crop_name.strip().lower() or (c.split()[0].lower() == crop_name.strip().lower()):
                matched_crop = c
                break
                
        if not matched_crop:
            supported = ", ".join(self.config.keys())
            raise ValueError(
                f"Unsupported crop for suitability evaluation: '{crop_name}'. "
                f"Supported crops are: [{supported}]."
            )
            
        crop_cfg = self.config[matched_crop]
        score = 100.0
        limiting_factors = []
        optimal_factors = []

        # -------------------------------------------------------------
        # 1. Soil pH Factor
        # -------------------------------------------------------------
        ph_cfg = crop_cfg['ph']
        if ph_cfg['opt_min'] <= soil_ph <= ph_cfg['opt_max']:
            optimal_factors.append(f"Soil pH ({soil_ph:.2f}) is in the optimal range [{ph_cfg['opt_min']} - {ph_cfg['opt_max']}].")
        elif soil_ph < ph_cfg['min']:
            deficit = ph_cfg['min'] - soil_ph
            penalty = min(ph_cfg['max_penalty'], 20.0 + (deficit * 12.0))
            score -= penalty
            limiting_factors.append({
                'parameter': 'Soil pH',
                'actual_value': round(soil_ph, 2),
                'ideal_range': [ph_cfg['opt_min'], ph_cfg['opt_max']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Soil pH ({soil_ph:.2f}) is severely acidic (below absolute minimum {ph_cfg['min']})."
            })
        elif soil_ph > ph_cfg['max']:
            surplus = soil_ph - ph_cfg['max']
            penalty = min(ph_cfg['max_penalty'], 20.0 + (surplus * 12.0))
            score -= penalty
            limiting_factors.append({
                'parameter': 'Soil pH',
                'actual_value': round(soil_ph, 2),
                'ideal_range': [ph_cfg['opt_min'], ph_cfg['opt_max']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Soil pH ({soil_ph:.2f}) is excessively alkaline (above absolute maximum {ph_cfg['max']})."
            })
        else:
            # Sub-optimal intermediate
            dist = min(abs(soil_ph - ph_cfg['opt_min']), abs(soil_ph - ph_cfg['opt_max']))
            penalty = min(12.0, dist * 10.0)
            score -= penalty
            limiting_factors.append({
                'parameter': 'Soil pH',
                'actual_value': round(soil_ph, 2),
                'ideal_range': [ph_cfg['opt_min'], ph_cfg['opt_max']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Soil pH ({soil_ph:.2f}) is slightly outside optimal range."
            })

        # -------------------------------------------------------------
        # 2. Nitrogen (N) Fertility
        # -------------------------------------------------------------
        n_cfg = crop_cfg['nitrogen_mg_kg']
        if nitrogen_mg_kg >= n_cfg['opt']:
            optimal_factors.append(f"Available Nitrogen ({nitrogen_mg_kg:.1f} mg/kg) is adequate (optimum >= {n_cfg['opt']}).")
        elif nitrogen_mg_kg >= n_cfg['min']:
            deficit_pct = (n_cfg['opt'] - nitrogen_mg_kg) / (n_cfg['opt'] - n_cfg['min'])
            penalty = round(deficit_pct * 8.0, 1)
            score -= penalty
            limiting_factors.append({
                'parameter': 'Nitrogen',
                'actual_value': round(nitrogen_mg_kg, 1),
                'ideal_range': [n_cfg['min'], n_cfg['opt']],
                'penalty_applied': penalty,
                'reason': f"Soil Nitrogen ({nitrogen_mg_kg:.1f} mg/kg) is moderate; supplemental basal N recommended."
            })
        else:
            deficit_pct = (n_cfg['min'] - nitrogen_mg_kg) / n_cfg['min']
            penalty = min(n_cfg['max_penalty'], 10.0 + (deficit_pct * 10.0))
            score -= penalty
            limiting_factors.append({
                'parameter': 'Nitrogen',
                'actual_value': round(nitrogen_mg_kg, 1),
                'ideal_range': [n_cfg['min'], n_cfg['opt']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Severe Nitrogen deficiency ({nitrogen_mg_kg:.1f} mg/kg is below minimum {n_cfg['min']})."
            })

        # -------------------------------------------------------------
        # 3. Phosphorus (P) Fertility
        # -------------------------------------------------------------
        p_cfg = crop_cfg['phosphorus_mg_kg']
        if phosphorus_mg_kg >= p_cfg['opt']:
            optimal_factors.append(f"Available Phosphorus ({phosphorus_mg_kg:.1f} mg/kg) meets optimal target.")
        elif phosphorus_mg_kg < p_cfg['min']:
            penalty = min(p_cfg['max_penalty'], 12.0)
            score -= penalty
            limiting_factors.append({
                'parameter': 'Phosphorus',
                'actual_value': round(phosphorus_mg_kg, 1),
                'ideal_range': [p_cfg['min'], p_cfg['opt']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Soil Phosphorus ({phosphorus_mg_kg:.1f} mg/kg) is deficient."
            })

        # -------------------------------------------------------------
        # 4. Potassium (K) Fertility
        # -------------------------------------------------------------
        k_cfg = crop_cfg['potassium_mg_kg']
        if potassium_mg_kg >= k_cfg['opt']:
            optimal_factors.append(f"Available Potassium ({potassium_mg_kg:.1f} mg/kg) meets optimal target.")
        elif potassium_mg_kg < k_cfg['min']:
            penalty = min(k_cfg['max_penalty'], 10.0)
            score -= penalty
            limiting_factors.append({
                'parameter': 'Potassium',
                'actual_value': round(potassium_mg_kg, 1),
                'ideal_range': [k_cfg['min'], k_cfg['opt']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Soil Potassium ({potassium_mg_kg:.1f} mg/kg) is deficient."
            })

        # -------------------------------------------------------------
        # 5. Salinity (Electrical Conductivity EC)
        # -------------------------------------------------------------
        ec_cfg = crop_cfg['ec_ds_m']
        if electrical_conductivity_ds_m <= ec_cfg['max_safe']:
            optimal_factors.append(f"Soil Salinity EC ({electrical_conductivity_ds_m:.2f} dS/m) is safe (< {ec_cfg['max_safe']}).")
        else:
            excess = electrical_conductivity_ds_m - ec_cfg['max_safe']
            penalty = min(ec_cfg['max_penalty'], excess * ec_cfg['penalty_per_unit'])
            score -= penalty
            limiting_factors.append({
                'parameter': 'Electrical Conductivity (Salinity)',
                'actual_value': round(electrical_conductivity_ds_m, 2),
                'ideal_range': [0.0, ec_cfg['max_safe']],
                'penalty_applied': round(penalty, 1),
                'reason': f"High salinity (EC {electrical_conductivity_ds_m:.2f} dS/m exceeds safe threshold {ec_cfg['max_safe']})."
            })

        # -------------------------------------------------------------
        # 6. Temperature Factor
        # -------------------------------------------------------------
        t_cfg = crop_cfg['temperature_c']
        if t_cfg['opt_min'] <= temperature_c <= t_cfg['opt_max']:
            optimal_factors.append(f"Average Temperature ({temperature_c:.1f}°C) is in the optimal growth window.")
        elif temperature_c < t_cfg['min'] or temperature_c > t_cfg['max']:
            penalty = min(t_cfg['max_penalty'], 22.0)
            score -= penalty
            limiting_factors.append({
                'parameter': 'Temperature',
                'actual_value': round(temperature_c, 1),
                'ideal_range': [t_cfg['opt_min'], t_cfg['opt_max']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Temperature ({temperature_c:.1f}°C) is outside viable growing limits [{t_cfg['min']} - {t_cfg['max']}]."
            })
        else:
            penalty = 8.0
            score -= penalty
            limiting_factors.append({
                'parameter': 'Temperature',
                'actual_value': round(temperature_c, 1),
                'ideal_range': [t_cfg['opt_min'], t_cfg['opt_max']],
                'penalty_applied': penalty,
                'reason': f"Temperature ({temperature_c:.1f}°C) is slightly sub-optimal."
            })

        # -------------------------------------------------------------
        # 7. Water Availability (Rainfall + Irrigation)
        # -------------------------------------------------------------
        w_cfg = crop_cfg['water_available_mm']
        if water_available_mm >= w_cfg['opt']:
            optimal_factors.append(f"Total Water Availability ({water_available_mm:.0f} mm) satisfies crop demand.")
        elif water_available_mm < w_cfg['min']:
            deficit = w_cfg['min'] - water_available_mm
            penalty = min(w_cfg['max_penalty'], 15.0 + (deficit / w_cfg['min'] * 20.0))
            score -= penalty
            limiting_factors.append({
                'parameter': 'Water Availability',
                'actual_value': round(water_available_mm, 0),
                'ideal_range': [w_cfg['min'], w_cfg['opt']],
                'penalty_applied': round(penalty, 1),
                'reason': f"Water deficit: {water_available_mm:.0f} mm available vs {w_cfg['min']:.0f} mm required."
            })
        else:
            penalty = 8.0
            score -= penalty
            limiting_factors.append({
                'parameter': 'Water Availability',
                'actual_value': round(water_available_mm, 0),
                'ideal_range': [w_cfg['min'], w_cfg['opt']],
                'penalty_applied': penalty,
                'reason': f"Water availability ({water_available_mm:.0f} mm) is slightly below optimal {w_cfg['opt']} mm."
            })

        # -------------------------------------------------------------
        # 8. Soil Texture & Type Compatibility
        # -------------------------------------------------------------
        pref_soils = crop_cfg.get('preferred_soils', [])
        unfav_soils = crop_cfg.get('unfavorable_soils', [])
        if any(ps.lower() in soil_type.lower() for ps in pref_soils):
            optimal_factors.append(f"Soil Type '{soil_type}' is highly favored for {matched_crop}.")
        elif any(us.lower() in soil_type.lower() for us in unfav_soils):
            penalty = crop_cfg.get('soil_penalty', 15.0)
            score -= penalty
            limiting_factors.append({
                'parameter': 'Soil Type',
                'actual_value': soil_type,
                'ideal_range': pref_soils,
                'penalty_applied': round(penalty, 1),
                'reason': f"Soil Type '{soil_type}' is unfavorable for {matched_crop}."
            })

        # -------------------------------------------------------------
        # Final Score Clamping & Classification
        # -------------------------------------------------------------
        final_score = round(max(0.0, min(100.0, float(score))), 1)
        
        if final_score >= 85.0:
            suitability_class = 'Highly Suitable'
        elif final_score >= 70.0:
            suitability_class = 'Moderately Suitable'
        elif final_score >= 50.0:
            suitability_class = 'Marginally Suitable'
        else:
            suitability_class = 'Unsuitable'

        return {
            'crop': matched_crop,
            'suitability_score': final_score,
            'suitability_class': suitability_class,
            'is_suitable': final_score >= 70.0,
            'limiting_factors': limiting_factors,
            'optimal_factors': optimal_factors
        }

if __name__ == '__main__':
    engine = CropSuitabilityEngine()
    print("Supported crops:", engine.get_supported_crops())
    
    # Test Ideal Wheat
    wheat_res = engine.evaluate_suitability(
        crop_name='Wheat',
        soil_ph=7.0,
        nitrogen_mg_kg=110.0,
        phosphorus_mg_kg=45.0,
        potassium_mg_kg=55.0,
        electrical_conductivity_ds_m=0.4,
        soil_type='Alluvial',
        temperature_c=19.0,
        water_available_mm=500.0
    )
    print("\n--- Ideal Wheat Test ---")
    print(f"Score: {wheat_res['suitability_score']} ({wheat_res['suitability_class']})")
    
    # Test Unsuitable Wheat (Sandy + Salty + Acidic)
    unsuitable_res = engine.evaluate_suitability(
        crop_name='Wheat',
        soil_ph=5.0,
        nitrogen_mg_kg=30.0,
        phosphorus_mg_kg=10.0,
        potassium_mg_kg=15.0,
        electrical_conductivity_ds_m=2.5,
        soil_type='Sandy Loam',
        temperature_c=12.0,
        water_available_mm=200.0
    )
    print("\n--- Unsuitable Wheat Test ---")
    print(f"Score: {unsuitable_res['suitability_score']} ({unsuitable_res['suitability_class']})")
    print("Limiting factors count:", len(unsuitable_res['limiting_factors']))

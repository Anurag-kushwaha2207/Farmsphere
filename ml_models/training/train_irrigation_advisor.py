"""
FarmSphere AI - Precision Irrigation & Water Requirement Engine
Based on FAO-56 Penman-Monteith & Crop Coefficient (Kc) Equations:
    ETc = ETo * Kc
    Depletion = (Field Capacity - Current Moisture) / Field Capacity
    Water Deficit (mm) = Root Depth * (Field Capacity - Current Moisture) * Soil Texture Factor
"""

import os
import json
import joblib

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

class IrrigationAdvisor:
    def __init__(self):
        # FAO Crop Growth Stages and Kc Coefficients
        self.crop_kc_database = {
            'Wheat': {
                'Initial / Crown Root Initiation': {'kc': 0.40, 'root_depth_m': 0.3, 'critical': False},
                'Tillering & Vegetative': {'kc': 0.80, 'root_depth_m': 0.6, 'critical': False},
                'Flowering & Heading': {'kc': 1.15, 'root_depth_m': 0.9, 'critical': True},
                'Grain Filling & Milking': {'kc': 1.05, 'root_depth_m': 1.0, 'critical': True},
                'Ripening / Maturity': {'kc': 0.45, 'root_depth_m': 1.0, 'critical': False}
            },
            'Rice (Paddy)': {
                'Transplanting & Nursery': {'kc': 1.10, 'root_depth_m': 0.2, 'critical': True},
                'Active Tillering': {'kc': 1.20, 'root_depth_m': 0.4, 'critical': True},
                'Panicle Initiation & Flowering': {'kc': 1.35, 'root_depth_m': 0.6, 'critical': True},
                'Grain Milking': {'kc': 1.15, 'root_depth_m': 0.6, 'critical': True},
                'Maturity & Pre-harvest Drainage': {'kc': 0.70, 'root_depth_m': 0.6, 'critical': False}
            },
            'Maize': {
                'Seedling & Early Vegetative': {'kc': 0.45, 'root_depth_m': 0.3, 'critical': False},
                'Rapid Vegetative Growth': {'kc': 0.85, 'root_depth_m': 0.7, 'critical': False},
                'Tasseling & Silking': {'kc': 1.20, 'root_depth_m': 1.0, 'critical': True},
                'Grain Fill & Dough': {'kc': 1.00, 'root_depth_m': 1.1, 'critical': True},
                'Physiological Maturity': {'kc': 0.55, 'root_depth_m': 1.1, 'critical': False}
            },
            'Cotton': {
                'Germination & Emergence': {'kc': 0.45, 'root_depth_m': 0.3, 'critical': False},
                'Square Formation': {'kc': 0.75, 'root_depth_m': 0.8, 'critical': False},
                'Peak Flowering & Boll Setting': {'kc': 1.15, 'root_depth_m': 1.2, 'critical': True},
                'Boll Development': {'kc': 0.85, 'root_depth_m': 1.2, 'critical': True},
                'Maturity & Defoliation': {'kc': 0.50, 'root_depth_m': 1.2, 'critical': False}
            }
        }
        
        # Soil Available Water Capacity (AWC in mm/m)
        self.soil_water_properties = {
            'Alluvial': {'field_capacity_pct': 58, 'wilting_point_pct': 22, 'awc_mm_per_m': 160},
            'Clay Loam': {'field_capacity_pct': 65, 'wilting_point_pct': 26, 'awc_mm_per_m': 180},
            'Clayey': {'field_capacity_pct': 75, 'wilting_point_pct': 32, 'awc_mm_per_m': 200},
            'Sandy Loam': {'field_capacity_pct': 42, 'wilting_point_pct': 16, 'awc_mm_per_m': 110},
            'Black Soil': {'field_capacity_pct': 70, 'wilting_point_pct': 30, 'awc_mm_per_m': 190}
        }

    def compute_eto_hargreaves(self, temp_c, temp_min=None, temp_max=None, solar_radiation=18.0):
        """Simplified Hargreaves / Blaney-Criddle Reference Evapotranspiration (mm/day)"""
        if temp_max is None or temp_min is None:
            t_max = temp_c + 6.0
            t_min = temp_c - 6.0
        else:
            t_max = temp_max
            t_min = temp_min
            
        t_range = max(t_max - t_min, 2.0)
        eto = 0.0023 * (temp_c + 17.8) * (t_range ** 0.5) * (solar_radiation / 2.45)
        return round(float(max(eto, 1.5)), 2)

    def calculate_irrigation(self, crop_name, growth_stage, current_soil_moisture_pct, 
                             soil_type='Alluvial', temp_c=22.0, recent_rainfall_mm=0.0, 
                             irrigation_method='Drip Irrigation'):
        # 1. Lookup Kc and Root Depth
        crop_db = self.crop_kc_database.get(crop_name, self.crop_kc_database['Wheat'])
        stage_info = crop_db.get(growth_stage, list(crop_db.values())[1])
        kc = stage_info['kc']
        root_depth = stage_info['root_depth_m']
        is_critical = stage_info['critical']
        
        # 2. Lookup Soil Properties
        soil_props = self.soil_water_properties.get(soil_type, self.soil_water_properties['Alluvial'])
        fc = soil_props['field_capacity_pct']
        wp = soil_props['wilting_point_pct']
        awc = soil_props['awc_mm_per_m']
        
        # 3. Reference and Crop Evapotranspiration
        eto = self.compute_eto_hargreaves(temp_c)
        etc = round(float(eto * kc), 2)
        
        # 4. Total Available Water (TAW) & Readily Available Water (RAW)
        # Maximum Allowable Depletion (MAD): 0.50 for cereals, 0.35 for critical flowering
        mad = 0.35 if is_critical else 0.50
        taw = awc * root_depth
        raw = taw * mad
        
        # 5. Soil Moisture Depletion
        moisture_deficit_pct = max(0.0, fc - current_soil_moisture_pct)
        water_deficit_mm = round(float((moisture_deficit_pct / 100.0) * root_depth * 1000.0), 1)
        
        # Factor recent rainfall
        effective_rainfall = recent_rainfall_mm * 0.75
        net_irrigation_needed_mm = max(0.0, water_deficit_mm - effective_rainfall)
        
        # Irrigation Efficiency factor
        eff = 0.90 if 'Drip' in irrigation_method else (0.75 if 'Sprinkler' in irrigation_method else 0.60)
        gross_irrigation_mm = round(net_irrigation_needed_mm / eff, 1)
        
        # Recommendation Logic
        needs_irrigation = current_soil_moisture_pct < (fc - (fc - wp) * mad)
        urgency = 'CRITICAL' if (needs_irrigation and is_critical) else ('HIGH' if needs_irrigation else 'NORMAL')
        
        # Runtime estimation (assuming standard 10 mm/hr drip or 25 mm/hr tube well delivery)
        delivery_rate_mm_hr = 8.0 if 'Drip' in irrigation_method else 20.0
        pump_hours = round(gross_irrigation_mm / delivery_rate_mm_hr, 1) if needs_irrigation else 0.0
        
        days_until_next = round(max(1, (raw - water_deficit_mm) / max(etc, 1.0)))
        
        return {
            'crop': crop_name,
            'growth_stage': growth_stage,
            'is_critical_stage': is_critical,
            'reference_eto_mm_day': eto,
            'crop_etc_mm_day': etc,
            'current_soil_moisture_pct': current_soil_moisture_pct,
            'field_capacity_pct': fc,
            'water_deficit_mm': water_deficit_mm,
            'net_irrigation_required_mm': net_irrigation_needed_mm,
            'gross_irrigation_required_mm': gross_irrigation_mm,
            'irrigation_recommended': needs_irrigation,
            'urgency': urgency,
            'recommended_pump_runtime_hours': pump_hours,
            'days_until_next_irrigation': max(1, days_until_next),
            'irrigation_method': irrigation_method,
            'application_efficiency_pct': int(eff * 100),
            'advisory': f"{urgency} ALERT: Water deficit is {water_deficit_mm} mm. Run {irrigation_method} for {pump_hours} hrs." if needs_irrigation else "Soil moisture is currently adequate. No irrigation needed today."
        }

def save_and_test():
    advisor = IrrigationAdvisor()
    save_path = os.path.join(SAVED_MODELS_DIR, 'irrigation_advisor.joblib')
    joblib.dump(advisor, save_path)
    print(f"Irrigation advisor saved to: {save_path}")
    
    # Test sample
    test_res = advisor.calculate_irrigation(
        crop_name='Wheat',
        growth_stage='Flowering & Heading',
        current_soil_moisture_pct=34.0, # Below optimal 55%
        soil_type='Alluvial',
        temp_c=22.0,
        recent_rainfall_mm=0.0,
        irrigation_method='Drip Irrigation'
    )
    print("\n--- Sample Irrigation Advisory ---")
    for k, v in test_res.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    save_and_test()

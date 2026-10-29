"""
FarmSphere AI - Precision Fertilizer & Nutrient Management Engine
Based on ICAR Soil Test Crop Response (STCR) & Agronomic Nutrient Balance Equations:
    Nutrient Deficit = Target Crop Requirement - Available Soil Nutrient * Soil Efficiency
    Fertilizer Quantity = Nutrient Deficit / Nutrient Concentration in Fertilizer
Calculates exact kg of Urea, DAP, MOP per hectare and per acre, with split application timing.
"""

import os
import json
import joblib

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

class FertilizerEngine:
    def __init__(self):
        # Target Crop Nutrient Requirements (kg/ha) for high yield potential
        self.crop_requirements = {
            'Wheat': {'target_N': 120, 'target_P': 60, 'target_K': 40, 'splits': [('Basal (Sowing)', 0.50, 1.0, 1.0), ('1st Top Dressing (30 DAS)', 0.25, 0.0, 0.0), ('2nd Top Dressing (50 DAS)', 0.25, 0.0, 0.0)]},
            'Rice (Paddy)': {'target_N': 130, 'target_P': 50, 'target_K': 60, 'splits': [('Basal (Transplanting)', 0.34, 1.0, 0.5), ('Active Tillering (25 DAT)', 0.33, 0.0, 0.0), ('Panicle Initiation (55 DAT)', 0.33, 0.0, 0.5)]},
            'Maize': {'target_N': 120, 'target_P': 60, 'target_K': 50, 'splits': [('Basal (Sowing)', 0.33, 1.0, 1.0), ('Knee High (30 DAS)', 0.33, 0.0, 0.0), ('Tasseling (50 DAS)', 0.34, 0.0, 0.0)]},
            'Cotton': {'target_N': 100, 'target_P': 50, 'target_K': 50, 'splits': [('Basal (Sowing)', 0.25, 1.0, 1.0), ('Square Formation (35 DAS)', 0.50, 0.0, 0.0), ('Boll Setting (65 DAS)', 0.25, 0.0, 0.0)]},
            'Chickpea': {'target_N': 25, 'target_P': 60, 'target_K': 30, 'splits': [('Basal (at Sowing)', 1.0, 1.0, 1.0)]},
            'Potato': {'target_N': 150, 'target_P': 80, 'target_K': 120, 'splits': [('Basal (at Planting)', 0.50, 1.0, 0.5), ('At Earthing Up (35 DAP)', 0.50, 0.0, 0.5)]}
        }
        
        # Nutrient contents in commercial fertilizers
        # Urea: 46% N
        # DAP: 18% N, 46% P2O5
        # MOP: 60% K2O
        self.fertilizers = {
            'urea': {'N': 0.46, 'P': 0.00, 'K': 0.00, 'name': 'Urea (46% N)'},
            'dap': {'N': 0.18, 'P': 0.46, 'K': 0.00, 'name': 'DAP (18-46-0)'},
            'mop': {'N': 0.00, 'P': 0.00, 'K': 0.60, 'name': 'MOP (60% K2O)'}
        }

    def calculate_fertilizer_schedule(self, crop_name, soil_n_mg_kg, soil_p_mg_kg, soil_k_mg_kg, 
                                      soil_ph=7.0, organic_carbon=0.6, field_area_ha=1.0):
        # 1. Convert soil test mg/kg to approx available kg/ha (depth 15cm, bulk density 1.35 g/cm3 -> factor ~ 2.0)
        avail_n_kg_ha = soil_n_mg_kg * 0.75  # Mineralizable N availability
        avail_p_kg_ha = soil_p_mg_kg * 0.50  # Bray/Olsen P availability
        avail_k_kg_ha = soil_k_mg_kg * 0.45  # Ammonium acetate K availability
        
        crop_req = self.crop_requirements.get(crop_name, self.crop_requirements['Wheat'])
        
        # 2. Net nutrient deficit (accounting for soil supply efficiency)
        net_n_def = max(10.0, crop_req['target_N'] - (avail_n_kg_ha * 0.35))
        net_p_def = max(5.0, crop_req['target_P'] - (avail_p_kg_ha * 0.40))
        net_k_def = max(5.0, crop_req['target_K'] - (avail_k_kg_ha * 0.40))
        
        # 3. Commercial Fertilizer Calculation
        # First fulfill P deficit through DAP (18-46-0)
        dap_kg_ha = net_p_def / self.fertilizers['dap']['P']
        n_supplied_by_dap = dap_kg_ha * self.fertilizers['dap']['N']
        
        # Remaining N deficit fulfilled through Urea (46% N)
        remaining_n_def = max(0.0, net_n_def - n_supplied_by_dap)
        urea_kg_ha = remaining_n_def / self.fertilizers['urea']['N']
        
        # Potassium fulfilled through MOP (60% K2O)
        mop_kg_ha = net_k_def / self.fertilizers['mop']['K']
        
        # Conversion to Acre and Field Area
        HA_TO_ACRE = 2.47105
        dap_kg_acre = dap_kg_ha / HA_TO_ACRE
        urea_kg_acre = urea_kg_ha / HA_TO_ACRE
        mop_kg_acre = mop_kg_ha / HA_TO_ACRE
        
        total_dap_field = dap_kg_ha * field_area_ha
        total_urea_field = urea_kg_ha * field_area_ha
        total_mop_field = mop_kg_ha * field_area_ha
        
        # Standard 50kg bag equivalents
        dap_bags = round(total_dap_field / 50.0, 1)
        urea_bags = round(total_urea_field / 50.0, 1)
        mop_bags = round(total_mop_field / 50.0, 1)
        
        # 4. Split Application Schedule
        split_details = []
        for stage_name, n_pct, p_pct, k_pct in crop_req['splits']:
            stage_urea = round(total_urea_field * n_pct, 1)
            stage_dap = round(total_dap_field * p_pct, 1)
            stage_mop = round(total_mop_field * k_pct, 1)
            split_details.append({
                'stage': stage_name,
                'urea_kg': stage_urea,
                'dap_kg': stage_dap,
                'mop_kg': stage_mop,
                'instructions': f"Apply {stage_dap} kg DAP, {stage_urea} kg Urea, and {stage_mop} kg MOP evenly."
            })
            
        # 5. Soil Amendment Advisory
        amendment = None
        if soil_ph < 5.8:
            amendment = {
                'type': 'Agricultural Lime (CaCO3)',
                'rate_kg_ha': round((6.5 - soil_ph) * 1200, 0),
                'reason': f'Acidic soil (pH {soil_ph}). Lime neutralizes acidity and mobilizes phosphorus.'
            }
        elif soil_ph > 8.0:
            amendment = {
                'type': 'Agricultural Gypsum (CaSO4·2H2O)',
                'rate_kg_ha': round((soil_ph - 7.5) * 1500, 0),
                'reason': f'Alkaline / sodic soil (pH {soil_ph}). Gypsum replaces sodium with calcium.'
            }
            
        return {
            'crop': crop_name,
            'field_area_ha': field_area_ha,
            'field_area_acres': round(field_area_ha * HA_TO_ACRE, 2),
            'soil_test': {'N_mg_kg': soil_n_mg_kg, 'P_mg_kg': soil_p_mg_kg, 'K_mg_kg': soil_k_mg_kg, 'pH': soil_ph, 'OC_pct': organic_carbon},
            'nutrient_deficit_kg_ha': {
                'N': round(net_n_def, 1),
                'P2O5': round(net_p_def, 1),
                'K2O': round(net_k_def, 1)
            },
            'commercial_fertilizer_rate_per_ha': {
                'urea_kg_ha': round(urea_kg_ha, 1),
                'dap_kg_ha': round(dap_kg_ha, 1),
                'mop_kg_ha': round(mop_kg_ha, 1)
            },
            'commercial_fertilizer_rate_per_acre': {
                'urea_kg_acre': round(urea_kg_acre, 1),
                'dap_kg_acre': round(dap_kg_acre, 1),
                'mop_kg_acre': round(mop_kg_acre, 1)
            },
            'total_field_requirements': {
                'urea_kg': round(total_urea_field, 1),
                'dap_kg': round(total_dap_field, 1),
                'mop_kg': round(total_mop_field, 1),
                'urea_50kg_bags': urea_bags,
                'dap_50kg_bags': dap_bags,
                'mop_50kg_bags': mop_bags
            },
            'split_application_calendar': split_details,
            'soil_amendment_advisory': amendment
        }

def save_and_test():
    engine = FertilizerEngine()
    save_path = os.path.join(SAVED_MODELS_DIR, 'fertilizer_engine.joblib')
    joblib.dump(engine, save_path)
    print(f"Fertilizer engine saved to: {save_path}")
    
    test_res = engine.calculate_fertilizer_schedule(
        crop_name='Wheat',
        soil_n_mg_kg=75.0,  # Medium
        soil_p_mg_kg=22.0,  # Deficient
        soil_k_mg_kg=55.0,  # Medium
        soil_ph=7.8,
        organic_carbon=0.55,
        field_area_ha=2.5
    )
    print("\n--- Precision Fertilizer Prescription (Wheat, 2.5 ha) ---")
    print(f"Total Urea Needed: {test_res['total_field_requirements']['urea_kg']} kg ({test_res['total_field_requirements']['urea_50kg_bags']} bags)")
    print(f"Total DAP Needed:  {test_res['total_field_requirements']['dap_kg']} kg ({test_res['total_field_requirements']['dap_50kg_bags']} bags)")
    print(f"Total MOP Needed:  {test_res['total_field_requirements']['mop_kg']} kg ({test_res['total_field_requirements']['mop_50kg_bags']} bags)")
    print("\nSplit Application Stages:")
    for split in test_res['split_application_calendar']:
        print(f"  * {split['stage']}: Urea={split['urea_kg']} kg, DAP={split['dap_kg']} kg, MOP={split['mop_kg']} kg")

if __name__ == '__main__':
    save_and_test()

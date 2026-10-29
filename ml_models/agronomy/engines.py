"""
FarmSphere AI - Core Agronomic Engines
1. IrrigationAdvisor: Hargreaves-Samani Evapotranspiration (ETo) and Root Zone Depletion Model
2. FertilizerEngine: Nutrient-Balance Estimate for Commercial Fertilizer Schedules

Note: These engines provide decision-support estimates.
Field capacities reflect volumetric soil water content (22-40% vol).
Fertilizer schedules are nutrient-balance advisories, not empirical ICAR STCR regressions.
"""

from typing import Dict, Any, Optional

class IrrigationAdvisor:
    """
    Computes irrigation requirements using Hargreaves-Samani reference evapotranspiration (ETo),
    crop coefficients (Kc) per growth stage, and soil volumetric water balance.
    """
    def __init__(self):
        # FAO Crop Growth Stages and Kc Coefficients for all 7 supported crops
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
            'Cotton (Seed Cotton)': {
                'Germination & Emergence': {'kc': 0.45, 'root_depth_m': 0.3, 'critical': False},
                'Square Formation': {'kc': 0.75, 'root_depth_m': 0.8, 'critical': False},
                'Peak Flowering & Boll Setting': {'kc': 1.15, 'root_depth_m': 1.2, 'critical': True},
                'Boll Development': {'kc': 0.85, 'root_depth_m': 1.2, 'critical': True},
                'Maturity & Defoliation': {'kc': 0.50, 'root_depth_m': 1.2, 'critical': False}
            },
            'Chickpea': {
                'Germination & Seedling': {'kc': 0.40, 'root_depth_m': 0.3, 'critical': False},
                'Branching & Vegetative': {'kc': 0.70, 'root_depth_m': 0.6, 'critical': False},
                'Flowering & Pod Initiation': {'kc': 1.00, 'root_depth_m': 0.8, 'critical': True},
                'Pod Filling': {'kc': 0.85, 'root_depth_m': 0.9, 'critical': True},
                'Maturity': {'kc': 0.35, 'root_depth_m': 0.9, 'critical': False}
            },
            'Mustard': {
                'Seedling & Rosette': {'kc': 0.40, 'root_depth_m': 0.3, 'critical': False},
                'Stem Elongation': {'kc': 0.75, 'root_depth_m': 0.6, 'critical': False},
                'Flowering & Siliqua Formation': {'kc': 1.05, 'root_depth_m': 0.9, 'critical': True},
                'Seed Development': {'kc': 0.85, 'root_depth_m': 1.0, 'critical': True},
                'Maturity': {'kc': 0.40, 'root_depth_m': 1.0, 'critical': False}
            },
            'Potato': {
                'Emergence & Early Vegetative': {'kc': 0.50, 'root_depth_m': 0.25, 'critical': False},
                'Tuber Initiation': {'kc': 0.85, 'root_depth_m': 0.45, 'critical': True},
                'Tuber Bulking': {'kc': 1.15, 'root_depth_m': 0.60, 'critical': True},
                'Maturation & Vine Senescence': {'kc': 0.70, 'root_depth_m': 0.60, 'critical': False}
            }
        }
        
        # Volumetric Soil Water Properties (typical field ranges 20-42% vol)
        self.soil_water_properties = {
            'Alluvial': {'field_capacity_pct': 32.0, 'wilting_point_pct': 15.0, 'awc_mm_per_m': 170.0},
            'Clay Loam': {'field_capacity_pct': 36.0, 'wilting_point_pct': 19.0, 'awc_mm_per_m': 170.0},
            'Clayey': {'field_capacity_pct': 40.0, 'wilting_point_pct': 23.0, 'awc_mm_per_m': 170.0},
            'Sandy Loam': {'field_capacity_pct': 22.0, 'wilting_point_pct': 9.0, 'awc_mm_per_m': 130.0},
            'Black Soil': {'field_capacity_pct': 40.0, 'wilting_point_pct': 21.0, 'awc_mm_per_m': 190.0}
        }

    def compute_eto_hargreaves(self, temp_c: float, temp_min: Optional[float] = None, 
                               temp_max: Optional[float] = None, solar_radiation: float = 18.0) -> float:
        """
        Estimates Reference Evapotranspiration (ETo) using the Hargreaves-Samani temperature equation.
        Formula: ETo = 0.0023 * (T_mean + 17.8) * sqrt(T_max - T_min) * Ra / 2.45
        """
        if temp_max is None or temp_min is None:
            t_max = temp_c + 6.0
            t_min = temp_c - 6.0
        else:
            t_max = temp_max
            t_min = temp_min
            
        t_range = max(t_max - t_min, 2.0)
        eto = 0.0023 * (temp_c + 17.8) * (t_range ** 0.5) * (solar_radiation / 2.45)
        return round(float(max(eto, 1.5)), 2)

    def calculate_irrigation(
        self,
        crop_name: str,
        growth_stage: str,
        current_soil_moisture_pct: float,
        soil_type: str = 'Alluvial',
        temp_c: float = 22.0,
        recent_rainfall_mm: float = 0.0,
        irrigation_method: str = 'Drip Irrigation'
    ) -> Dict[str, Any]:
        """
        Calculates irrigation requirement and pump run time based on volumetric depletion.
        Raises ValueError if crop is not supported (no silent fallback).
        """
        matched_crop = None
        for c in self.crop_kc_database.keys():
            if c.lower() == crop_name.strip().lower() or c.split()[0].lower() == crop_name.strip().lower():
                matched_crop = c
                break
                
        if not matched_crop:
            supported = ", ".join(self.crop_kc_database.keys())
            raise ValueError(
                f"Unsupported crop for irrigation advisory: '{crop_name}'. "
                f"Supported crops are: [{supported}]."
            )
            
        crop_db = self.crop_kc_database[matched_crop]
        
        # Match growth stage
        matched_stage = None
        for stage in crop_db.keys():
            if stage.lower() in growth_stage.lower() or growth_stage.lower() in stage.lower():
                matched_stage = stage
                break
        if not matched_stage:
            matched_stage = list(crop_db.keys())[1] # fallback to vegetative within known crop
            
        stage_info = crop_db[matched_stage]
        kc = stage_info['kc']
        root_depth = stage_info['root_depth_m']
        is_critical = stage_info['critical']
        
        # Soil properties
        matched_soil = None
        for s in self.soil_water_properties.keys():
            if s.lower() in soil_type.lower():
                matched_soil = s
                break
        if not matched_soil:
            matched_soil = 'Alluvial'
            
        soil_props = self.soil_water_properties[matched_soil]
        fc = soil_props['field_capacity_pct']
        wp = soil_props['wilting_point_pct']
        awc = soil_props['awc_mm_per_m']
        
        eto = self.compute_eto_hargreaves(temp_c)
        etc = round(float(eto * kc), 2)
        
        mad = 0.35 if is_critical else 0.50
        taw = awc * root_depth
        raw = taw * mad
        
        moisture_deficit_pct = max(0.0, fc - current_soil_moisture_pct)
        water_deficit_mm = round(float((moisture_deficit_pct / 100.0) * root_depth * 1000.0), 1)
        
        effective_rainfall = recent_rainfall_mm * 0.75
        net_irrigation_needed_mm = max(0.0, water_deficit_mm - effective_rainfall)
        
        eff = 0.90 if 'Drip' in irrigation_method else (0.75 if 'Sprinkler' in irrigation_method else 0.60)
        gross_irrigation_mm = round(net_irrigation_needed_mm / eff, 1)
        
        threshold_moisture = fc - (fc - wp) * mad
        needs_irrigation = current_soil_moisture_pct < threshold_moisture
        urgency = 'CRITICAL' if (needs_irrigation and is_critical) else ('HIGH' if needs_irrigation else 'NORMAL')
        
        delivery_rate_mm_hr = 8.0 if 'Drip' in irrigation_method else 20.0
        pump_hours = round(gross_irrigation_mm / delivery_rate_mm_hr, 1) if needs_irrigation else 0.0
        days_until_next = round(max(1, (raw - water_deficit_mm) / max(etc, 1.0)))
        
        return {
            'crop': matched_crop,
            'growth_stage': matched_stage,
            'is_critical_stage': is_critical,
            'reference_eto_method': 'Hargreaves-Samani ETo estimation',
            'reference_eto_mm_day': eto,
            'crop_kc': kc,
            'crop_etc_mm_day': etc,
            'current_soil_moisture_pct': current_soil_moisture_pct,
            'field_capacity_pct': fc,
            'wilting_point_pct': wp,
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


class FertilizerEngine:
    """
    Computes nutrient-balance estimates for fertilizer requirements.
    Converts soil test values (mg/kg) to furrow slice available nutrient (kg/ha)
    using documented agronomic conversion factors.
    Output is labelled as advisory.
    """
    SOIL_PPM_TO_KG_HA = 2.0  # 1 mg/kg * 2.0 = kg/ha furrow slice (top 15 cm, BD ~1.33 g/cm3)
    
    # Fractional crop uptake availability from soil test pool
    EFF_SOIL_N = 0.30
    EFF_SOIL_P = 0.20
    EFF_SOIL_K = 0.35

    def __init__(self):
        # Recommended nutrient demand and split calendar for all 7 crops
        self.crop_requirements = {
            'Wheat': {
                'target_N': 120, 'target_P': 60, 'target_K': 40,
                'splits': [
                    ('Basal (Sowing)', 0.50, 1.0, 1.0),
                    ('1st Top Dressing (CRI Stage, 21 DAS)', 0.25, 0.0, 0.0),
                    ('2nd Top Dressing (Jointing Stage, 45 DAS)', 0.25, 0.0, 0.0)
                ]
            },
            'Rice (Paddy)': {
                'target_N': 130, 'target_P': 50, 'target_K': 60,
                'splits': [
                    ('Basal (Transplanting)', 0.34, 1.0, 0.50),
                    ('Active Tillering (25 DAT)', 0.33, 0.0, 0.00),
                    ('Panicle Initiation (55 DAT)', 0.33, 0.0, 0.50)
                ]
            },
            'Maize': {
                'target_N': 120, 'target_P': 60, 'target_K': 50,
                'splits': [
                    ('Basal (Sowing)', 0.33, 1.0, 1.0),
                    ('Knee High (30 DAS)', 0.33, 0.0, 0.0),
                    ('Tasseling (50 DAS)', 0.34, 0.0, 0.0)
                ]
            },
            'Cotton (Seed Cotton)': {
                'target_N': 100, 'target_P': 50, 'target_K': 50,
                'splits': [
                    ('Basal (Sowing)', 0.25, 1.0, 1.0),
                    ('Square Formation (35 DAS)', 0.50, 0.0, 0.0),
                    ('Boll Setting (65 DAS)', 0.25, 0.0, 0.0)
                ]
            },
            'Chickpea': {
                'target_N': 25, 'target_P': 50, 'target_K': 30,
                'splits': [
                    ('Basal (at Sowing)', 1.0, 1.0, 1.0)
                ]
            },
            'Mustard': {
                'target_N': 80, 'target_P': 40, 'target_K': 40,
                'splits': [
                    ('Basal (at Sowing)', 0.50, 1.0, 1.0),
                    ('Flowering Stage (35 DAS)', 0.50, 0.0, 0.0)
                ]
            },
            'Potato': {
                'target_N': 150, 'target_P': 80, 'target_K': 120,
                'splits': [
                    ('Basal (at Planting)', 0.50, 1.0, 0.50),
                    ('At Earthing Up (35 DAP)', 0.50, 0.0, 0.50)
                ]
            }
        }
        
        self.fertilizers = {
            'urea': {'N': 0.46, 'P': 0.00, 'K': 0.00, 'name': 'Urea (46% N)'},
            'dap':  {'N': 0.18, 'P': 0.46, 'K': 0.00, 'name': 'DAP (18-46-0)'},
            'mop':  {'N': 0.00, 'P': 0.00, 'K': 0.60, 'name': 'MOP (60% K2O)'}
        }

    def calculate_fertilizer_schedule(
        self,
        crop_name: str,
        soil_n_mg_kg: float,
        soil_p_mg_kg: float,
        soil_k_mg_kg: float,
        soil_ph: float = 7.0,
        organic_carbon: float = 0.6,
        field_area_ha: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates commercial fertilizer schedule based on nutrient balance.
        Raises ValueError on unsupported crop (no silent fallback).
        """
        matched_crop = None
        for c in self.crop_requirements.keys():
            if c.lower() == crop_name.strip().lower() or c.split()[0].lower() == crop_name.strip().lower():
                matched_crop = c
                break
                
        if not matched_crop:
            supported = ", ".join(self.crop_requirements.keys())
            raise ValueError(
                f"Unsupported crop for fertilizer recommendation: '{crop_name}'. "
                f"Supported crops are: [{supported}]."
            )
            
        crop_req = self.crop_requirements[matched_crop]
        
        # 1. Convert soil test mg/kg (ppm) to kg/ha furrow slice (top 15cm)
        soil_n_kg_ha = soil_n_mg_kg * self.SOIL_PPM_TO_KG_HA
        soil_p_kg_ha = soil_p_mg_kg * self.SOIL_PPM_TO_KG_HA
        soil_k_kg_ha = soil_k_mg_kg * self.SOIL_PPM_TO_KG_HA
        
        # 2. Compute effective crop uptake from soil pool
        avail_n_for_crop = soil_n_kg_ha * self.EFF_SOIL_N
        avail_p_for_crop = soil_p_kg_ha * self.EFF_SOIL_P
        avail_k_for_crop = soil_k_kg_ha * self.EFF_SOIL_K
        
        # 3. Net deficit to be supplied externally (clamped to >= 0)
        net_n_def = max(0.0, crop_req['target_N'] - avail_n_for_crop)
        net_p_def = max(0.0, crop_req['target_P'] - avail_p_for_crop)
        net_k_def = max(0.0, crop_req['target_K'] - avail_k_for_crop)
        
        # 4. Commercial fertilizer rate calculation
        # DAP supplies P2O5 (46%) and N (18%)
        dap_kg_ha = net_p_def / self.fertilizers['dap']['P']
        n_supplied_by_dap = dap_kg_ha * self.fertilizers['dap']['N']
        
        # Urea supplies remaining N
        remaining_n_def = max(0.0, net_n_def - n_supplied_by_dap)
        urea_kg_ha = remaining_n_def / self.fertilizers['urea']['N']
        
        # MOP supplies K2O (60%)
        mop_kg_ha = net_k_def / self.fertilizers['mop']['K']
        
        HA_TO_ACRE = 2.47105
        dap_kg_acre = dap_kg_ha / HA_TO_ACRE
        urea_kg_acre = urea_kg_ha / HA_TO_ACRE
        mop_kg_acre = mop_kg_ha / HA_TO_ACRE
        
        total_dap_field = dap_kg_ha * field_area_ha
        total_urea_field = urea_kg_ha * field_area_ha
        total_mop_field = mop_kg_ha * field_area_ha
        
        dap_bags = round(total_dap_field / 50.0, 1)
        urea_bags = round(total_urea_field / 50.0, 1)
        mop_bags = round(total_mop_field / 50.0, 1)
        
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
            'crop': matched_crop,
            'estimation_method': 'Nutrient-balance estimate',
            'field_area_ha': field_area_ha,
            'field_area_acres': round(field_area_ha * HA_TO_ACRE, 2),
            'soil_test': {
                'N_mg_kg': soil_n_mg_kg,
                'P_mg_kg': soil_p_mg_kg,
                'K_mg_kg': soil_k_mg_kg,
                'pH': soil_ph,
                'OC_pct': organic_carbon
            },
            'soil_nutrient_pool_kg_ha': {
                'N_kg_ha': round(soil_n_kg_ha, 1),
                'P_kg_ha': round(soil_p_kg_ha, 1),
                'K_kg_ha': round(soil_k_kg_ha, 1)
            },
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
            'soil_amendment_advisory': amendment,
            'disclaimer': 'Advisory estimate only. Verify with local soil test laboratory or agricultural extension officer before application.'
        }

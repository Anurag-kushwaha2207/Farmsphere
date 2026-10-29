import pytest
from ml_models.agronomy.engines import IrrigationAdvisor, FertilizerEngine

# ==============================================================================
# 1. Irrigation Advisor Tests
# ==============================================================================

@pytest.fixture
def irrig_advisor():
    return IrrigationAdvisor()

def test_irrigation_hand_calculated_values(irrig_advisor):
    # Wheat at vegetative stage on Alluvial soil
    res = irrig_advisor.calculate_irrigation(
        crop_name='Wheat',
        growth_stage='Tillering & Vegetative',
        current_soil_moisture_pct=20.0,
        soil_type='Alluvial',
        temp_c=22.0,
        recent_rainfall_mm=0.0,
        irrigation_method='Drip Irrigation'
    )
    
    # Hand-calculated expectations:
    # Field capacity of Alluvial = 32.0%, moisture deficit = 32 - 20 = 12.0%
    # Root depth = 0.6 m -> Water deficit = 12% * 0.6 * 1000 = 72.0 mm
    assert res['water_deficit_mm'] == 72.0
    assert res['net_irrigation_required_mm'] == 72.0
    
    # Drip efficiency = 0.90 -> Gross irrigation = 72.0 / 0.90 = 80.0 mm
    assert res['gross_irrigation_required_mm'] == 80.0
    
    # Delivery rate = 8.0 mm/hr -> Pump runtime = 80.0 / 8.0 = 10.0 hours
    assert res['recommended_pump_runtime_hours'] == 10.0
    assert res['irrigation_recommended'] is True
    assert res['reference_eto_method'] == 'Hargreaves-Samani ETo estimation'

def test_irrigation_adequate_moisture(irrig_advisor):
    res = irrig_advisor.calculate_irrigation(
        crop_name='Wheat',
        growth_stage='Tillering & Vegetative',
        current_soil_moisture_pct=30.0, # Near FC (32%)
        soil_type='Alluvial',
        temp_c=20.0
    )
    assert res['irrigation_recommended'] is False
    assert res['recommended_pump_runtime_hours'] == 0.0

def test_irrigation_unsupported_crop_raises(irrig_advisor):
    with pytest.raises(ValueError, match="Unsupported crop for irrigation"):
        irrig_advisor.calculate_irrigation(
            crop_name='Sugarcane',
            growth_stage='Vegetative',
            current_soil_moisture_pct=25.0
        )

# ==============================================================================
# 2. Fertilizer Engine Tests
# ==============================================================================

@pytest.fixture
def fert_engine():
    return FertilizerEngine()

def test_fertilizer_hand_calculated_values(fert_engine):
    # Wheat: Target N=120, P=60, K=40
    # Soil: N=50 mg/kg, P=20 mg/kg, K=30 mg/kg
    res = fert_engine.calculate_fertilizer_schedule(
        crop_name='Wheat',
        soil_n_mg_kg=50.0,
        soil_p_mg_kg=20.0,
        soil_k_mg_kg=30.0,
        field_area_ha=1.0
    )
    
    # Soil furrow slice: 1 mg/kg = 2.0 kg/ha
    # N pool = 100 kg/ha; P pool = 40 kg/ha; K pool = 60 kg/ha
    assert res['soil_nutrient_pool_kg_ha']['N_kg_ha'] == 100.0
    assert res['soil_nutrient_pool_kg_ha']['P_kg_ha'] == 40.0
    assert res['soil_nutrient_pool_kg_ha']['K_kg_ha'] == 60.0
    
    # Crop available fractions: N=30% (30 kg), P=20% (8 kg), K=35% (21 kg)
    # Net deficit: N = 120 - 30 = 90 kg; P = 60 - 8 = 52 kg; K = 40 - 21 = 19 kg
    assert res['nutrient_deficit_kg_ha']['N'] == 90.0
    assert res['nutrient_deficit_kg_ha']['P2O5'] == 52.0
    assert res['nutrient_deficit_kg_ha']['K2O'] == 19.0
    
    # DAP = 52 / 0.46 = 113.0 kg/ha
    # N from DAP = 113.04 * 0.18 = 20.35 kg/ha
    # Remaining N = 90 - 20.35 = 69.65 kg -> Urea = 69.65 / 0.46 = 151.4 kg/ha
    # MOP = 19 / 0.60 = 31.7 kg/ha
    assert abs(res['commercial_fertilizer_rate_per_ha']['dap_kg_ha'] - 113.0) < 0.2
    assert abs(res['commercial_fertilizer_rate_per_ha']['urea_kg_ha'] - 151.4) < 0.2
    assert abs(res['commercial_fertilizer_rate_per_ha']['mop_kg_ha'] - 31.7) < 0.2
    
    # Must contain advisory disclaimer
    assert 'Advisory estimate only' in res['disclaimer']
    assert res['estimation_method'] == 'Nutrient-balance estimate'

def test_fertilizer_unsupported_crop_raises(fert_engine):
    with pytest.raises(ValueError, match="Unsupported crop for fertilizer"):
        fert_engine.calculate_fertilizer_schedule(
            crop_name='Sugarcane',
            soil_n_mg_kg=50.0,
            soil_p_mg_kg=20.0,
            soil_k_mg_kg=30.0
        )

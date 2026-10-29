import pytest
from ml_models.agronomy.suitability import CropSuitabilityEngine

@pytest.fixture
def engine():
    return CropSuitabilityEngine()

def test_supported_crops_coverage(engine):
    supported = engine.get_supported_crops()
    assert len(supported) >= 7
    expected = ['Wheat', 'Rice (Paddy)', 'Maize', 'Cotton (Seed Cotton)', 'Chickpea', 'Mustard', 'Potato']
    for crop in expected:
        assert crop in supported

def test_ideal_wheat_soil_highly_suitable(engine):
    res = engine.evaluate_suitability(
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
    assert res['suitability_score'] >= 85.0
    assert res['suitability_class'] == 'Highly Suitable'
    assert res['is_suitable'] is True
    assert len(res['limiting_factors']) == 0
    assert len(res['optimal_factors']) > 0

def test_unsuitable_wheat_soil(engine):
    res = engine.evaluate_suitability(
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
    assert res['suitability_score'] < 50.0
    assert res['suitability_class'] == 'Unsuitable'
    assert res['is_suitable'] is False
    assert len(res['limiting_factors']) >= 3
    
    # Check that limiting factors contain structured details
    for factor in res['limiting_factors']:
        assert 'parameter' in factor
        assert 'actual_value' in factor
        assert 'ideal_range' in factor
        assert 'penalty_applied' in factor
        assert factor['penalty_applied'] > 0

def test_unsupported_crop_raises_value_error(engine):
    with pytest.raises(ValueError, match="Unsupported crop"):
        engine.evaluate_suitability(
            crop_name='Sugarcane',
            soil_ph=6.5,
            nitrogen_mg_kg=100.0,
            phosphorus_mg_kg=30.0,
            potassium_mg_kg=40.0
        )

def test_all_master_crops_can_be_evaluated(engine):
    for crop in engine.get_supported_crops():
        res = engine.evaluate_suitability(
            crop_name=crop,
            soil_ph=6.8,
            nitrogen_mg_kg=80.0,
            phosphorus_mg_kg=35.0,
            potassium_mg_kg=45.0,
            electrical_conductivity_ds_m=0.5,
            soil_type='Clay Loam',
            temperature_c=25.0,
            water_available_mm=600.0
        )
        assert 0.0 <= res['suitability_score'] <= 100.0
        assert res['suitability_class'] in ['Highly Suitable', 'Moderately Suitable', 'Marginally Suitable', 'Unsuitable']

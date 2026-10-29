import pytest
from ml_models.agronomy.yield_predictor import YieldPredictor

@pytest.fixture
def yield_predictor():
    return YieldPredictor()

def test_unsupported_crop_raises_error(yield_predictor):
    with pytest.raises(ValueError, match="Unsupported crop for yield prediction"):
        yield_predictor.predict_yield(
            crop_name='Sugarcane',
            soil_type='Alluvial',
            irrigation_source='Tube Well (Canal / Groundwater)',
            season='Rabi (Winter)',
            avg_temp_c=20.0,
            season_rainfall_mm=100.0,
            avg_humidity_pct=60.0
        )

def test_valid_crop_prediction_with_uncertainty(yield_predictor):
    res = yield_predictor.predict_yield(
        crop_name='Wheat',
        soil_type='Alluvial',
        irrigation_source='Tube Well (Canal / Groundwater)',
        season='Rabi (Winter)',
        avg_temp_c=18.5,
        season_rainfall_mm=75.0,
        avg_humidity_pct=60.0
    )
    
    assert res['crop'] == 'Wheat'
    assert 'predicted_yield_tonnes_per_ha' in res
    assert 'lower_bound_tonnes_per_ha' in res
    assert 'upper_bound_tonnes_per_ha' in res
    assert 'yield_range_tonnes_per_ha' in res
    
    # Check uncertainty bounds ordering
    assert res['lower_bound_tonnes_per_ha'] <= res['predicted_yield_tonnes_per_ha']
    assert res['predicted_yield_tonnes_per_ha'] <= res['upper_bound_tonnes_per_ha']
    assert '-' in res['yield_range_tonnes_per_ha']
    
    # Check quintals per acre
    assert res['predicted_yield_quintals_per_acre'] > 0
    assert abs(res['predicted_yield_quintals_per_acre'] - round(res['predicted_yield_tonnes_per_ha'] * 4.047, 1)) < 0.2

def test_potato_yield_higher_than_cereal(yield_predictor):
    potato = yield_predictor.predict_yield(
        crop_name='Potato',
        soil_type='Sandy Loam',
        irrigation_source='Tube Well (Canal / Groundwater)',
        season='Rabi (Winter)',
        avg_temp_c=18.0,
        season_rainfall_mm=80.0,
        avg_humidity_pct=65.0
    )
    wheat = yield_predictor.predict_yield(
        crop_name='Wheat',
        soil_type='Alluvial',
        irrigation_source='Tube Well (Canal / Groundwater)',
        season='Rabi (Winter)',
        avg_temp_c=18.0,
        season_rainfall_mm=80.0,
        avg_humidity_pct=65.0
    )
    assert potato['predicted_yield_tonnes_per_ha'] > wheat['predicted_yield_tonnes_per_ha'] * 3

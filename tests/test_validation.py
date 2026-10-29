import pytest
from ml_models.predict import FarmSpherePredictor, validate_field_inputs

@pytest.fixture
def predictor():
    return FarmSpherePredictor()

def test_validate_rejects_impossible_ph():
    is_valid, errors, _ = validate_field_inputs({'soil_ph': 2.0})
    assert is_valid is False
    assert any('soil_ph' in e for e in errors)

def test_validate_rejects_impossible_nitrogen():
    is_valid, errors, _ = validate_field_inputs({'nitrogen': 900.0})
    assert is_valid is False
    assert any('nitrogen' in e for e in errors)

def test_validate_rejects_negative_humidity():
    is_valid, errors, _ = validate_field_inputs({'humidity': -10.0})
    assert is_valid is False
    assert any('humidity' in e for e in errors)

def test_analyze_field_returns_error_status_on_invalid_input(predictor):
    invalid_data = {
        'soil_ph': 2.0,
        'nitrogen': 900.0,
        'phosphorus': 50.0,
        'potassium': 40.0,
        'temperature': 22.0,
        'humidity': 60.0,
        'rainfall': 500.0
    }
    report = predictor.analyze_field(invalid_data)
    assert report['status'] == 'error'
    assert report['error_type'] == 'ValidationError'
    assert len(report['errors']) >= 2

def test_out_of_distribution_warning(predictor):
    ood_data = {
        'soil_ph': 7.0,
        'nitrogen': 180.0, # Within valid agronomic range (0-500) but outside training distribution (0-150)
        'phosphorus': 40.0,
        'potassium': 40.0,
        'temperature': 25.0,
        'humidity': 65.0,
        'rainfall': 150.0
    }
    report = predictor.analyze_field(ood_data)
    assert report['status'] == 'success'
    assert len(report['out_of_distribution_warnings']) > 0

def test_unsupported_target_crop_no_silent_fallback(predictor):
    data = {
        'soil_ph': 7.0,
        'nitrogen': 80.0,
        'phosphorus': 40.0,
        'potassium': 40.0,
        'temperature': 22.0,
        'humidity': 60.0,
        'rainfall': 200.0,
        'target_crop': 'Sugarcane'
    }
    report = predictor.analyze_field(data)
    assert report['status'] == 'success'
    assert report['active_crop'] == 'Sugarcane'
    # Suitability, Yield, Irrigation, and Fertilizer must explicitly report unsupported, NOT fake numbers
    assert report['crop_suitability']['status'] == 'unsupported'
    assert report['crop_yield_prediction']['status'] == 'unsupported'
    assert report['irrigation_advisory']['status'] == 'unsupported'
    assert report['fertilizer_schedule']['status'] == 'unsupported'

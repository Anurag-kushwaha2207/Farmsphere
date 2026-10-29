import os
import pandas as pd
import numpy as np
import pytest
from ml_models.evaluation.split_utils import get_crop_rec_splits, get_yield_group_splits

def test_crop_rec_splits_strictly_disjoint():
    csv_path = 'datasets/raw/Crop_recommendation.csv'
    if not os.path.exists(csv_path):
        pytest.skip("Kaggle Crop_recommendation.csv not found")
        
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test, le, feature_cols = get_crop_rec_splits(df, test_size=0.20, random_state=42)
    
    # Verify index sets are completely disjoint
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(train_idx) + len(test_idx) == len(df)
    assert len(test_idx) == int(len(df) * 0.20)

def test_yield_group_splits_disjoint_fields():
    csv_path = 'datasets/processed/yield_prediction_clean.csv'
    if not os.path.exists(csv_path):
        pytest.skip("yield_prediction_clean.csv not found")
        
    df = pd.read_csv(csv_path)
    train_df, test_df = get_yield_group_splits(df, test_size=0.20, random_state=42, group_col='field_id')
    
    train_fields = set(train_df['field_id'].unique())
    test_fields = set(test_df['field_id'].unique())
    
    # Crucial anti-leakage test: No field appearing in the test set should be present in the training set
    assert len(train_fields.intersection(test_fields)) == 0
    assert len(train_df) + len(test_df) == len(df)

def test_evaluator_never_uses_training_rows_in_test_metrics():
    from ml_models.evaluation.model_evaluator import evaluate_all_models
    summary = evaluate_all_models()
    
    crop_rec = summary['models']['crop_recommendation']
    assert crop_rec['train_samples'] == 1760
    assert crop_rec['test_samples'] == 440
    assert crop_rec['train_samples'] + crop_rec['test_samples'] == crop_rec['total_samples']
    
    # Test accuracy is not equal to 100% train accuracy
    assert crop_rec['test_metrics_held_out']['accuracy'] <= crop_rec['train_metrics']['accuracy']
    
    yield_rec = summary['models']['yield_prediction']
    assert yield_rec['train_samples'] == 2160
    assert yield_rec['test_samples'] == 540
    assert yield_rec['train_samples'] + yield_rec['test_samples'] == yield_rec['total_samples']

"""
FarmSphere AI - Shared Dataset Split & Evaluation Utilities
Ensures 100% consistent train/test splits across training and evaluation scripts.
Provides Group-based splitting and naive agricultural baselines.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42

def get_crop_rec_splits(df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_STATE):
    """
    Stratified train/test split for crop recommendation.
    Returns (X_train, X_test, y_train, y_test, label_encoder, feature_cols)
    """
    feature_cols = [c for c in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'] if c in df.columns]
    target_col = 'label'
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_col])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], y_encoded, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test, le, feature_cols

def get_yield_group_splits(df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_STATE, group_col: str = 'field_id'):
    """
    Group-aware train/test split by field_id.
    Prevents data leakage when multiple historical harvests come from the same physical field.
    """
    if group_col in df.columns:
        groups = df[group_col]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    else:
        # Fallback to standard split if field_id column not present
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
    return train_df, test_df

def compute_naive_crop_mean_baseline(y_train: np.ndarray, y_test: np.ndarray, crop_train: np.ndarray, crop_test: np.ndarray):
    """
    Naive baseline: Predicts the training-set mean yield for each crop.
    Because crop identity alone explains ~90-95% of yield variance (e.g. Potato 25t vs Chickpea 1.5t),
    any regression model MUST be benchmarked against this crop-mean baseline.
    """
    crop_means = {}
    for c in np.unique(crop_train):
        crop_means[c] = np.mean(y_train[crop_train == c])
        
    global_mean = np.mean(y_train)
    y_baseline_pred = np.array([crop_means.get(c, global_mean) for c in crop_test])
    
    baseline_r2 = r2_score(y_test, y_baseline_pred)
    baseline_mae = mean_absolute_error(y_test, y_baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline_pred))
    
    return {
        'baseline_r2': round(float(baseline_r2), 4),
        'baseline_mae': round(float(baseline_mae), 3),
        'baseline_rmse': round(float(baseline_rmse), 3),
        'predictions': y_baseline_pred,
        'crop_means': {k: round(float(v), 2) for k, v in crop_means.items()}
    }

def compute_within_crop_r2(y_true: np.ndarray, y_pred: np.ndarray, crop_labels: np.ndarray):
    """
    Within-crop R2 measures the variance explained *within* each crop class,
    removing the dominant effect of cross-crop mean differences.
    Formula: 1 - sum((y - y_hat)^2) / sum((y - y_bar_crop)^2)
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    
    denominator = 0.0
    for c in np.unique(crop_labels):
        mask = (crop_labels == c)
        if np.sum(mask) > 1:
            y_c = y_true[mask]
            y_bar_c = np.mean(y_c)
            denominator += np.sum((y_c - y_bar_c) ** 2)
            
    if denominator <= 1e-9:
        return 0.0
        
    within_r2 = 1.0 - (numerator / denominator)
    return round(float(within_r2), 4)

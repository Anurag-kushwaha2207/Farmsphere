"""
FarmSphere AI - Crop Yield Prediction Model Training Pipeline
Algorithm: Random Forest Regressor with Tree-Ensemble Uncertainty Estimation
Features:
- Categorical: crop_name, soil_type, irrigation_source, season
- Numerical: avg_temp_c, season_rainfall_mm, avg_humidity_pct
- NOTE: field_area_ha is explicitly DROPPED as a predictor (land area does not alter per-hectare yield).

Validation & Evaluation:
- Group-aware split by field_id (GroupShuffleSplit) to prevent field-level data leakage.
- Naive crop-mean baseline benchmark (crop identity alone explains ~90-95% of cross-crop variance).
- Within-crop R2 reporting to honestly assess climate/soil variance explanation.
- Tree-spread uncertainty intervals (10th to 90th percentile).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ml_models.evaluation.split_utils import (
    get_yield_group_splits,
    compute_naive_crop_mean_baseline,
    compute_within_crop_r2,
    RANDOM_STATE
)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/processed/yield_prediction_clean.csv'))
SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def train_yield_predictor():
    print("=" * 70)
    print("   FARMSPHERE AI: TRAINING CROP YIELD PREDICTOR (GROUP-AWARE)")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing yield dataset at: {DATA_PATH}. Run dataset generator first.")

    df = pd.read_csv(DATA_PATH)
    supported_crops = sorted(list(df['crop_name'].unique()))
    print(f"Loaded {len(df)} harvest records across {len(supported_crops)} crops: {supported_crops}")
    
    # Explicitly drop field_area_ha as a predictive feature
    categorical_cols = ['crop_name', 'soil_type', 'irrigation_source', 'season']
    numeric_cols = ['avg_temp_c', 'season_rainfall_mm', 'avg_humidity_pct']
    feature_cols = categorical_cols + numeric_cols
    print(f"Predictive features ({len(feature_cols)}): {feature_cols}")
    print("Note: 'field_area_ha' is excluded to prevent spurious area-to-yield bias.")

    # Group-aware train/test split by field_id
    train_df, test_df = get_yield_group_splits(df, test_size=0.20, random_state=RANDOM_STATE, group_col='field_id')
    print(f"Split by field_id: Train={len(train_df)} rows ({train_df['field_id'].nunique()} fields) | "
          f"Test={len(test_df)} rows ({test_df['field_id'].nunique()} fields)")

    X_train = train_df[feature_cols]
    y_train = train_df['actual_yield_tonnes_per_ha'].values
    crop_train = train_df['crop_name'].values

    X_test = test_df[feature_cols]
    y_test = test_df['actual_yield_tonnes_per_ha'].values
    crop_test = test_df['crop_name'].values

    # Naive Crop-Mean Baseline
    baseline = compute_naive_crop_mean_baseline(y_train, y_test, crop_train, crop_test)
    print("\n--- Naive Crop-Mean Baseline Benchmark ---")
    print(f"  Baseline R2:   {baseline['baseline_r2']:.4f}")
    print(f"  Baseline MAE:  {baseline['baseline_mae']:.3f} tonnes/ha")
    print(f"  Baseline RMSE: {baseline['baseline_rmse']:.3f} tonnes/ha")

    # Column Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    # Candidate Pipelines
    candidates = {
        'Ridge Regression': Pipeline([
            ('prep', preprocessor),
            ('reg', Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ]),
        'Random Forest Regressor': Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=1))
        ]),
        'Gradient Boosting Regressor': Pipeline([
            ('prep', preprocessor),
            ('reg', GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=RANDOM_STATE))
        ])
    }

    print("\n--- 5-Fold Cross-Validation on Training Set ---")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    best_pipe = None
    best_name = None
    best_cv_r2 = -float('inf')

    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2')
        cv_results[name] = {
            'mean_r2': round(float(scores.mean()), 4),
            'std_r2': round(float(scores.std()), 4)
        }
        print(f"  {name:30s}: Mean R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_cv_r2:
            best_cv_r2 = scores.mean()
            best_name = name
            best_pipe = pipe

    # We select RandomForestRegressor to provide tree-ensemble uncertainty intervals
    selected_name = 'Random Forest Regressor'
    selected_pipe = candidates[selected_name]
    print(f"\n>> Selected Pipeline: {selected_name} (Enables 10th-90th percentile tree spread)")

    # Fit selected pipeline on training set
    selected_pipe.fit(X_train, y_train)

    # Evaluate on Held-Out Test Set
    y_test_pred = selected_pipe.predict(X_test)
    y_train_pred = selected_pipe.predict(X_train)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    within_r2 = compute_within_crop_r2(y_test, y_test_pred, crop_test)

    # Uncertainty spread on test set
    reg = selected_pipe.named_steps['reg']
    prep = selected_pipe.named_steps['prep']
    X_test_prep = prep.transform(X_test)
    tree_preds = np.array([tree.predict(X_test_prep) for tree in reg.estimators_])
    test_p10 = np.percentile(tree_preds, 10, axis=0)
    test_p90 = np.percentile(tree_preds, 90, axis=0)
    avg_interval_width = float(np.mean(test_p90 - test_p10))

    # Feature Importance Extraction
    cat_feature_names = prep.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = list(cat_feature_names) + numeric_cols
    importances = reg.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {'feature': all_feature_names[i], 'importance': round(float(importances[i]), 4)}
        for i in sorted_idx[:10]
    ]

    print("\n--- Held-Out Test Evaluation (Field Group Split) ---")
    print(f"  Train R2:                     {train_r2:.4f}")
    print(f"  Held-Out Test R2:             {test_r2:.4f} (Beats Baseline: {test_r2 > baseline['baseline_r2']})")
    print(f"  Held-Out Test MAE:            {test_mae:.3f} tonnes/ha")
    print(f"  Held-Out Test RMSE:           {test_rmse:.3f} tonnes/ha")
    print(f"  Within-Crop Test R2:          {within_r2:.4f}")
    print(f"  Avg 80% Uncertainty Width:    {avg_interval_width:.2f} tonnes/ha")

    print("\n--- Top 6 Feature Importances ---")
    for item in top_features[:6]:
        print(f"  {item['feature']:35s}: {item['importance']*100:.2f}%")

    # Attach model metadata to the pipeline object
    selected_pipe.supported_crops_ = supported_crops
    selected_pipe.feature_cols_ = feature_cols
    selected_pipe.categorical_cols_ = categorical_cols
    selected_pipe.numeric_cols_ = numeric_cols

    # Save Model Artifact
    save_path = os.path.join(SAVED_MODELS_DIR, 'yield_prediction_model.joblib')
    joblib.dump(selected_pipe, save_path)
    print(f"\nSaved model binary: {save_path}")

    # Save Metrics JSON
    metrics_payload = {
        'model_name': 'Crop Yield Prediction Regressor',
        'algorithm': 'RandomForestRegressor (n_estimators=100, max_depth=12)',
        'supported_crops': supported_crops,
        'features': {
            'categorical': categorical_cols,
            'numeric': numeric_cols,
            'dropped_features': ['field_area_ha']
        },
        'split_strategy': 'GroupShuffleSplit by field_id (80% train, 20% test)',
        'cross_validation': cv_results,
        'baseline_crop_mean': {k: v for k, v in baseline.items() if k != 'predictions'},
        'test_metrics_held_out': {
            'train_r2': round(float(train_r2), 4),
            'test_r2': round(float(test_r2), 4),
            'test_mae_tonnes_per_ha': round(float(test_mae), 3),
            'test_rmse_tonnes_per_ha': round(float(test_rmse), 3),
            'within_crop_r2': round(float(within_r2), 4),
            'beats_baseline': bool(test_r2 > baseline['baseline_r2']),
            'avg_uncertainty_interval_tonnes_per_ha': round(avg_interval_width, 2)
        },
        'top_feature_importances': top_features
    }

    metrics_path = os.path.join(SAVED_MODELS_DIR, 'yield_prediction_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved evaluation metrics: {metrics_path}")

    return selected_pipe, metrics_payload

if __name__ == '__main__':
    train_yield_predictor()

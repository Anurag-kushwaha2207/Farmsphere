"""
FarmSphere AI - Crop Recommendation Model Training Pipeline
Trained on the Real Kaggle "Crop Recommendation" Dataset (~2,200 samples, 22 crops).

INPUT UNITS & FEATURE DEFINITIONS:
  - N: Soil available Nitrogen ratio (measured from soil sample, NOT crop requirement kg/ha demand)
  - P: Soil available Phosphorus ratio (measured from soil sample)
  - K: Soil available Potassium ratio (measured from soil sample)
  - temperature: Ambient temperature in °C
  - humidity: Relative humidity in %
  - ph: Soil pH (0 to 14)
  - rainfall: Rainfall in mm

22 Crops Covered:
  apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute,
  kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange,
  papaya, pigeonpeas, pomegranate, rice, watermelon.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Add evaluation dir to path for shared split_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation')))
from split_utils import get_crop_rec_splits, RANDOM_STATE

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/processed/crop_recommendation_clean.csv'))
RAW_KAGGLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/raw/Crop_recommendation.csv'))
SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def train_crop_recommender():
    print("=" * 70)
    print("   FARMSPHERE AI: TRAINING REAL KAGGLE CROP RECOMMENDATION MODEL")
    print("=" * 70)
    
    # 1. Enforce existence of real Kaggle dataset; NO silent fallback
    if not os.path.exists(RAW_KAGGLE_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Mandatory dataset missing: {RAW_KAGGLE_PATH}\n"
            "Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n"
            "Save as datasets/raw/Crop_recommendation.csv. Silent fallback is prohibited."
        )

    # 2. Load Processed Dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} samples across {df['label'].nunique()} crop classes.")
    
    # 3. Stratified Train / Test Split using shared helper (80% Train, 20% Test)
    X_train, X_test, y_train, y_test, le, feature_cols = get_crop_rec_splits(
        df, test_size=0.20, random_state=RANDOM_STATE
    )
    classes = list(le.classes_)
    
    # 4. Feature Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Selection & Cross-Validation
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_model_name = None
    best_cv_score = 0.0
    best_clf = None
    cv_results = {}
    
    print("\n--- 5-Fold Stratified Cross-Validation on Training Set ---")
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        cv_results[name] = {'mean_accuracy': round(float(mean_score), 4), 'std': round(float(std_score), 4)}
        print(f"  {name:20s}: Mean Accuracy = {mean_score*100:.2f}% (+/- {std_score*100:.2f}%)")
        
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_model_name = name
            best_clf = model
            
    print(f"\n>> Selected Best Model: {best_model_name} (CV: {best_cv_score*100:.2f}%)")
    
    # 6. Fit Best Model on Full Training Set
    best_clf.fit(X_train_scaled, y_train)
    
    # 7. Evaluate on Unseen Held-Out Test Set
    y_pred = best_clf.predict(X_test_scaled)
    y_prob = best_clf.predict_proba(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_pred)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    
    # Top-3 Accuracy
    top3_correct = sum(
        1 for i in range(len(y_test)) 
        if y_test[i] in np.argsort(y_prob[i])[::-1][:3]
    )
    top3_acc = top3_correct / len(y_test)
    
    print("\n--- Held-Out Test Set Evaluation ---")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Top-3 Accuracy:        {top3_acc * 100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:     {weighted_f1 * 100:.2f}%")
    print(f"Weighted Precision:    {weighted_p * 100:.2f}%")
    print(f"Weighted Recall:       {weighted_r * 100:.2f}%")
    
    # 8. Feature Importances
    if hasattr(best_clf, 'feature_importances_'):
        importances = dict(zip(feature_cols, [round(float(v), 4) for v in best_clf.feature_importances_]))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print("\nFeature Importances:")
        for feat, imp in sorted_imp:
            print(f"  {feat:15s}: {imp * 100:.2f}%")
    else:
        importances = {}
        
    # 9. Save Artifacts
    model_artifact = {
        'model': best_clf,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': feature_cols,
        'classes': classes,
        'algorithm': best_model_name,
        'dataset_source': 'Kaggle Crop Recommendation Dataset (22 crops)'
    }
    
    model_save_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_model.joblib')
    joblib.dump(model_artifact, model_save_path)
    print(f"\n Model binary saved to: {model_save_path}")
    
    # Save Metrics JSON
    metrics_report = {
        'model_name': 'Crop Recommendation Classifier',
        'algorithm': best_model_name,
        'dataset_source': 'Real Kaggle Crop Recommendation Dataset (~2200 samples, 22 crops)',
        'input_units': {
            'N': 'Soil available Nitrogen ratio (measured from soil test)',
            'P': 'Soil available Phosphorus ratio (measured from soil test)',
            'K': 'Soil available Potassium ratio (measured from soil test)',
            'temperature': 'Celsius',
            'humidity': 'Percentage %',
            'ph': 'pH units (0-14)',
            'rainfall': 'mm'
        },
        'features': feature_cols,
        'num_classes': len(classes),
        'classes': classes,
        'cross_validation': cv_results,
        'test_metrics_held_out': {
            'accuracy': round(float(test_acc), 4),
            'top3_accuracy': round(float(top3_acc), 4),
            'macro_f1': round(float(macro_f1), 4),
            'weighted_precision': round(float(weighted_p), 4),
            'weighted_recall': round(float(weighted_r), 4),
            'weighted_f1': round(float(weighted_f1), 4)
        },
        'feature_importances': importances,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_report, f, indent=2)
    print(f" Evaluation metrics saved to: {metrics_path}")
    
    # 10. Sample Test Prediction
    print("\n--- Sample Test Prediction (Rice conditions) ---")
    # Sample input: N=90, P=42, K=43, temp=21°C, hum=82%, pH=6.5, rain=203mm (Typical Rice soil in Kaggle dataset)
    sample_df = pd.DataFrame([[90, 42, 43, 20.88, 82.0, 6.5, 202.94]], columns=feature_cols)
    sample_scaled = scaler.transform(sample_df)
    pred_idx = best_clf.predict(sample_scaled)[0]
    pred_crop = le.inverse_transform([pred_idx])[0]
    probs = best_clf.predict_proba(sample_scaled)[0]
    
    top3_indices = np.argsort(probs)[::-1][:3]
    print(f"Input: N=90, P=42, K=43, Temp=20.9°C, Humidity=82%, pH=6.5, Rain=203mm")
    print(f"Top Recommendation: {pred_crop} (Confidence: {probs[pred_idx]*100:.1f}%)")
    print("Top 3 Recommendations:")
    for idx in top3_indices:
        print(f"  - {classes[idx]}: {probs[idx]*100:.1f}%")
        
    return metrics_report

if __name__ == '__main__':
    train_crop_recommender()

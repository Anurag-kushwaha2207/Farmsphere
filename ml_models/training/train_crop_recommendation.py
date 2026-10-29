"""
FarmSphere AI - Crop Recommendation Model Training Pipeline
Algorithm: Random Forest Classifier vs Gradient Boosting vs Decision Tree
Inputs: N (mg/kg), P (mg/kg), K (mg/kg), temperature (°C), humidity (%), pH, rainfall (mm)
Output: Best Crop Recommendation + Top 3 Alternatives + Class Probabilities
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/processed/crop_recommendation_clean.csv'))
SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def train_crop_recommender():
    print("=" * 60)
    print("   FARMSPHERE AI: TRAINING CROP RECOMMENDATION MODEL")
    print("=" * 60)
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} samples across {df['label'].nunique()} crop classes.")
    
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols]
    y = df['label']
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    
    # 3. Train / Test Split (80% Train, 20% Test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    # 4. Feature Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Selection & Cross-Validation
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = None
    best_cv_score = 0.0
    best_clf = None
    cv_results = {}
    
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    for name, model in models.items():
        # Tree models can use raw or scaled; scaled works uniformly
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
    
    # 7. Evaluate on Unseen Test Set
    y_pred = best_clf.predict(X_test_scaled)
    y_prob = best_clf.predict_proba(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print("\n--- Test Set Evaluation ---")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Weighted Precision:    {precision * 100:.2f}%")
    print(f"Weighted Recall:       {recall * 100:.2f}%")
    print(f"Weighted F1-Score:     {f1 * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
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
        'algorithm': best_model_name
    }
    
    model_save_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_model.joblib')
    joblib.dump(model_artifact, model_save_path)
    print(f"\n Model binary saved to: {model_save_path}")
    
    # Save Metrics JSON
    metrics_report = {
        'model_name': 'Crop Recommendation Classifier',
        'algorithm': best_model_name,
        'features': feature_cols,
        'classes': classes,
        'cross_validation': cv_results,
        'test_metrics': {
            'accuracy': round(float(test_acc), 4),
            'precision_weighted': round(float(precision), 4),
            'recall_weighted': round(float(recall), 4),
            'f1_weighted': round(float(f1), 4)
        },
        'feature_importances': importances,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'crop_recommendation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_report, f, indent=2)
    print(f" Evaluation metrics saved to: {metrics_path}")
    
    # 10. Sample Test Prediction
    print("\n--- Sample Test Prediction ---")
    sample_input = np.array([[120, 60, 40, 20.0, 55.0, 6.8, 500.0]]) # Ideal Wheat parameters
    sample_scaled = scaler.transform(sample_input)
    pred_idx = best_clf.predict(sample_scaled)[0]
    pred_crop = le.inverse_transform([pred_idx])[0]
    probs = best_clf.predict_proba(sample_scaled)[0]
    
    top3_indices = np.argsort(probs)[::-1][:3]
    print(f"Input: N=120, P=60, K=40, Temp=20°C, Humidity=55%, pH=6.8, Rain=500mm")
    print(f"Top Recommendation: {pred_crop} (Confidence: {probs[pred_idx]*100:.1f}%)")
    print("Alternative Suitable Crops:")
    for idx in top3_indices[1:]:
        print(f"  - {classes[idx]}: {probs[idx]*100:.1f}%")
        
    return metrics_report

if __name__ == '__main__':
    train_crop_recommender()

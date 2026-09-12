from pathlib import Path
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
  roc_auc_score,
  precision_score,
  recall_score,
  f1_score,
  confusion_matrix,
  classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))
if str(BASE_DIR) not in sys.path:
  sys.path.insert(0, str(BASE_DIR))

try: 
  from data_pipeline import load_data, clean_data
  from feature_engineering import build_feature_matrix
except ImportError:
  from src.data_pipeline import load_data, clean_data
  from src.feature_engineering import build_feature_matrix

def train_and_evaluate_model(
    models_dir: str | Path = BASE_DIR / 'models',
    data_dir: str | Path = BASE_DIR / 'data/processed',
):
  models_dir = Path(models_dir)
  data_dir = Path(data_dir)
  os.makedirs(models_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  print("----- Step 1: Load and clean data -----")
  patients, vitals, labs, history, outcomes = clean_data(*load_data())

  print("----- Step 2: Feature engineering -----")
  df_features = build_feature_matrix(patients, vitals, labs, outcomes)
  processed_csv_path = data_dir / "sepsis_features.csv"
  df_features.to_csv(processed_csv_path, index=False)
  print(f"saved to {processed_csv_path} and has dataframe shape of {df_features.shape}")

  X = df_features.drop(columns=['patient_id', 'sepsis_event'])
  y = df_features['sepsis_event']
  feature_names = list(X.columns)

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
  )

  # 1 Baseline Model: Logistic Regression
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  baseline_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
  baseline_lr.fit(X_train_scaled, y_train)

  lr_probs = baseline_lr.predict_proba(X_test_scaled)[:, 1]
  lr_preds = (lr_probs > 0.5).astype(int)

  lr_roc = roc_auc_score(y_test, lr_probs)
  lr_precision = precision_score(y_test, lr_preds, zero_division=0)
  lr_recall = recall_score(y_test, lr_preds, zero_division=0)
  lr_f1 = f1_score(y_test, lr_preds, zero_division=0)

  print("\n----- Baseline Logistic Regression Model Evaluation -----") 
  print(f"ROC AUC: {lr_roc:.4f}")
  print(f"Precision: {lr_precision:.4f}")
  print(f"Recall: {lr_recall:.4f}")
  print(f"F1 Score: {lr_f1:.4f}")

  # 2 Primary Model : HistGradientBoostingClassifier
  primary_hgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.08,
    max_depth=5,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
  )

  primary_hgb.fit(X_train, y_train)

  hgb_probs = primary_hgb.predict_proba(X_test)[:, 1]
  hgb_preds = (hgb_probs > 0.5).astype(int)

  hgb_roc = roc_auc_score(y_test, hgb_probs)
  hgb_precision = precision_score(y_test, hgb_preds, zero_division=0)
  hgb_recall = recall_score(y_test, hgb_preds, zero_division=0)
  hgb_f1 = f1_score(y_test, hgb_preds, zero_division=0)

  print("\n----- Primary HistGradientBoostingClassifier Model Evaluation -----")
  print(f"ROC AUC: {hgb_roc:.4f}")
  print(f"Precision: {hgb_precision:.4f}")
  print(f"Recall: {hgb_recall:.4f}")
  print(f"F1 Score: {hgb_f1:.4f}")

  # Save artifacts

  model_playload = {
    'model': primary_hgb,
    'feature_names': feature_names,
    'medians': X.median().to_dict(),
    'baseline_model': baseline_lr,
    'scaler': scaler
  }
  model_path = os.path.join(models_dir, 'sepsis_model.joblib')
  joblib.dump(model_playload, model_path)

  metadata = {
    'model_type': 'HistGradientBoostingClassifier',
    'features': feature_names,
    'primary_metrics': {
      'roc-auc': round(hgb_roc, 4),
      'precision': round(hgb_precision, 4),
      'recall': round(hgb_recall, 4),
      'f1-score': round(hgb_f1, 4)
    },
    'baseline_metrics': {
      'model_type': 'LogisticRegression',
      'roc-auc': round(lr_roc, 4),
      'precision': round(lr_precision, 4),
      'recall': round(lr_recall, 4),
      'f1-score': round(lr_f1, 4)
    },
    'training_samples': X_train.shape[0],
    'testing_samples': X_test.shape[0]
  }

  metadata_path = os.path.join(models_dir, 'model_metadata.json')
  with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

  return metadata

if __name__ == "__main__":
  train_and_evaluate_model()
from pathlib import Path
import json
import os
import sys

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
  roc_auc_score,
  precision_score,
  recall_score,
  f1_score,
  confusion_matrix,
  classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'

if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from data_pipeline import load_data, clean_data
from feature_engineering_2 import build_feature_matrix

def evaluate_model(y_true, probabilities, threshold=0.5):
  """ Calculate classification metrics at the selected threshold."""
  predictions = (probabilities >= threshold).astype(int)

  return {
    "roc_auc": roc_auc_score(y_true, probabilities),
    "precision": precision_score(y_true, predictions, zero_division=0),
    "recall": recall_score(y_true, predictions, zero_division=0),
    "f1": f1_score(y_true, predictions, zero_division=0),
  }

def print_metrics(model_name, metrics):
  """Print model metrics in a consistent format."""
  print(f"----- {model_name} Evaluation Metrics -----")
  print(f"ROC AUC: {metrics['roc_auc']:.4f}")
  print(f"Precision: {metrics['precision']:.4f}")
  print(f"Recall: {metrics['recall']:.4f}")
  print(f"F1 Score: {metrics['f1']:.4f}")

def train_and_evaluate_model(
    models_dir: Path = BASE_DIR / 'models',
    data_dir: Path = BASE_DIR / 'data/processed',
):
  """Train Baseline and Primary Sepsis-risk models."""
  models_dir = Path(models_dir)
  data_dir = Path(data_dir)

  os.makedirs(models_dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)

  print("----- Step 1: Load and clean data -----")
  patients, vitals, labs, history, outcomes = clean_data(*load_data())

  print("----- Step 2: Feature engineering -----")
  features = build_feature_matrix(
    patients,
    vitals,
    labs,
    history,
    outcomes,
  )

  feature_csv_path = data_dir / "sepsis_features_2.csv"
  features.to_csv(feature_csv_path, index=False)

  print(f"Saved features to {feature_csv_path} with shape {features.shape}")

  X = features.drop(columns=['patient_id', 'sepsis_event'])
  y = features['sepsis_event']
  feature_names = X.columns.tolist()

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
  )

  print("----- Step 3: Train and evaluate models -----")
  imputer = SimpleImputer(strategy='median')

  X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=feature_names,
    index=X_train.index
  )

  X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_names,
    index=X_test.index
  )

  print("----- Step 4: Baseline Logistic Regression -----")
  scaler = StandardScaler()

  X_train_scaled = scaler.fit_transform(X_train_imputed)
  X_test_scaled = scaler.transform(X_test_imputed)

  baseline_lr_2 = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
  )

  baseline_lr_2.fit(X_train_scaled, y_train)

  lr_probabilities = baseline_lr_2.predict_proba(X_test_scaled)[:, 1]
  lr_metrics = evaluate_model(y_test, lr_probabilities)
  print_metrics("Baseline Logistic Regression", lr_metrics)

  print("----- Step 5: Primary HistGradientBoostingClassifier -----")
  primary_hbg_2 = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
  )

  primary_hbg_2.fit(X_train_imputed, y_train)

  hbg_probabilities = primary_hbg_2.predict_proba(X_test_imputed)[:, 1]
  hbg_metrics = evaluate_model(y_test, hbg_probabilities)
  print_metrics("Primary HistGradientBoostingClassifier", hbg_metrics)

  print("----- Step 6: Save models and artifacts -----")
  model_playload = {
    "model": primary_hbg_2,
    "baseline_model": baseline_lr_2,
    "imputer": imputer,
    "scaler": scaler,
    "feature_names": feature_names,
    "prediction_threshold": 0.5,
  }

  model_path = models_dir / "sepsis_model_2.joblib"
  joblib.dump(model_playload, model_path)

  metadata = {
    "model_type": "HistGradientBoostingClassifier",
    "baseline_model_type": "LogisticRegression",
    "feature_engineering_source": "feature_engineering_2.py",
    "feature_count": len(feature_names),
    "features": feature_names,
    "prediction_threshold": 0.5,
    "primary_metrics": {
      name: round(value, 4) 
      for name, value in hbg_metrics.items()
    },
    "baseline_metrics": {
      name: round(value, 4) 
      for name, value in lr_metrics.items()
    },
    "training_samples": int(X_train.shape[0]),
    "testing_samples": int(X_test.shape[0]),
    "sepsis_prevalence_train": round(float(y_train.mean()), 4),
    "sepsis_prevalence_test": round(float(y_test.mean()), 4),
  }

  metadata_path = models_dir / "sepsis_model_metadata_2.json"
  
  with open(metadata_path, 'w', encoding='utf-8') as file:
    json.dump(metadata, file, indent=2)

  print(f"Saved model to {model_path}")
  print(f"Saved metadata to {metadata_path}")

  return metadata

if __name__ == "__main__":
  train_and_evaluate_model()
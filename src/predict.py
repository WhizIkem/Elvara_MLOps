from pathlib import Path
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "sepsis_model.joblib"

class SepsisPredictor:
    def __init__(self, model_path: Union[str, Path, None] = None):
        if model_path is None:
            resolved_path = DEFAULT_MODEL_PATH
        else:
            resolved_path = Path(model_path)
            if not resolved_path.exists():
                resolved_path = BASE_DIR / model_path

        if not resolved_path.exists():
            raise FileNotFoundError(f"Model file not found at {resolved_path}. Train the model first.")
        
        payload = joblib.load(resolved_path)
        self.model = payload["model"]
        self.feature_names = payload["feature_names"]
        self.medians = payload["medians"]

    def predict_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts patient data dictionary containing:
        - age: int
        - gender: str ("Male", "Female")
        - comorbidity_count: int
        - vitals: List[dict] (timestamp, heart_rate, temperature, oxygen_saturation, respiratory_rate, blood_pressure)
        - labs: List[dict] (timestamp, white_cell_count, crp, lactate, creatinine, platelet_count)
        """
        # Build DataFrame row
        row = {}

        # 1. Static features
        row["age"] = float(patient_data.get("age", 60))
        gender = str(patient_data.get("gender", "Male")).strip().lower()
        row["gender_Male"] = 1 if gender == "male" else 0
        row["comorbidity_count"] = float(patient_data.get("comorbidity_count", 0))

        # 2. Vital sign features
        vitals = patient_data.get("vitals", [])
        if len(vitals) > 0:
            df_vitals = pd.DataFrame(vitals)
            for col in ['heart_rate', 'temperature', 'oxygen_saturation', 'respiratory_rate', 'blood_pressure']:
                if col in df_vitals:
                    vals = df_vitals[col].dropna()
                    if len(vals) > 0:
                        row[f"{col}_mean"] = float(vals.mean())
                        row[f"{col}_min"] = float(vals.min())
                        row[f"{col}_max"] = float(vals.max())
                        row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
                        row[f"{col}_last"] = float(vals.iloc[-1])
                        row[f"{col}_rate_per_hr"] = 0.0
                    else:
                        row[f"{col}_mean"] = self.medians.get(f"{col}_mean", np.nan)
                        row[f"{col}_min"] = self.medians.get(f"{col}_min", np.nan)
                        row[f"{col}_max"] = self.medians.get(f"{col}_max", np.nan)
                        row[f"{col}_std"] = 0.0
                        row[f"{col}_last"] = self.medians.get(f"{col}_last", np.nan)
                        row[f"{col}_rate_per_hr"] = 0.0
                else:
                    for stat in ['mean', 'min', 'max', 'std', 'last', 'rate_per_hr']:
                        row[f"{col}_{stat}"] = self.medians.get(f"{col}_{stat}", 0.0 if stat in ['std', 'rate_per_hr'] else np.nan)
        else:
            for col in ['heart_rate', 'temperature', 'oxygen_saturation', 'respiratory_rate', 'blood_pressure']:
                for stat in ['mean', 'min', 'max', 'std', 'last', 'rate_per_hr']:
                    row[f"{col}_{stat}"] = self.medians.get(f"{col}_{stat}", 0.0 if stat in ['std', 'rate_per_hr'] else np.nan)

        # 3. Lab features
        labs = patient_data.get("labs", [])
        if len(labs) > 0:
            df_labs = pd.DataFrame(labs)
            for col in ['white_cell_count', 'crp', 'lactate', 'creatinine', 'platelet_count']:
                if col in df_labs:
                    vals = df_labs[col].dropna()
                    if len(vals) > 0:
                        row[f"{col}_mean"] = float(vals.mean())
                        row[f"{col}_last"] = float(vals.iloc[-1])
                    else:
                        row[f"{col}_mean"] = self.medians.get(f"{col}_mean", np.nan)
                        row[f"{col}_last"] = self.medians.get(f"{col}_last", np.nan)
                else:
                    row[f"{col}_mean"] = self.medians.get(f"{col}_mean", np.nan)
                    row[f"{col}_last"] = self.medians.get(f"{col}_last", np.nan)
        else:
            for col in ['white_cell_count', 'crp', 'lactate', 'creatinine', 'platelet_count']:
                row[f"{col}_mean"] = self.medians.get(f"{col}_mean", np.nan)
                row[f"{col}_last"] = self.medians.get(f"{col}_last", np.nan)

        # Assemble single row DataFrame matching training columns
        feat_df = pd.DataFrame([row])
        for col in self.feature_names:
            if col not in feat_df.columns or pd.isna(feat_df[col].iloc[0]):
                feat_df[col] = self.medians.get(col, 0.0)

        feat_df = feat_df[self.feature_names]

        # Model Predict
        risk_score = float(self.model.predict_proba(feat_df)[0, 1])

        # Categorize Risk
        if risk_score >= 0.70:
            risk_category = "High"
        elif risk_score >= 0.35:
            risk_category = "Moderate"
        else:
            risk_category = "Low"

        # Risk drivers identification
        risk_drivers = []
        if row.get("heart_rate_last", 80) > 90:
            risk_drivers.append(f"Elevated Heart Rate ({row.get('heart_rate_last'):.1f} bpm)")
        if row.get("temperature_last", 37) > 38.0:
            risk_drivers.append(f"Fever / High Temp ({row.get('temperature_last'):.1f} °C)")
        elif row.get("temperature_last", 37) < 36.0:
            risk_drivers.append(f"Hypothermia / Low Temp ({row.get('temperature_last'):.1f} °C)")
        if row.get("oxygen_saturation_last", 98) < 95.0:
            risk_drivers.append(f"Low Oxygen Saturation ({row.get('oxygen_saturation_last'):.1f}%)")
        if row.get("respiratory_rate_last", 16) > 20:
            risk_drivers.append(f"Tachypnea / High Resp Rate ({row.get('respiratory_rate_last'):.1f} /min)")
        if row.get("lactate_last", 1.0) > 2.0:
            risk_drivers.append(f"Elevated Serum Lactate ({row.get('lactate_last'):.2f} mmol/L)")
        if row.get("crp_last", 5.0) > 20.0:
            risk_drivers.append(f"Elevated CRP Inflammatory Marker ({row.get('crp_last'):.1f} mg/L)")

        if not risk_drivers:
            risk_drivers.append("All vital signs and laboratory markers within expected reference ranges.")

        return {
            "patient_id": patient_data.get("patient_id", 0),
            "sepsis_risk_score": round(risk_score, 4),
            "risk_category": risk_category,
            "prediction_window": "6-12 hours",
            "key_risk_factors": risk_drivers,
            "features_extracted": feat_df.iloc[0].to_dict()
        }

if __name__ == "__main__":
    predictor = SepsisPredictor()
    sample_patient = {
        "patient_id": 999,
        "age": 68,
        "gender": "Male",
        "comorbidity_count": 2,
        "vitals": [
            {"heart_rate": 105.0, "temperature": 38.5, "oxygen_saturation": 93.0, "respiratory_rate": 24.0, "blood_pressure": 95.0}
        ],
        "labs": [
            {"white_cell_count": 14.2, "crp": 85.0, "lactate": 3.1, "creatinine": 1.8, "platelet_count": 140.0}
        ]
    }
    result = predictor.predict_patient(sample_patient)
    print("Sample Risk Prediction:\n", json.dumps(result, indent=2))
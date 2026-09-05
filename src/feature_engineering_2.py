from pathlib import Path

import numpy as np
import pandas as pd

VITAL_COLS = [
    'heart_rate',
    'temperature',
    'oxygen_saturation',
    'respiratory_rate',
    'blood_pressure'
]

LAB_COLS = [
    'white_cell_count',
    'crp',
    'lactate',
    'creatinine',
    'platelet_count'
]

VITAL_LOOKBACK_HOURS = 6
LAB_LOOKBACK_HOURS = 24

def load_data(data_dir: Path):
  """Load cleaned patient, time-series, history, and outcome datasets"""
  patients = pd.read_csv(data_dir / 'patients_cleaned.csv', parse_dates=['registration_date'])
  vitals = pd.read_csv(data_dir / 'vital_signs_cleaned.csv', parse_dates=['timestamp'])
  labs = pd.read_csv(data_dir / 'laboratory_results_cleaned.csv', parse_dates=['timestamp'])
  history = pd.read_csv(data_dir / 'clinical_history_clean.csv')
  outcomes = pd.read_csv(data_dir / 'sepsis_outcomes_cleaned.csv', parse_dates=['diagnosis_time'])
  return patients, vitals, labs, history, outcomes

def count_items(value) -> int:
  """Count comma-separated clinical-history or condition values."""
  if pd.isna(value) or str(value).strip().lower() in {
    "",
    "none",
    "none reported",
    "unknown",
  }:
    return 0

  return len([
    item.strip() 
    for item in str(value).split(",")
    if item.strip() 
  ])

def get_prediction_times(
    outcomes_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    seed: int = 7,
) -> pd.DataFrame:
  """Create one time-aligned prediction point per patient."""
  rng = np.random.default_rng(seed)
  outcomes = outcomes_df.copy()

  def calc_prediction_time(row):
    if row['sepsis_event']:
      if pd.isna(row['diagnosis_time']):
        raise ValueError(
          f"Sepsis event for patient {row['patient_id']} has no diagnosis_time."
        )
      return row['diagnosis_time'] - pd.Timedelta(hours=9)

    patient_vitals = vitals_df[
      vitals_df['patient_id'] == row['patient_id']
    ]

    if patient_vitals.empty:
      raise ValueError(
        f"Patient {row['patient_id']} has no vital signs records"
      )

    start = patient_vitals['timestamp'].min()
    end = patient_vitals['timestamp'].max()

    span_hours = max((end - start).total_seconds() / 3600, 1.0)

    offset_hours = rng.uniform(0.4, 0.9) * span_hours

    return start + pd.Timedelta(hours=offset_hours)

  outcomes['prediction_time'] = outcomes.apply(
    calc_prediction_time, 
    axis=1,
  )

  return outcomes

def extract_vital_features(
    vitals_df: pd.DataFrame,
    patient_id: int,
    prediction_time: pd.Timestamp,
) -> dict:
  """Create six-hour vital-sign summary, trend, and availibility features."""
  window = vitals_df[
    (vitals_df['patient_id'] == patient_id) &
    (vitals_df['timestamp'] <= prediction_time) &
    (vitals_df['timestamp'] >= prediction_time - pd.Timedelta(hours=VITAL_LOOKBACK_HOURS))
  ].sort_values(by='timestamp')

  features = {
    "vitals_observations_count_6h": len(window),
    "hours_since_last_vitals_6h": (
      (prediction_time - window['timestamp'].iloc[-1]).total_seconds() / 3600
      if not window.empty
      else np.nan
    ),
  }

  for column in VITAL_COLS:
    observed = window.dropna(subset=[column])
    values = observed[column]

    features[f"{column}_mean_6h"] = values.mean() 
    features[f"{column}_min_6h"] = values.min() 
    features[f"{column}_max_6h"] = values.max()
    features[f"{column}_last_6h"] = (
      values.iloc[-1] 
      if not observed.empty 
      else np.nan
    )
    if len(values) > 1:
      features[f"{column}_std_6h"] = values.std()

      elapsed_hours = (
        observed['timestamp'].iloc[-1] - observed['timestamp'].iloc[0]
      ).total_seconds() / 3600

      features[f"{column}_rate_per_hour_6h"] = (
        (values.iloc[-1] - values.iloc[0]) / elapsed_hours
        if elapsed_hours > 0
        else 0.0
      )
    elif len(values) == 1:
      features[f"{column}_std_6h"] = 0.0
      features[f"{column}_rate_per_hour_6h"] = 0.0
    else:
      features[f"{column}_std_6h"] = np.nan
      features[f"{column}_rate_per_hour_6h"] = np.nan

    features[f"{column}_missing_6h"] = int(
      window.empty or window[column].isna().all()
    ) 

  return features

def extract_lab_features(
    labs_df: pd.DataFrame,
    patient_id: int,
    prediction_time: pd.Timestamp,
) -> dict:
  """Create 24-hour lab summary and availability features."""
  window = labs_df[
    (labs_df['patient_id'] == patient_id) &
    (labs_df['timestamp'] <= prediction_time) &
    (labs_df['timestamp'] >= prediction_time - pd.Timedelta(hours=LAB_LOOKBACK_HOURS))
  ].sort_values(by='timestamp')

  features = {
    "labs_observations_count_24h": len(window),
    "hours_since_last_lab_24h": (
      (prediction_time - window['timestamp'].iloc[-1]).total_seconds() / 3600
      if not window.empty
      else np.nan
    ),
  }

  for column in LAB_COLS:
    observed = window.dropna(subset=[column])
    values = observed[column]

    features[f"{column}_mean_24h"] = values.mean() 
    features[f"{column}_min_24h"] = values.min()
    features[f"{column}_max_24h"] = values.max()
    features[f"{column}_last_24h"] = (
      values.iloc[-1] 
      if not observed.empty 
      else np.nan
    )

    if len(values) > 1:
      features[f"{column}_std_24h"] = values.std()

      elapsed_hours = (
        observed['timestamp'].iloc[-1] - observed['timestamp'].iloc[0]
      ).total_seconds() / 3600

      features[f"{column}_rate_per_hour_24h"] = (
        (values.iloc[-1] - values.iloc[0]) / elapsed_hours
        if elapsed_hours > 0
        else 0.0
      )
    elif len(values) == 1:
      features[f"{column}_std_24h"] = 0.0
      features[f"{column}_rate_per_hour_24h"] = 0.0
    else:
      features[f"{column}_std_24h"] = np.nan
      features[f"{column}_rate_per_hour_24h"] = np.nan

    features[f"{column}_missing_24h"] = int(
      window.empty or window[column].isna().all()
    )

  return features

def create_history_features(history_df: pd.DataFrame) -> pd.DataFrame:
  """
  Aggregate non-time-stamped history records.
  
  Keep these features only if history reflects information available before the prediction time; otherwise, remove them to prevent leakage/
  """
  return history_df.groupby('patient_id', as_index=False).agg(
    history_record_count=("patient_id", "size"),
    diagnosis_history_count=(
      "diagnosis_history",
      lambda values: sum(count_items(value) for value in values),
    ),
    infection_history_count=(
      "infection_history",
      lambda values: sum(count_items(value) for value in values),
    ),
    medication_history_count=(
      "medication_history",
      lambda values: sum(count_items(value) for value in values),
    ),
    treatment_history_count=(
      "treatment_history",
      lambda values: sum(count_items(value) for value in values),
    ), 
  )

def threshold_flag(series: pd.Series, condition) -> pd.Series:
  """Create a clinical threshold feature while preserving missing values"""
  return pd.Series(
    np.where(series.notna(), condition(series).astype(int), np.nan),
    index=series.index,
  )

def build_feature_matrix(
    patients_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    history_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> pd.DataFrame:
  """Build the improved. time-aligned sepsis feature matrix."""
  vitals_df = vitals_df.sort_values(
    ['patient_id', 'timestamp']
  ).reset_index(drop=True)

  labs_df = labs_df.sort_values(
    ['patient_id', 'timestamp']
  ).reset_index(drop=True)

  outcomes_with_time = get_prediction_times(
    outcomes_df, 
    vitals_df
  )

  vital_rows = [
    {
      "patient_id": patient_id,
      **extract_vital_features(
        vitals_df, 
        patient_id, 
        prediction_time,
      ),
    }
    for patient_id, prediction_time in zip(
      outcomes_with_time['patient_id'], 
      outcomes_with_time['prediction_time']
    )
  ]

  lab_rows = [
    {
      "patient_id": patient_id,
      **extract_lab_features(
        labs_df, 
        patient_id, 
        prediction_time,
      ),
    }
    for patient_id, prediction_time in zip(
      outcomes_with_time['patient_id'], 
      outcomes_with_time['prediction_time']
    )
  ] 

  vitals_features_df = pd.DataFrame(vital_rows)
  labs_features_df = pd.DataFrame(lab_rows)

  static_features = patients_df[
    ['patient_id', 'age', 'gender', 'medical_conditions']
  ].copy()

  static_features['comorbidity_count'] = static_features[
    'medical_conditions'
  ].apply(count_items)

  static_features = static_features.drop(
    columns=["medical_conditions"]
  )

  static_features = pd.get_dummies(
    static_features,
    columns=['gender'],
    drop_first=True,
  )

  static_features["age_65_or_over"] = (
    static_features["age"] >= 65
  ).astype(int)

  history_features = create_history_features(history_df)

  features = (
    static_features
    .merge(vitals_features_df, on='patient_id', how='left')
    .merge(labs_features_df, on='patient_id', how='left')
    .merge(history_features, on='patient_id', how='left')
    .merge(
      outcomes_with_time[['patient_id', 'sepsis_event']], 
      on='patient_id', 
      how='left'
    )
  )

  features["heart_rate_high_6h"] = threshold_flag(
    features["heart_rate_max_6h"], 
    lambda values: values >= 100,
  )

  features["fever_6h"] = threshold_flag(
    features["temperature_max_6h"],
    lambda values: values >= 38.0,
  )

  features["low_oxygen_saturation_6h"] = threshold_flag(
    features["oxygen_saturation_min_6h"],
    lambda values: values < 94.0,
  )

  features["high_respiratory_rate_6h"] = threshold_flag(
    features["respiratory_rate_max_6h"],
    lambda values: values >= 22,
  )

  features["low_blood_pressure_6h"] = threshold_flag(
    features["blood_pressure_min_6h"],
    lambda values: values <= 90,
  )

  features_columns = [
    column
    for column in features.columns
    if column not in ("patient_id", "sepsis_event")
  ]

  numeric_columns = features[features_columns].select_dtypes(
    include='number'
  ).columns

  for column in numeric_columns:
    if features[column].isna().any():
      features[f"{column}_missing"] = (
        features[column].isna().astype(int)
      )

  features["sepsis_event"] = features["sepsis_event"].astype(int)
  return features

def main():
  data_dir = Path('../data/processed')
  output_path = data_dir / 'sepsis_features_2.csv'

  patients, vitals, labs, history, outcomes = load_data(data_dir)

  features = build_feature_matrix(
    patients, 
    vitals, 
    labs, 
    history, 
    outcomes
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  features.to_csv(output_path, index=False)

  print(f"Created: {output_path}")
  print(f"Feature matrix shape: {features.shape}")
  print(f"Total missing values: {features.isna().sum().sum()}")

if __name__ == "__main__":
  main()

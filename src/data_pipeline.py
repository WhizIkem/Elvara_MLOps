import os
import pandas as pd
import numpy as np

def load_data(data_dir: str = 'data/raw'):
  if not os.path.exists(data_dir) and os.path.exists('patients.csv'):
    data_dir = '.'
    patients = pd.read_csv('../data/raw/patients.csv', parse_dates=['registration_date'])
    labs = pd.read_csv('../data/raw/labs.csv', parse_dates=['timestamp'])
    vitals = pd.read_csv('../data/raw/vitals.csv', parse_dates=['timestamp'])
    history = pd.read_csv('../data/raw/history.csv')
    outcomes = pd.read_csv('../data/raw/outcomes.csv', parse_dates=['diagnosis_time'])

    return patients, labs, vitals, history, outcomes

def clean_data(
    patients: pd.DataFrame, 
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    history: pd.DataFrame,
    outcomes: pd.DataFrame
):
  patients_clean = patients.copy()
  vitals_clean = vitals.copy()
  labs_clean = labs.copy()
  history_clean = history.copy()
  outcomes_clean = outcomes.copy()

  vitals_clean = vitals_clean.drop_duplicates(subset=['patient_id', 'timestamp']).reset_index(drop=True)
  labs_clean = labs_clean.drop_duplicates(subset=['patient_id', 'timestamp']).reset_index(drop=True)

  vital_ranges = {
    'heart_rate': (30, 220),
    'temperature': (32, 43),
    'oxygen_saturation': (50, 100),
    'respiratory_rate': (5, 60),
    'blood_pressure': (40, 220)
  }

  vitals_clean['heart_rate'] = vitals_clean['heart_rate'].clip(*vital_ranges['heart_rate'])
  vitals_clean['temperature'] = vitals_clean['temperature'].clip(*vital_ranges['temperature'])
  vitals_clean['oxygen_saturation'] = vitals_clean['oxygen_saturation'].clip(*vital_ranges['oxygen_saturation'])
  vitals_clean['respiratory_rate'] = vitals_clean['respiratory_rate'].clip(*vital_ranges['respiratory_rate'])
  vitals_clean['blood_pressure'] = vitals_clean['blood_pressure'].clip(*vital_ranges['blood_pressure'])

  labs_ranges = {
    'white_cell_count': (0.1, 50.0),
    'crp': (0.0, 500.0),
    'lactate': (1.0, 20.0),
    'creatinine': (0.1, 10.0),
    'platelet_count': (5.0, 700.0)
  }

  labs_clean['white_cell_count'] = labs_clean['white_cell_count'].clip(*labs_ranges['white_cell_count'])
  labs_clean['crp'] = labs_clean['crp'].clip(*labs_ranges['crp'])
  labs_clean['lactate'] = labs_clean['lactate'].clip(*labs_ranges['lactate'])
  labs_clean['creatinine'] = labs_clean['creatinine'].clip(*labs_ranges['creatinine'])
  labs_clean['platelet_count'] = labs_clean['platelet_count'].clip(*labs_ranges['platelet_count'])

  vitals_cols = ['heart_rate', 'temperature', 'oxygen_saturation', 'respiratory_rate', 'blood_pressure']
  labs_cols = ['white_cell_count', 'crp', 'lactate', 'creatinine', 'platelet_count']

  vitals_clean[vitals_cols] = vitals_clean.groupby('patient_id')[vitals_cols].transform(lambda s: s.fillna())
  vitals_clean[vitals_cols] = vitals_clean.groupby('patient_id')[vitals_cols].transform(lambda s: s.fillna(s.median()))

  labs_clean[labs_cols] = labs_clean.groupby('patient_id')[labs_cols].transform(lambda s: s.fillna())
  labs_clean[labs_cols] = labs_clean.groupby('patient_id')[labs_cols].transform(lambda s: s.fillna(s.median()))

  return patients_clean, vitals_clean, labs_clean, history_clean, outcomes_clean

if __name__ == "__main__":
  patients, labs, vitals, history, outcomes = load_data()
  p_c, v_c, l_c, h_c, o_c = clean_data(patients, vitals, labs, history, outcomes)
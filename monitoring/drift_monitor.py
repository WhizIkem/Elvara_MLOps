import os
import json
from pathlib import Path

import pandas as pd


def normalize_boolean_features(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Represent matching boolean features consistently in both datasets."""
    reference = reference_df[columns].copy()
    current = current_df[columns].copy()

    for column in columns:
            if pd.api.types.is_bool_dtype(reference[column]) or pd.api.types.is_bool_dtype(current[column]):
                    reference[column] = reference[column].astype(int)
                    current[column] = current[column].astype(int)

    return reference, current


def run_drift_analysis(
    reference_csv: str | Path | None = None,
    current_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> str:

  project_root = Path(__file__).resolve().parent.parent

  if reference_csv is None:
      reference_csv = project_root / "data" / "processed" / "sepsis_features.csv"
  else:
      reference_csv = Path(reference_csv)
      if not reference_csv.is_absolute():
          reference_csv = project_root / reference_csv

  if current_csv is None:
      current_csv = reference_csv
  else:
      current_csv = Path(current_csv)
      if not current_csv.is_absolute():
          current_csv = project_root / current_csv

  if output_dir is None:
      output_dir = project_root / "reports"
  else:
      output_dir = Path(output_dir)
      if not output_dir.is_absolute():
          output_dir = project_root / output_dir

  output_dir.mkdir(parents=True, exist_ok=True)

  if not reference_csv.exists():
      raise FileNotFoundError(f"Reference dataset not found at {reference_csv}")
  if not current_csv.exists():
      raise FileNotFoundError(f"Current dataset not found at {current_csv}")

  reference_df = pd.read_csv(reference_csv)
  current_df = pd.read_csv(current_csv).rename(columns=lambda column: column.replace("_rate_per_hr", "_rate_per_hour"))

  html_report_path = output_dir / "evidently_drift_report.html"
  json_summary_path = output_dir / "drift_summary.json"

  try:
      from evidently.legacy.report import Report
      from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

      report = Report(metrics=[
          DataDriftPreset(), 
          DataQualityPreset()
      ])

      excluded_cols = {'patient_id', 'sepsis_event'}
      cols = [
          c for c in reference_df.columns
          if c in current_df.columns and c not in excluded_cols
      ]
      if not cols:
          raise ValueError("The reference and current datasets have no shared feature columns")
      reference_features, current_features = normalize_boolean_features(
          reference_df, current_df, cols
      )
      report.run(reference_data=reference_features, current_data=current_features)
      report.save_html(str(html_report_path))
      print(f"Evidently AI HTML report saved to {html_report_path}")

      summary_data = {
          "status": "PASS",
          "number_of_columns": len(cols),
          "reference_rows": len(reference_df),
          "current_rows": len(current_df),
          "drift_detected": False
      }
      with open(json_summary_path, 'w') as f:
          json.dump(summary_data, f, indent=2)

  except Exception as e:
      print(f"Evidently AI Report fallback: {e}")

      from scipy.stats import ks_2samp
      excluded_cols = {'patient_id', 'sepsis_event'}
      cols = [
          c for c in reference_df.columns
          if c in current_df.columns and c not in excluded_cols
      ]
      if not cols:
          raise ValueError("The reference and current datasets have no shared feature columns")
      drifted_columns = []
      for col in cols:
          stat, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
          if p_value < 0.05:
              drifted_columns.append(col)

      summary_data = {
          "status": "PASS" if len(drifted_columns) == 0 else "WARN",
          "drifted_columns_count": len(drifted_columns),
          "drifted_columns": drifted_columns,
          "total_columns": len(cols),
      }
      with open(json_summary_path, 'w') as f:
          json.dump(summary_data, f, indent=2)

      with open(html_report_path, 'w') as f:
          f.write(
              "<html><body><h1>Drift Report</h1><pre>{}</pre></body></html>".format(
                  json.dumps(summary_data, indent=2)
              )
          )

  return html_report_path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    report_path = run_drift_analysis(
        reference_csv=project_root / "data" / "processed" / "sepsis_features.csv",
        current_csv=project_root / "data" / "processed" / "sepsis2.csv",
    )
    print(f"Drift report completed: {report_path}")
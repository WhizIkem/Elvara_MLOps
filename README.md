# Elvara: Sepsis Detection and Early-Warning System

Elvara is a machine-learning clinical decision-support system that estimates a patient's risk of sepsis from demographic information, vital signs, laboratory results, and comorbidity information. It produces a risk score, a Low/Moderate/High category, a 6-12 hour prediction window, and the clinical signals that contributed to the result.

The project was developed as an end-to-end workflow:

1. Load and clean the source clinical datasets.
2. Build time-aware patient features from observations before a prediction cutoff.
3. Train and evaluate a baseline Logistic Regression model and a primary HistGradientBoosting model.
4. Save the trained model, feature names, imputation medians, and evaluation metadata.
5. Serve predictions through a FastAPI application.
6. Expose operational metrics and generate data-drift reports.

## System layout

```text
Elvara/
├── app/                 FastAPI service, schemas, and Prometheus metrics
├── src/                 Data pipeline, feature engineering, training, prediction
├── data/raw/            Source clinical CSV files
├── data/processed/      Cleaned data and generated feature matrices
├── models/              Trained joblib models and JSON metadata
├── monitoring/          Drift analysis and Prometheus configuration
├── reports/             Drift reports and summaries
├── notebooks/           Exploration and feature-engineering notebooks
└── tests/               Unit and integration tests
```

## 1. Environment setup

The project uses Python 3.11 in its Docker image. A local Python 3.11 environment is recommended for reproducing the pipeline.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies are pandas and NumPy for data processing, scikit-learn and joblib for machine learning, FastAPI and Uvicorn for serving predictions, and `prometheus_client` for application metrics.

## 2. Data preparation

The raw input data is stored in `data/raw/`:

- `patients.csv`
- `vital_signs.csv`
- `laboratory_results.csv`
- `clinical_history.csv`
- `sepsis_outcomes.csv`

`src/data_pipeline.py` loads these files and cleans the observations. Cleaning includes removing duplicate patient observations, clipping implausible vital and laboratory values to clinical ranges, forward-filling values within each patient, and using patient-level medians for remaining missing vital or laboratory values.

Run the data-loading and cleaning stage with:

```bash
python src/data_pipeline.py
```

This validates the load and clean operations in memory. The training workflow calls these functions directly and uses their cleaned data in the next stage. The repository already contains cleaned CSV files in `data/processed/` for the feature-engineering and monitoring workflows.

## 3. Feature engineering

`src/feature_engineering.py` creates the feature matrix used by the primary model. For each patient, it selects observations available before a prediction time and calculates:

- Demographics: age, gender, and comorbidity count
- Vital-sign statistics over a 9-hour lookback: mean, minimum, maximum, standard deviation, latest value, and rate of change
- Laboratory statistics over a 24-hour lookback: mean and latest value
- The binary target: `sepsis_event`

For sepsis cases, the prediction cutoff is set nine hours before diagnosis. For non-sepsis cases, a reproducible cutoff is sampled from the patient's observation period. This prevents future observations from being used to construct the prediction features.

To build the feature matrix directly from the processed datasets, run the script from `src/` because its standalone entry point uses paths relative to that directory:

```bash
cd src
python feature_engineering.py
cd ..
```

The extended alternative, `src/feature_engineering_2.py`, adds richer six-hour vital-sign and 24-hour laboratory statistics, missingness indicators, observation counts, time-since-observation features, clinical-history counts, and abnormal-value flags. Run it from `src/` in the same way; it writes `data/processed/sepsis_features_2.csv`.

## 4. Train and evaluate the model

Run the primary training workflow from the project root:

```bash
python src/train.py
```

The workflow loads and cleans the raw data, builds `data/processed/sepsis_features.csv`, splits the data into stratified training and test sets, trains a Logistic Regression baseline, and trains the primary `HistGradientBoostingClassifier`. It saves:

- `models/sepsis_model.joblib`: the deployed model, feature names, medians, baseline model, and scaler
- `models/model_metadata.json`: model type, sample counts, and evaluation metrics

The recorded primary-model test metrics are:

| Metric | HistGradientBoosting | Logistic Regression baseline |
| --- | ---: | ---: |
| ROC AUC | 0.9559 | 0.9430 |
| Precision | 0.9769 | 0.8964 |
| Recall | 0.8817 | 0.8730 |
| F1 score | 0.9269 | 0.8846 |

These are offline test-set results, not a substitute for clinical validation or prospective evaluation. The model should support clinical review rather than replace it.

An alternative expanded training workflow is available in `src/train_2.py` and uses the richer feature-engineering implementation. Confirm its model metadata and input feature contract before using its artifact with the deployed API.

## 5. Launch the FastAPI prediction service

After `models/sepsis_model.joblib` exists, start the API locally:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`: confirms whether the model loaded successfully
- `POST /predict-risk`: returns a patient risk prediction
- `GET /docs`: opens the interactive Swagger documentation
- `GET /metrics`: exposes Prometheus metrics
- `POST /monitoring/drift-report`: generates a drift report
- `GET /monitoring/drift-report`: serves the generated HTML report

The API loads `models/sepsis_model.joblib` by default. Set `MODEL_PATH` to use another compatible model artifact:

```bash
MODEL_PATH=models/sepsis_model.joblib uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. Sample prediction

Send demographic, vital-sign, and laboratory observations to `/predict-risk`:

```bash
curl -X POST http://localhost:8000/predict-risk \
  -H 'Content-Type: application/json' \
  -d '{
    "patient_id": 101,
    "age": 68,
    "gender": "Male",
    "comorbidity_count": 2,
    "vitals": [{
      "heart_rate": 105,
      "temperature": 38.5,
      "oxygen_saturation": 93,
      "respiratory_rate": 24,
      "blood_pressure": 95
    }],
    "labs": [{
      "white_cell_count": 14.2,
      "crp": 85,
      "lactate": 3.1,
      "creatinine": 1.8,
      "platelet_count": 140
    }]
  }'
```

The response contains `sepsis_risk_score`, `risk_category`, `prediction_window`, and `key_risk_factors`. The service categorises scores as Low below 0.35, Moderate from 0.35 to below 0.70, and High at 0.70 or above. Risk factors include elevated heart rate, fever or hypothermia, low oxygen saturation, high respiratory rate, elevated lactate, and elevated CRP where applicable.

## 7. Launch monitoring with Docker Compose

The Compose stack starts the API, Prometheus, and Grafana:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The services are available at:

- API: <http://localhost:8080>
- API documentation: <http://localhost:8080/docs>
- Health check: <http://localhost:8080/health>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000>

Prometheus scrapes the API's `/metrics` endpoint every 15 seconds. Elvara publishes:

- `elvara_sepsis_prediction_total`: prediction count by risk category
- `elvara_sepsis_prediction_latency_seconds`: prediction latency histogram
- `elvara_sepsis_model_loaded`: whether the model loaded successfully

Generate a drift report through the API after the feature datasets are available:

```bash
curl -X POST http://localhost:8080/monitoring/drift-report -o reports/evidently_drift_report.html
```

The monitoring code compares reference and current feature data, excludes identifiers and the target label, and writes an HTML report plus `reports/drift_summary.json`. The standalone script compares `sepsis_features.csv` with `sepsis2.csv`.

## 8. Run tests

Run the available test suite from the project root:

```bash
pytest -q
```

The integration tests verify that the Prometheus metric objects exist and that the FastAPI application exposes the health and metrics routes. For a running service, also verify the deployment manually:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## Clinical features and interpretation

The model combines static patient context with trends and recent measurements. The principal clinical inputs are age, comorbidity burden, heart rate, temperature, oxygen saturation, respiratory rate, blood pressure, white-cell count, CRP, lactate, creatinine, and platelet count. Aggregates such as the latest value, mean, range, variability, and rate of change help represent both the current state and deterioration over time.

The output is an early-warning estimate for a 6-12 hour window. It is intended for monitoring and decision support. Thresholds, calibration, fairness, clinical safety, and prospective performance should be reviewed with qualified clinical and governance teams before operational use.

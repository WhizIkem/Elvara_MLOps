from prometheus_client import Counter, Gauge, Histogram

PREDICTION_REQUESTS_TOTAL = Counter(
  "elvara_sepsis_prediction_total",
  "Total number of sepsis prediction served",
  ["risk_category"]
)

PREDICTION_LATENCY_SECONDS = Histogram(
  "elvara_sepsis_prediction_latency_seconds",
  "time taken to execute feature engineering and model inference",
  buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

MODEL_LOADED_GAUGE = Gauge(
  "elvara_sepsis_model_loaded",
  "1 if model artifact is loaded successfully, 0 otherwise"
)
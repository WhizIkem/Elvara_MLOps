import time
import os
import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app import metrics as app_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from app.schemas import *
    from src.predict import SepsisPredictor
except ImportError:
    from schemas import *
    from src.predict import SepsisPredictor


app = FastAPI(
  title="Elvara Health | Early Sepsis warning and deterioration system Prediction",
  description="Machine Learning Operations (MLOps) & Clinical Decision Support System",
  version="1.0.0",
    openapi_tags=[
            {"name": "Health"},
            {"name": "Prediction"},
            {"name": "Monitoring"},
    ],
)

predictor = None
app_metrics.MODEL_LOADED_GAUGE.set(0)

@app.on_event("startup")
def load_model():
    global predictor
    try:
        model_path = os.getenv("MODEL_PATH", "models/sepsis_model.joblib")
        predictor = SepsisPredictor(model_path)
        app_metrics.MODEL_LOADED_GAUGE.set(1)
    except Exception as e:
        predictor = None
        app_metrics.MODEL_LOADED_GAUGE.set(0)
        print(f"Error loading model: {e}")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Elvara Sepsis CDSS API. See /docs for API documentation."}

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"], summary="Health Check")
def health_check():
    is_loaded = predictor is not None
    return HealthCheckResponse(
        status="healthy" if is_loaded else "unhealthy",
        service="Elvara Sepsis cdss",
        model_loaded=is_loaded,
        version="1.0.0"
    )

@app.post("/predict-risk", response_model=SepsisPredictionResponse, tags=["Prediction"], summary="Predict Risk")
def predict_risk(request: SepsisPredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Sepsis ML Model not loaded.")

    start_time = time.time()
    try:
        patient_dict = request.dict()
        result = predictor.predict_patient(patient_dict)

        # Update Prometheus metrics
        latency = time.time() - start_time
        app_metrics.PREDICTION_LATENCY_SECONDS.observe(latency)
        app_metrics.PREDICTION_REQUESTS_TOTAL.labels(risk_category=result["risk_category"]).inc()

        return SepsisPredictionResponse(
            patient_id=result["patient_id"],
            sepsis_risk_score=result["sepsis_risk_score"],
            risk_category=result["risk_category"],
            prediction_window=result["prediction_window"],
            key_risk_factors=result["key_risk_factors"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics", tags=["Monitoring"], summary="Metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/monitoring/drift-report", tags=["Monitoring"], summary="Generate Drift Report")
def generate_drift_report():
    try:
        from monitoring.drift_monitor import run_drift_analysis
        report_path = run_drift_analysis()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate evidently drift report: {str(e)}")

    return FileResponse(report_path, media_type="text/html")


@app.get("/monitoring/drift-report", include_in_schema=False)
def view_drift_report():
    report_path = BASE_DIR / "reports" / "evidently_drift_report.html"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Drift report not found. Generate it with POST /monitoring/drift-report.",
        )

    return FileResponse(report_path, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
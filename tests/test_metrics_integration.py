from app.main import app
from app import metrics
from simulate_traffic import generate_random_patient


def test_prometheus_metrics_are_available():
    assert metrics.PREDICTION_REQUESTS_TOTAL is not None
    assert metrics.PREDICTION_LATENCY_SECONDS is not None
    assert metrics.MODEL_LOADED_GAUGE is not None


def test_fastapi_app_has_metrics_routes():
    routes = {route.path for route in app.routes}
    assert "/metrics" in routes
    assert "/health" in routes


def test_generate_random_patient_can_create_moderate_risk_profile():
    patient = generate_random_patient(42, risk_tier="Moderate")

    assert patient["risk_tier"] == "Moderate"
    assert patient["age"] >= 55
    assert patient["comorbidity_count"] >= 1
    assert patient["vitals"][-1]["heart_rate"] >= 90
    assert patient["labs"][-1]["lactate"] >= 1.8

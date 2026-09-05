from app.main import app
from app import metrics


def test_prometheus_metrics_are_available():
    assert metrics.PREDICTION_REQUESTS_TOTAL is not None
    assert metrics.PREDICTION_LATENCY_SECONDS is not None
    assert metrics.MODEL_LOADED_GAUGE is not None


def test_fastapi_app_has_metrics_routes():
    routes = {route.path for route in app.routes}
    assert "/metrics" in routes
    assert "/health" in routes

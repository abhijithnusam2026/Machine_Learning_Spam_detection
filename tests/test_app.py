"""Tests for the FastAPI serving layer."""

import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health_check():
    """Verify the health endpoint is always responsive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "distilbert-scam-classifier"
    }

def test_detect_scam_model_missing():
    """
    Since the test environment typically won't have the heavy model weights 
    downloaded in the 'scam-classifier-model' directory, the app lifespan 
    should raise a RuntimeError during startup. TestClient catches this 
    during context initialization.
    """
    # If the model dir is missing, TestClient should fail to start because 
    # of the RuntimeError we added to the lifespan.
    with pytest.raises(RuntimeError, match="Model load failure"):
        with TestClient(app) as test_client:
            pass # App startup triggers the lifespan exception

"""
tests/test_api.py
Tests for FastAPI endpoints using TestClient.
The model is mocked so no real weights are needed during CI.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "madewithml"))


# ── Mock the model loading so tests work without trained artifacts ──────────────
@pytest.fixture(autouse=True)
def mock_model_loading():
    """Patch serve.py's module-level model load before importing the app."""
    import torch

    mock_model = MagicMock()
    mock_model.eval.return_value = None

    # Fake logits: 4 classes, first class wins
    fake_logits = torch.tensor([[2.0, 0.5, 0.3, 0.1]])
    mock_model.return_value = fake_logits

    mock_vocab     = MagicMock()
    mock_vocab.return_value = [1, 2, 3]
    mock_tokenizer = MagicMock(return_value=["test", "text"])
    mock_idx2label = {"0": "nlp", "1": "cv", "2": "mlops", "3": "graph"}

    with patch("serve._model",     mock_model), \
         patch("serve._vocab",     mock_vocab), \
         patch("serve._idx2label", mock_idx2label), \
         patch("serve._tokenizer", mock_tokenizer), \
         patch("serve._device",    "cpu"):
        yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve import app
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok(client):
    data = response = client.get("/health").json()
    assert data["status"] == "ok"


# ── /predict ──────────────────────────────────────────────────────────────────
def test_predict_returns_200(client):
    response = client.post("/predict", json={"text": "BERT for NLP classification"})
    assert response.status_code == 200


def test_predict_returns_label(client):
    response = client.post("/predict", json={"text": "BERT for NLP classification"})
    data = response.json()
    assert "label" in data
    assert isinstance(data["label"], str)


def test_predict_returns_probabilities(client):
    response = client.post("/predict", json={"text": "BERT for NLP classification"})
    data = response.json()
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)


def test_predict_rejects_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422   # FastAPI validation error

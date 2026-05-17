"""
serve.py
FastAPI app that serves the trained model.

Run locally:
    uvicorn madewithml.serve:app --reload --port 8000

Endpoints:
    GET  /health        → liveness check (used by Jenkins)
    POST /predict       → classify text
"""

import json
import pickle
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchtext.data.utils import get_tokenizer

from madewithml.config import logger
from madewithml.models import TextClassifier

app = FastAPI(title="MadeWithML Classifier", version="1.0")

# ── Load artifacts at startup ──────────────────────────────────────────────────
ARTIFACT_DIR = Path("/tmp/eval_artifacts")


def load_model():
    if not ARTIFACT_DIR.exists():
        raise RuntimeError(f"Artifact directory not found: {ARTIFACT_DIR}")

    with open(ARTIFACT_DIR / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    idx2label = json.load(open(ARTIFACT_DIR / "idx2label.json"))
    tokenizer = get_tokenizer("basic_english")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextClassifier(len(vocab), embed_dim=64, num_classes=len(idx2label))
    model.load_state_dict(torch.load(ARTIFACT_DIR / "best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    return model, vocab, idx2label, tokenizer, device


try:
    _model, _vocab, _idx2label, _tokenizer, _device = load_model()
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"Could not load model: {e}")
    _model = None


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe — Jenkins calls this after deploy."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokens  = torch.tensor(_vocab(_tokenizer(request.text)), dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        logits = _model(tokens.to(_device), offsets.to(_device))
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        pred   = logits.argmax(1).item()

    label = _idx2label[str(pred)]
    return PredictResponse(
        label=label,
        probabilities={_idx2label[str(i)]: float(p) for i, p in enumerate(probs)},
    )

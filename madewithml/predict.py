"""
predict.py
Run inference on a single text input.

Usage:
    python madewithml/predict.py --text "Attention mechanism for NLP"
"""

import json
import pickle
import torch
from torchtext.data.utils import get_tokenizer

from madewithml.config import logger
from madewithml.models import TextClassifier


def load_artifacts(artifact_dir: str = "/tmp/eval_artifacts"):
    """Load vocab, label map, and model weights."""
    with open(f"{artifact_dir}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    idx2label = json.load(open(f"{artifact_dir}/idx2label.json"))
    return vocab, idx2label


def predict(
    text: str = "",
    artifact_dir: str = "/tmp/eval_artifacts",
):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, idx2label = load_artifacts(artifact_dir)
    tokenizer = get_tokenizer("basic_english")

    num_classes = len(idx2label)
    model       = TextClassifier(len(vocab), embed_dim=64, num_classes=num_classes)
    model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    tokens  = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        logits = model(tokens.to(device), offsets.to(device))
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        pred   = logits.argmax(1).item()

    label = idx2label[str(pred)]
    logger.info(f"Prediction: {label}")
    for i, p in enumerate(probs):
        logger.info(f"  {idx2label[str(i)]}: {p:.4f}")

    return {"label": label, "probabilities": {idx2label[str(i)]: p for i, p in enumerate(probs)}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--artifact-dir", default="/tmp/eval_artifacts")
    args = parser.parse_args()
    predict(text=args.text, artifact_dir=args.artifact_dir)

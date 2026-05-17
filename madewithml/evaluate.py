"""
evaluate.py
Evaluate the trained model on the test split and write metrics.json.
The Jenkins Performance Gate reads metrics.json to decide pass/fail.

Usage:
    python madewithml/evaluate.py \
        --experiment-name "my_experiment" \
        --dataset-loc data/projects.csv
"""

import json
import pickle
import mlflow
import torch
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from madewithml.config import logger, MLFLOW_TRACKING_URI
from madewithml.data import load_data, preprocess, split_data
from madewithml.models import TextClassifier
from madewithml.train import TextDataset, collate_batch


def evaluate_model(
    experiment_name: str = "mlops_experiment",
    dataset_loc:     str = "data/dataset.csv",
    run_id_file:     str = "run_id.txt",
    batch_size:      int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load run artifacts ─────────────────────────────────────────────────────
    run_id = Path(run_id_file).read_text().strip()
    logger.info(f"Evaluating run_id: {run_id}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client      = mlflow.tracking.MlflowClient()
    artifact_dir = client.download_artifacts(run_id, ".", dst_path="/tmp/eval_artifacts")

    with open(f"{artifact_dir}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    idx2label = json.load(open(f"{artifact_dir}/idx2label.json"))
    label2idx = {v: int(k) for k, v in idx2label.items()}

    # ── Data ──────────────────────────────────────────────────────────────────
    df                  = load_data(dataset_loc)
    df                  = preprocess(df)
    _, _, test_df       = split_data(df)

    tokenizer = get_tokenizer("basic_english")
    test_ds   = TextDataset(
        test_df["text"].tolist(),
        [label2idx[t] for t in test_df["tag"]],
        vocab,
        tokenizer,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = len(label2idx)
    model       = TextClassifier(len(vocab), embed_dim=64, num_classes=num_classes)
    model.load_state_dict(torch.load(f"{artifact_dir}/best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, offsets, labels in test_loader:
            texts, offsets = texts.to(device), offsets.to(device)
            preds = model(texts, offsets).argmax(1).cpu().tolist()
            all_preds  += preds
            all_labels += labels.tolist()

    # ── Metrics ───────────────────────────────────────────────────────────────
    target_names   = [idx2label[str(i)] for i in range(num_classes)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    report = classification_report(all_labels, all_preds, target_names=target_names)

    logger.info(f"\n{report}")
    logger.info(f"Weighted F1: {f1:.4f}")

    # ── Write metrics.json (read by Jenkins Performance Gate) ─────────────────
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    logger.info("✅ metrics.json written")

    # ── Log back to MLflow ─────────────────────────────────────────────────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"test_precision": precision, "test_recall": recall, "test_f1": f1})
        mlflow.log_artifact("metrics.json")

    logger.info("✅ Evaluation complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="mlops_experiment")
    parser.add_argument("--dataset-loc", default="data/dataset.csv")
    parser.add_argument("--run-id-file", default="run_id.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate_model(
        experiment_name=args.experiment_name,
        dataset_loc=args.dataset_loc,
        run_id_file=args.run_id_file,
        batch_size=args.batch_size,
    )

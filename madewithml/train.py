"""
train.py
Training workload: loads data, trains model, logs to MLflow.

Usage:
    python madewithml/train.py \
        --experiment-name "my_experiment" \
        --dataset-loc data/projects.csv \
        --num-epochs 10
"""

import json
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from pathlib import Path

from madewithml.config import logger, MLFLOW_TRACKING_URI
from madewithml.data import load_data, preprocess, split_data
from madewithml.models import TextClassifier
from madewithml.utils import set_seeds

# ── Dataset ────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.vocab     = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens  = self.vocab(self.tokenizer(self.texts[idx]))
        label   = self.labels[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_batch(batch):
    texts, labels = zip(*batch)
    offsets = [0] + [len(t) for t in texts[:-1]]
    offsets = torch.tensor(offsets).cumsum(0)
    texts   = torch.cat(list(texts))
    labels  = torch.stack(list(labels))
    return texts, offsets, labels


# ── Training helpers ───────────────────────────────────────────────────────────
def train_step(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for texts, offsets, labels in loader:
        texts, offsets, labels = texts.to(device), offsets.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts, offsets)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def val_step(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for texts, offsets, labels in loader:
            texts, offsets, labels = texts.to(device), offsets.to(device), labels.to(device)
            outputs = model(texts, offsets)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


# ── Main training function ─────────────────────────────────────────────────────
def train_model(
    experiment_name: str = "mlops_experiment",
    dataset_loc:     str = "data/dataset.csv",
    num_epochs:      int = 10,
    embed_dim:       int = 64,
    lr:            float = 0.01,
    batch_size:      int = 32,
    seed:            int = 42,
):
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df                      = load_data(dataset_loc)
    df                      = preprocess(df)
    train_df, val_df, _     = split_data(df, seed=seed)

    tokenizer = get_tokenizer("basic_english")

    # Build vocabulary from training data only
    def yield_tokens(texts):
        for text in texts:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(train_df["text"]),
        specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    # Encode labels
    classes     = sorted(df["tag"].unique())
    label2idx   = {c: i for i, c in enumerate(classes)}
    idx2label   = {i: c for c, i in label2idx.items()}

    train_ds = TextDataset(train_df["text"].tolist(), [label2idx[t] for t in train_df["tag"]], vocab, tokenizer)
    val_ds   = TextDataset(val_df["text"].tolist(),   [label2idx[t] for t in val_df["tag"]],   vocab, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = TextClassifier(len(vocab), embed_dim, len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "num_epochs": num_epochs,
            "embed_dim":  embed_dim,
            "lr":         lr,
            "batch_size": batch_size,
            "seed":       seed,
        })

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            train_loss, train_acc = train_step(model, train_loader, optimizer, criterion, device)
            val_loss,   val_acc   = val_step(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1:02d}/{num_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

        # Log model artifact
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Save vocab and label map for inference
        import pickle
        with open("vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        json.dump(idx2label, open("idx2label.json", "w"))

        mlflow.log_artifact("vocab.pkl")
        mlflow.log_artifact("idx2label.json")
        mlflow.log_artifact("best_model.pt")

        # Save run_id for evaluate.py to pick up
        Path("run_id.txt").write_text(run.info.run_id)
        logger.info(f"MLflow run_id: {run.info.run_id}")

    logger.info("✅ Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="mlops_experiment")
    parser.add_argument("--dataset-loc", default="data/dataset.csv")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_model(
        experiment_name=args.experiment_name,
        dataset_loc=args.dataset_loc,
        num_epochs=args.num_epochs,
        embed_dim=args.embed_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

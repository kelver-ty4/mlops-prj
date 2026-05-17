"""
models.py
Model architecture definition.
"""

import torch.nn as nn


class TextClassifier(nn.Module):
    """
    Simple bag-of-embeddings classifier.
    Embeddings → mean pool → dropout → linear → logits.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(embed_dim, num_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        dropped  = self.dropout(embedded)
        return self.fc(dropped)

"""
tests/test_models.py
Unit tests for model architecture.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "madewithml"))

from models import TextClassifier


VOCAB_SIZE   = 1000
EMBED_DIM    = 32
NUM_CLASSES  = 4
BATCH_SIZE   = 8
SEQ_LEN      = 20


@pytest.fixture
def model():
    return TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)


@pytest.fixture
def dummy_batch():
    """Simulate a collated batch from the DataLoader."""
    texts   = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SEQ_LEN,))
    offsets = torch.arange(0, BATCH_SIZE * SEQ_LEN, SEQ_LEN)
    return texts, offsets


def test_model_output_shape(model, dummy_batch):
    texts, offsets = dummy_batch
    logits = model(texts, offsets)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_model_output_is_not_nan(model, dummy_batch):
    texts, offsets = dummy_batch
    logits = model(texts, offsets)
    assert not torch.isnan(logits).any()


def test_model_softmax_sums_to_one(model, dummy_batch):
    texts, offsets = dummy_batch
    logits = model(texts, offsets)
    probs  = torch.softmax(logits, dim=1)
    sums   = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)


def test_model_eval_mode_no_grad(model, dummy_batch):
    model.eval()
    texts, offsets = dummy_batch
    with torch.no_grad():
        logits = model(texts, offsets)
    assert logits is not None


def test_model_different_seeds_give_different_weights():
    torch.manual_seed(0)
    m1 = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)
    torch.manual_seed(99)
    m2 = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)
    w1 = m1.fc.weight.data
    w2 = m2.fc.weight.data
    assert not torch.equal(w1, w2)

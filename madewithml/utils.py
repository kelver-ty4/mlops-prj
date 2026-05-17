"""
utils.py
Shared utility functions used across train, evaluate, predict, etc.
"""

import os
import random
import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str) -> dict:
    """Load a JSON file as a Python dict."""
    import json
    with open(path) as f:
        return json.load(f)


def save_dict(d: dict, path: str, cls=None, sortkeys: bool = False) -> None:
    """Save a Python dict as a JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(d, f, cls=cls, sort_keys=sortkeys, indent=4)

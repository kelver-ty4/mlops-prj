"""
config.py
Central configuration: paths, MLflow setup, and logging.
All other scripts import from here.
"""

import logging
import logging.config
import sys
from pathlib import Path

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent      # project root
LOGS_DIR  = Path(BASE_DIR, "logs")
DATA_DIR  = Path(BASE_DIR, "data")

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow ─────────────────────────────────────────────────────────────────────
import mlflow

MODEL_REGISTRY    = Path("/tmp/mlflow")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Logging configuration ──────────────────────────────────────────────────────
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal":  {"format": "%(message)s"},
        "detailed": {
            "format": (
                "%(levelname)s %(asctime)s "
                "[%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
            )
        },
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "stream":    sys.stdout,
            "formatter": "minimal",
            "level":     logging.DEBUG,
        },
        "info": {
            "class":        "logging.handlers.RotatingFileHandler",
            "filename":     Path(LOGS_DIR, "info.log"),
            "maxBytes":     10_485_760,   # 10 MB
            "backupCount":  10,
            "formatter":    "detailed",
            "level":        logging.INFO,
        },
        "error": {
            "class":        "logging.handlers.RotatingFileHandler",
            "filename":     Path(LOGS_DIR, "error.log"),
            "maxBytes":     10_485_760,
            "backupCount":  10,
            "formatter":    "detailed",
            "level":        logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level":    logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()

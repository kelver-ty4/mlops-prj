"""
monitoring.py
Drift detection and monitoring using EvidentlyAI.

Compares a reference dataset (training data) against a
current window of production data (recent predictions).

Usage:
    python madewithml/monitoring.py \
        --reference-loc data/projects.csv \
        --current-loc   data/current_window.csv \
        --report-dir    reports/
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)

from madewithml.config import logger

ALERT_THRESHOLD = 0.05          # p-value; below this → drift detected


def load_and_prepare(path: str, text_col: str = "text", label_col: str = "tag") -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    if label_col not in df.columns:
        df[label_col] = None
    df["text_length"] = df["text"].str.split().str.len()
    result = df[["text", label_col, "text_length"]]
    # drop columns that are entirely NaN (e.g. missing label column)
    return result.dropna(axis=1, how="all")


def run_monitoring(
    reference_loc: str = "data/dataset.csv",
    current_loc:   str = "data/current_window.csv",
    report_dir:    str = "reports/",
):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Loading reference and current datasets ...")
    reference = load_and_prepare(reference_loc)
    current   = load_and_prepare(current_loc)

    # align columns — only keep columns present in both datasets
    common_cols = list(set(reference.columns) & set(current.columns))
    reference = reference[common_cols]
    current   = current[common_cols]

    if len(current) == 0:
        logger.warning("Current dataset is empty — skipping drift report")
        return {
            "timestamp":           datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset_drift":       False,
            "share_drifted_cols":  0,
            "num_drifted_columns": 0,
            "num_columns":         0,
            "alert_threshold":     ALERT_THRESHOLD,
        }

    # ── EvidentlyAI Report ────────────────────────────────────────────────────
    logger.info("Running EvidentlyAI drift report ...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ])

    report.run(reference_data=reference, current_data=current)

    # Save HTML report
    report_path = Path(report_dir) / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))
    logger.info(f"HTML report saved → {report_path}")

    # ── Extract JSON results for alerting ────────────────────────────────────
    result     = report.as_dict()
    # Find DatasetDriftMetric by searching for the key "dataset_drift"
    drift_info = {}
    for m in result["metrics"]:
        r = m.get("result", {})
        if "dataset_drift" in r and "number_of_columns" in r:
            drift_info = r
            break

    share_drifted = drift_info.get("share_of_drifted_columns", 0)
    dataset_drifted = drift_info.get("dataset_drift", False)

    summary = {
        "timestamp":           timestamp,
        "dataset_drift":       dataset_drifted,
        "share_drifted_cols":  share_drifted,
        "num_drifted_columns": drift_info.get("number_of_drifted_columns", 0),
        "num_columns":         drift_info.get("number_of_columns", 0),
        "alert_threshold":     ALERT_THRESHOLD,
    }

    summary_path = Path(report_dir) / f"drift_summary_{timestamp}.json"
    json.dump(summary, open(summary_path, "w"), indent=2)
    logger.info(f"Summary saved → {summary_path}")

    # ── Alerting ──────────────────────────────────────────────────────────────
    if dataset_drifted:
        logger.warning(
            f"🚨 DRIFT DETECTED | "
            f"share_drifted_cols={share_drifted:.2%} | "
            f"timestamp={timestamp}"
        )
        logger.warning("Recommended actions:")
        logger.warning("  1. Inspect the drift report HTML")
        logger.warning("  2. Check for data schema changes")
        logger.warning("  3. Trigger retraining pipeline (push to main)")
    else:
        logger.info(
            f"✅ No significant drift | "
            f"share_drifted_cols={share_drifted:.2%} | "
            f"timestamp={timestamp}"
        )

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-loc", default="data/dataset.csv")
    parser.add_argument("--current-loc", default="data/current_window.csv")
    parser.add_argument("--report-dir", default="reports/")
    args = parser.parse_args()
    run_monitoring(
        reference_loc=args.reference_loc,
        current_loc=args.current_loc,
        report_dir=args.report_dir,
    )

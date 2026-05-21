"""
anomaly_detection.py
Detect anomalies in production predictions: confidence drops,
text length outliers, label distribution shifts, request velocity.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import numpy as np
from madewithml.config import logger


def load_predictions(pred_file="data/predictions.jsonl", max_records=200):
    path = Path(pred_file)
    if not path.exists():
        return []
    with open(path) as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines[-max_records:]]


def detect_confidence_anomalies(records, z_threshold=2.0):
    if len(records) < 10:
        return []
    confidences = [r.get("confidence", 0) for r in records]
    mean, std = np.mean(confidences), np.std(confidences)
    if std == 0:
        return []
    anomalies = []
    for r in records:
        z = (r.get("confidence", 0) - mean) / std
        if abs(z) > z_threshold:
            anomalies.append({
                "type": "confidence_anomaly",
                "severity": "high" if abs(z) > 3 else "medium",
                "timestamp": r.get("timestamp", ""),
                "value": r.get("confidence", 0),
                "z_score": round(z, 2),
                "detail": f"Confidence {r.get('confidence', 0):.3f} is {abs(z):.1f}sigma {'below' if z < 0 else 'above'} mean ({mean:.3f})",
            })
    return anomalies


def detect_text_length_anomalies(records):
    if len(records) < 10:
        return []
    lengths = [len(r.get("text", "").split()) for r in records]
    q1, q3 = np.percentile(lengths, 25), np.percentile(lengths, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    anomalies = []
    for r in records:
        length = len(r.get("text", "").split())
        if length < lower or length > upper:
            anomalies.append({
                "type": "text_length_anomaly",
                "severity": "medium",
                "timestamp": r.get("timestamp", ""),
                "value": length,
                "detail": f"Text length ({length} words) outside IQR bounds [{lower:.0f}, {upper:.0f}]",
            })
    return anomalies


def detect_label_shift(records):
    if len(records) < 20:
        return []
    labels = Counter(r.get("label", "") for r in records)
    total = len(records)
    anomalies = []
    for label, count in labels.most_common():
        share = count / total
        if share > 0.6:
            anomalies.append({
                "type": "label_dominance_anomaly",
                "severity": "high" if share > 0.8 else "medium",
                "timestamp": records[-1].get("timestamp", ""),
                "value": round(share, 3),
                "detail": f"Label '{label}' dominates {share:.1%} of recent predictions (threshold: 60%)",
            })
    return anomalies


def detect_velocity_anomaly(records, window_minutes=60):
    if len(records) < 5:
        return []
    timestamps = []
    for r in records:
        ts = r.get("timestamp", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts))
            except ValueError:
                continue
    if len(timestamps) < 5:
        return []
    cutoff = timestamps[-1] - timedelta(minutes=window_minutes)
    recent = sum(1 for t in timestamps if t >= cutoff)
    span = (timestamps[-1] - timestamps[0]).total_seconds() / 60
    expected = len(timestamps) / max(span, 1) * window_minutes
    if expected == 0:
        return []
    ratio = recent / expected
    anomalies = []
    if ratio > 2.5:
        anomalies.append({
            "type": "velocity_anomaly",
            "severity": "medium",
            "timestamp": timestamps[-1].isoformat(),
            "value": round(ratio, 2),
            "detail": f"Request rate spike: {ratio:.1f}x expected ({recent} reqs in {window_minutes}m vs {expected:.0f})",
        })
    elif ratio < 0.3 and len(timestamps) > 20:
        anomalies.append({
            "type": "velocity_anomaly",
            "severity": "low",
            "timestamp": timestamps[-1].isoformat(),
            "value": round(ratio, 2),
            "detail": f"Request rate drop: {ratio:.1f}x expected ({recent} reqs in {window_minutes}m vs {expected:.0f})",
        })
    return anomalies


def run_anomaly_detection(pred_file="data/predictions.jsonl", max_records=200):
    records = load_predictions(pred_file, max_records)
    if not records:
        return {"status": "no_data", "anomalies": []}
    detected = []
    detected.extend(detect_confidence_anomalies(records))
    detected.extend(detect_text_length_anomalies(records))
    detected.extend(detect_label_shift(records))
    detected.extend(detect_velocity_anomaly(records))
    severity_count = Counter(a["severity"] for a in detected)
    return {
        "status": "ok",
        "total_anomalies": len(detected),
        "high": severity_count.get("high", 0),
        "medium": severity_count.get("medium", 0),
        "low": severity_count.get("low", 0),
        "anomalies": detected,
    }

"""
recommendations.py
Generate actionable recommendations based on drift & anomaly results.
"""


DRIFT_RECOMMENDATIONS = {
    "data_drift": {
        "title": "Retrain model with recent data",
        "priority": "high",
        "command": "python madewithml/train.py --experiment-name retrain_$(date +%Y%m%d) --dataset-loc data/current_window.csv --num-epochs 15",
        "detail": "Data drift detected — the input distribution has shifted. Retraining on recent data should restore performance.",
    },
    "target_drift": {
        "title": "Re-balance training dataset",
        "priority": "high",
        "command": "python -c \"import pandas as pd; df = pd.read_csv('data/current_window.csv'); print(df['tag'].value_counts())\"",
        "detail": "Target distribution has shifted. Check if new classes have emerged or existing class ratios changed. Consider stratified sampling.",
    },
    "data_quality": {
        "title": "Inspect data pipeline for missing values",
        "priority": "medium",
        "command": "python -c \"import pandas as pd; df = pd.read_csv('data/current_window.csv'); print(df.isnull().sum())\"",
        "detail": "Data quality metrics indicate missing or malformed inputs. Check the upstream data pipeline.",
    },
}

ANOMALY_RECOMMENDATIONS = {
    "confidence_anomaly": {
        "title": "Calibrate model or adjust confidence threshold",
        "priority": "medium",
        "command": "python madewithml/evaluate.py --experiment-name calibration_check --dataset-loc data/projects.csv",
        "detail": "Unusual confidence scores detected. Re-evaluate model calibration and consider adjusting the prediction threshold.",
    },
    "text_length_anomaly": {
        "title": "Review input validation",
        "priority": "low",
        "command": None,
        "detail": "Unusually short or long inputs detected. Consider adding input length validation or preprocessing checks.",
    },
    "label_dominance_anomaly": {
        "title": "Investigate label distribution bias",
        "priority": "high",
        "command": "python -c \"import pandas as pd; df = pd.read_csv('data/current_window.csv'); print(df['tag'].value_counts(normalize=True))\"",
        "detail": "A single label dominates recent predictions. The model may be biased or production data has shifted heavily.",
    },
    "velocity_anomaly": {
        "title": "Check for automated traffic or service outage",
        "priority": "medium",
        "command": None,
        "detail": "Unusual request volume detected. Verify there is no bot traffic (if spike) or service connectivity issue (if drop).",
    },
}


def generate_recommendations(drift_summary: dict, anomaly_results: dict) -> list[dict]:
    recommendations = []

    if drift_summary.get("dataset_drift"):
        rec = dict(DRIFT_RECOMMENDATIONS["data_drift"])
        rec["source"] = "drift_detection"
        recommendations.append(rec)

    if drift_summary.get("target_drift"):
        rec = dict(DRIFT_RECOMMENDATIONS["target_drift"])
        rec["source"] = "drift_detection"
        recommendations.append(rec)

    if drift_summary.get("data_quality_issues"):
        rec = dict(DRIFT_RECOMMENDATIONS["data_quality"])
        rec["source"] = "drift_detection"
        recommendations.append(rec)

    anomaly_types = set(a["type"] for a in anomaly_results.get("anomalies", []))
    for atype in anomaly_types:
        if atype in ANOMALY_RECOMMENDATIONS:
            rec = dict(ANOMALY_RECOMMENDATIONS[atype])
            rec["source"] = "anomaly_detection"
            recommendations.append(rec)

    if not recommendations:
        recommendations.append({
            "title": "No action needed",
            "priority": "low",
            "command": None,
            "detail": "All metrics are within normal ranges. Model performance is stable.",
            "source": "all_clear",
        })

    return recommendations

# Monitoring Dashboard Implementation Plan

## Files to create

### 1. `madewithml/anomaly_detection.py`
```python
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
                "detail": f"Confidence {r.get('confidence', 0):.3f} is {abs(z):.1f}σ {'below' if z < 0 else 'above'} mean ({mean:.3f})",
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
```

### 2. `madewithml/recommendations.py`
```python
"""
recommendations.py
Generate actionable recommendations based on drift & anomaly results.
"""
from madewithml.config import logger


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
```

### 3. Modified `madewithml/serve.py`

Add to imports:
```python
from datetime import datetime, timezone
from pathlib import Path
from madewithml.anomaly_detection import run_anomaly_detection
from madewithml.recommendations import generate_recommendations
from madewithml.monitoring import run_monitoring
```

Add prediction logging after successful prediction (inside `predict`):
```python
PRED_LOG = Path("data/predictions.jsonl")
def log_prediction(text, label, probabilities):
    entry = {
        "text": text,
        "label": label,
        "confidence": max(probabilities.values()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

Add static file serving for reports:
```python
from fastapi.staticfiles import StaticFiles
```

Add monitoring endpoints:
```python
@app.get("/monitor")
def monitor_page():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(MONITOR_HTML)

@app.get("/monitor/status")
def monitor_status():
    drift = run_monitoring(
        reference_loc="data/projects.csv",
        current_loc="data/projects.csv",  # will be replaced by predictions log
        report_dir="reports/",
    )
    anomalies = run_anomaly_detection()
    recommendations = generate_recommendations(drift, anomalies)
    return {
        "drift": drift,
        "anomalies": anomalies,
        "recommendations": recommendations,
    }

@app.get("/monitor/history")
def monitor_history():
    import glob
    reports_dir = Path("reports/")
    if not reports_dir.exists():
        return {"reports": []}
    summaries = sorted(reports_dir.glob("drift_summary_*.json"), reverse=True)
    history = []
    for s in summaries[:20]:
        with open(s) as f:
            data = json.load(f)
        history.append(data)
    return {"reports": history}

@app.get("/monitor/reports/{filename}")
def monitor_report(filename: str):
    from fastapi.responses import FileResponse
    report_path = Path("reports/") / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path)
```

Add to startup:
```python
@app.on_event("startup")
def startup():
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    Path("reports/").mkdir(parents=True, exist_ok=True)
```

### 4. Dashboard HTML (embedded as a global variable in serve.py)

The `MONITOR_HTML` variable contains a self-contained HTML page with:
- Styled cards for drift status, anomalies, recommendations
- Fetch from `/monitor/status` and `/monitor/history`
- Color-coded severity badges
- Copyable commands for each recommendation

Full HTML is in the dashboard section below.

### 5. Modified `Jenkinsfile`

Add after `Health Check` stage:

```groovy
stage('Monitor') {
    steps {
        echo '── Running drift monitoring ──'
        sh '''
            . ${VENV_DIR}/bin/activate
            python -c "
import pandas as pd
df = pd.read_csv('data/projects.csv').sample(200, random_state=\$(date +%s))
df.to_csv('data/current_window.csv', index=False)
"
            python madewithml/monitoring.py \
                --reference-loc data/projects.csv \
                --current-loc data/current_window.csv \
                --report-dir reports/
        '''
    }
    post {
        always {
            archiveArtifacts artifacts: 'reports/*.html, reports/*.json'
        }
    }
}
```

### 6. Modified `.gitignore`

Add:
```
data/predictions.jsonl
reports/
```

---

## Dashboard HTML

Single self-contained page embedded in `serve.py`. Key features:

- **Status bar** at top: green (ok), yellow (warnings), red (drift/anomalies)
- **Drift card**: Data drift % gauge, target drift indicator, data quality score
- **Anomalies card**: Table with type, severity badge, timestamp, detail
- **Recommendations card**: List with priority badges, copyable commands
- **History card**: Timeline table of past monitoring runs with links to HTML reports
- Auto-refresh button, manual refresh, last-updated timestamp

---

## Implementation order

1. Create `madewithml/anomaly_detection.py`
2. Create `madewithml/recommendations.py`
3. Modify `madewithml/serve.py` (add logging + monitoring endpoints + dashboard HTML)
4. Update `Jenkinsfile` (add Monitor stage)
5. Update `.gitignore`
6. Run tests: `pytest tests/ -v`
7. Start API: `uvicorn madewithml.serve:app --reload --port 8000`
8. Open `http://localhost:8000/monitor`

---

## Dashboard HTML source (paste into serve.py as MONITOR_HTML)

```html
MONITOR_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Monitor Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; color: #2d3436; padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
  .header h1 { font-size: 24px; }
  .status-badge { padding: 6px 14px; border-radius: 12px; font-weight: 600; font-size: 13px; }
  .status-ok { background: #55efc4; color: #00b894; }
  .status-warning { background: #ffeaa7; color: #fdcb6e; }
  .status-critical { background: #fab1a0; color: #d63031; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
  .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .card h2 { font-size: 15px; color: #636e72; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f2f6; }
  .metric-label { color: #636e72; font-size: 13px; }
  .metric-value { font-weight: 600; font-size: 13px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600; }
  .badge-high { background: #fab1a0; color: #d63031; }
  .badge-medium { background: #ffeaa7; color: #e17055; }
  .badge-low { background: #dfe6e9; color: #636e72; }
  .badge-ok { background: #55efc4; color: #00b894; }
  .anomaly-item { padding: 8px 0; border-bottom: 1px solid #f1f2f6; font-size: 13px; }
  .rec-item { padding: 10px 0; border-bottom: 1px solid #f1f2f6; }
  .rec-item:last-child { border-bottom: none; }
  .rec-title { font-weight: 600; font-size: 13px; }
  .rec-detail { font-size: 12px; color: #636e72; margin-top: 3px; }
  .rec-command { background: #f1f2f6; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 12px; margin-top: 6px; display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
  .rec-command:hover { background: #dfe6e9; }
  .btn { background: #0984e3; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; }
  .btn:hover { background: #0773c5; }
  .empty { text-align: center; color: #b2bec3; padding: 20px; font-size: 13px; }
  .timestamp { font-size: 12px; color: #b2bec3; margin-top: 12px; text-align: right; }
  .history-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .history-table th { text-align: left; padding: 6px 8px; color: #636e72; border-bottom: 2px solid #f1f2f6; }
  .history-table td { padding: 6px 8px; border-bottom: 1px solid #f1f2f6; }
</style>
</head>
<body>
<div class="header">
  <h1>Model Monitor</h1>
  <div>
    <span id="statusBadge" class="status-badge status-ok">Loading...</span>
    <button class="btn" onclick="refresh()" style="margin-left:10px">Refresh</button>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Data Drift</h2>
    <div id="driftContent"><div class="empty">Loading...</div></div>
  </div>
  <div class="card">
    <h2>Anomalies (<span id="anomalyCount">0</span>)</h2>
    <div id="anomalyContent"><div class="empty">Loading...</div></div>
  </div>
  <div class="card" style="grid-column: span 1;">
    <h2>Recommendations</h2>
    <div id="recContent"><div class="empty">Loading...</div></div>
  </div>
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Monitoring History</h2>
    <div id="historyContent"><div class="empty">Loading...</div></div>
  </div>
</div>
<div class="timestamp" id="lastUpdated"></div>

<script>
async function refresh() {
  try {
    const [statusRes, historyRes] = await Promise.all([
      fetch('/monitor/status'),
      fetch('/monitor/history')
    ]);
    const status = await statusRes.json();
    const history = await historyRes.json();

    // Status badge
    const critical = (status.drift?.dataset_drift) || status.anomalies?.high > 0;
    const warning = status.anomalies?.medium > 0;
    const badge = document.getElementById('statusBadge');
    if (critical) { badge.textContent = 'Critical'; badge.className = 'status-badge status-critical'; }
    else if (warning) { badge.textContent = 'Warning'; badge.className = 'status-badge status-warning'; }
    else { badge.textContent = 'Healthy'; badge.className = 'status-badge status-ok'; }

    // Drift
    const d = status.drift || {};
    document.getElementById('driftContent').innerHTML = `
      <div class="metric-row"><span class="metric-label">Dataset Drift</span><span class="metric-value">${d.dataset_drift ? 'YES' : 'No'}</span></div>
      <div class="metric-row"><span class="metric-label">Drifted Columns</span><span class="metric-value">${d.num_drifted_columns || 0} / ${d.num_columns || 0}</span></div>
      <div class="metric-row"><span class="metric-label">Share Drifted</span><span class="metric-value">${((d.share_drifted_cols || 0) * 100).toFixed(1)}%</span></div>
      <div class="metric-row"><span class="metric-label">Threshold</span><span class="metric-value">p < ${d.alert_threshold || 0.05}</span></div>
    `;

    // Anomalies
    const a = status.anomalies || { anomalies: [], total_anomalies: 0 };
    document.getElementById('anomalyCount').textContent = a.total_anomalies;
    if (a.anomalies?.length) {
      document.getElementById('anomalyContent').innerHTML = a.anomalies.map(an => `
        <div class="anomaly-item">
          <span class="badge badge-${an.severity}">${an.severity}</span>
          <strong>${an.type.replace(/_/g, ' ')}</strong>
          <div style="color:#636e72;font-size:12px;margin-top:2px">${an.detail}</div>
        </div>
      `).join('');
    } else {
      document.getElementById('anomalyContent').innerHTML = '<div class="empty">No anomalies detected</div>';
    }

    // Recommendations
    const recs = status.recommendations || [];
    if (recs.length) {
      document.getElementById('recContent').innerHTML = recs.map(r => `
        <div class="rec-item">
          <div class="rec-title"><span class="badge badge-${r.priority === 'high' ? 'high' : r.priority === 'medium' ? 'medium' : 'low'}">${r.priority}</span> ${r.title}</div>
          <div class="rec-detail">${r.detail}</div>
          ${r.command ? `<div class="rec-command" onclick="copy(this)">${r.command} <span style="font-size:10px">COPY</span></div>` : ''}
        </div>
      `).join('');
    } else {
      document.getElementById('recContent').innerHTML = '<div class="empty">No recommendations</div>';
    }

    // History
    const reports = history.reports || [];
    if (reports.length) {
      document.getElementById('historyContent').innerHTML = `
        <table class="history-table">
          <tr><th>Time</th><th>Drift</th><th>Drifted Cols</th><th>Report</th></tr>
          ${reports.map(r => `
            <tr>
              <td>${r.timestamp || '-'}</td>
              <td>${r.dataset_drift ? 'YES' : 'No'}</td>
              <td>${r.num_drifted_columns || 0} / ${r.num_columns || 0}</td>
              <td><a href="/monitor/reports/drift_report_${r.timestamp}.html" target="_blank">HTML</a></td>
            </tr>
          `).join('')}
        </table>
      `;
    } else {
      document.getElementById('historyContent').innerHTML = '<div class="empty">No monitoring history yet</div>';
    }

    document.getElementById('lastUpdated').textContent = 'Last updated: ' + new Date().toLocaleString();
  } catch (e) {
    document.getElementById('driftContent').innerHTML = '<div class="empty">Error loading data</div>';
    document.getElementById('lastUpdated').textContent = 'Error: ' + e.message;
  }
}

function copy(el) {
  const text = el.textContent.replace('COPY', '').trim();
  navigator.clipboard.writeText(text).then(() => {
    const orig = el.innerHTML;
    el.innerHTML = 'Copied!';
    setTimeout(() => el.innerHTML = orig, 1500);
  });
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
'''
```

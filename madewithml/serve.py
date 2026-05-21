"""
serve.py
FastAPI app that serves the trained model.

Run locally:
    uvicorn madewithml.serve:app --reload --port 8000

Endpoints:
    GET  /health        → liveness check (used by Jenkins)
    POST /predict       → classify text
"""

import json
import pickle
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchtext.data.utils import get_tokenizer

from madewithml.config import logger
from madewithml.models import TextClassifier
from datetime import datetime, timezone
from fastapi.staticfiles import StaticFiles
from madewithml.anomaly_detection import run_anomaly_detection
from madewithml.recommendations import generate_recommendations
from madewithml.monitoring import run_monitoring

app = FastAPI(title="MadeWithML Classifier", version="1.0")

# ── Load artifacts at startup ──────────────────────────────────────────────────
ARTIFACT_DIR = Path("/tmp/eval_artifacts")
PRED_LOG = Path("data/predictions.jsonl")


def load_model():
    if not ARTIFACT_DIR.exists():
        raise RuntimeError(f"Artifact directory not found: {ARTIFACT_DIR}")

    with open(ARTIFACT_DIR / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    idx2label = json.load(open(ARTIFACT_DIR / "idx2label.json"))
    tokenizer = get_tokenizer("basic_english")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextClassifier(len(vocab), embed_dim=64, num_classes=len(idx2label))
    model.load_state_dict(torch.load(ARTIFACT_DIR / "best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    return model, vocab, idx2label, tokenizer, device


try:
    _model, _vocab, _idx2label, _tokenizer, _device = load_model()
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"Could not load model: {e}")
    _model = None
    _vocab = None
    _idx2label = None
    _tokenizer = None
    _device = None


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


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]


# ── Shared UI layout ──────────────────────────────────────────────────────────
UI_HEAD = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; color: #2d3436; }
  nav { background: #2d3436; padding: 0 20px; display: flex; align-items: center; height: 52px; }
  nav a { color: #dfe6e9; text-decoration: none; padding: 0 16px; font-size: 14px; font-weight: 500; line-height: 52px; }
  nav a:hover, nav a.active { color: #fff; background: #636e72; }
  nav .brand { color: #55efc4; font-weight: 700; font-size: 16px; margin-right: 24px; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px 20px; }
  .header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
  .header-row h1 { font-size: 22px; }
  .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .card h2 { font-size: 15px; color: #636e72; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
  .btn { background: #0984e3; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; }
  .btn:hover { background: #0773c5; }
  .btn-green { background: #00b894; }
  .btn-green:hover { background: #00a381; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600; }
  .badge-high { background: #fab1a0; color: #d63031; }
  .badge-medium { background: #ffeaa7; color: #e17055; }
  .badge-low { background: #dfe6e9; color: #636e72; }
  .badge-ok { background: #55efc4; color: #00b894; }
  .status-badge { padding: 6px 14px; border-radius: 12px; font-weight: 600; font-size: 13px; }
  .status-ok { background: #55efc4; color: #00b894; }
  .status-warning { background: #ffeaa7; color: #fdcb6e; }
  .status-critical { background: #fab1a0; color: #d63031; }
  .empty { text-align: center; color: #b2bec3; padding: 20px; font-size: 13px; }
  .metric-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f2f6; }
  .metric-label { color: #636e72; font-size: 13px; }
  .metric-value { font-weight: 600; font-size: 13px; }
  .timestamp { font-size: 12px; color: #b2bec3; margin-top: 12px; text-align: right; }
  input, textarea { width: 100%; padding: 10px 12px; border: 1px solid #dfe6e9; border-radius: 8px; font-size: 14px; font-family: inherit; }
  input:focus, textarea:focus { outline: none; border-color: #0984e3; }
  textarea { min-height: 100px; resize: vertical; }
  .result-box { background: #f1f2f6; border-radius: 8px; padding: 16px; margin-top: 12px; }
  .prob-bar { display: flex; align-items: center; margin: 4px 0; }
  .prob-bar-fill { height: 20px; border-radius: 4px; min-width: 2px; }
  .prob-label { width: 80px; font-size: 13px; font-weight: 600; }
  .prob-pct { width: 50px; text-align: right; font-size: 12px; color: #636e72; }
  .hero-card { text-align: center; padding: 40px 20px; }
  .hero-card h1 { font-size: 28px; margin-bottom: 8px; }
  .hero-card p { color: #636e72; margin-bottom: 24px; }
  .home-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-top: 24px; }
  .home-card { background: white; border-radius: 10px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; cursor: pointer; transition: transform 0.15s; text-decoration: none; color: inherit; display: block; }
  .home-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
  .home-card .icon { font-size: 32px; margin-bottom: 12px; }
  .home-card h3 { font-size: 16px; margin-bottom: 6px; }
  .home-card p { font-size: 13px; color: #636e72; }
  .anomaly-item { padding: 8px 0; border-bottom: 1px solid #f1f2f6; font-size: 13px; }
  .rec-item { padding: 10px 0; border-bottom: 1px solid #f1f2f6; }
  .rec-item:last-child { border-bottom: none; }
  .rec-title { font-weight: 600; font-size: 13px; }
  .rec-detail { font-size: 12px; color: #636e72; margin-top: 3px; }
  .rec-command { background: #f1f2f6; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 12px; margin-top: 6px; display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
  .rec-command:hover { background: #dfe6e9; }
  .history-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .history-table th { text-align: left; padding: 6px 8px; color: #636e72; border-bottom: 2px solid #f1f2f6; }
  .history-table td { padding: 6px 8px; border-bottom: 1px solid #f1f2f6; }
</style>
</head>
<body>
<nav>
  <span class="brand">MadeWithML</span>
  <a href="/" class="{home_active}">Home</a>
  <a href="/ui/predict" class="{predict_active}">Predict</a>
  <a href="/monitor" class="{monitor_active}">Monitor</a>
  <a href="/docs">API Docs</a>
</nav>
<div class="container">
'''

UI_FOOT = '''
</div>
</body>
</html>
'''


def page(title, body, home_active="", predict_active="", monitor_active=""):
    from fastapi.responses import HTMLResponse
    html = UI_HEAD.replace("{title}", title) \
                  .replace("{home_active}", home_active) \
                  .replace("{predict_active}", predict_active) \
                  .replace("{monitor_active}", monitor_active) \
            + body + UI_FOOT
    return HTMLResponse(html)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe — Jenkins calls this after deploy."""
    return {"status": "ok"}


@app.get("/")
def home():
    body = '''
    <div class="hero-card card">
      <h1>MadeWithML Classifier</h1>
      <p>Text classification API with monitoring, drift detection, and anomaly alerting.</p>
      <div style="display:flex;gap:10px;justify-content:center">
        <span class="status-badge status-ok">API Online</span>
        <span class="badge badge-ok">v1.0</span>
      </div>
    </div>
    <div class="home-grid">
      <a href="/ui/predict" class="home-card">
        <div class="icon">🔮</div>
        <h3>Predict</h3>
        <p>Classify text with the trained model. Test real-time predictions.</p>
      </a>
      <a href="/monitor" class="home-card">
        <div class="icon">📊</div>
        <h3>Monitor</h3>
        <p>Drift detection, anomaly alerts, and actionable recommendations.</p>
      </a>
      <a href="/docs" class="home-card">
        <div class="icon">📖</div>
        <h3>API Docs</h3>
        <p>Swagger documentation for all API endpoints.</p>
      </a>
      <a href="https://github.com/kelver-ty4/mlops-prj" class="home-card">
        <div class="icon">📦</div>
        <h3>Source Code</h3>
        <p>GitHub repository with CI/CD pipeline and ML code.</p>
      </a>
    </div>
    '''
    return page("MadeWithML Classifier", body, home_active="active")


@app.get("/ui/predict")
def predict_page():
    body = '''
    <div class="card" style="max-width:640px;margin:0 auto">
      <h2>Test Prediction</h2>
      <textarea id="inputText" placeholder="Enter text to classify... e.g. 'BERT for NLP classification'"></textarea>
      <div style="margin-top:10px;display:flex;gap:8px">
        <button class="btn btn-green" onclick="classify()">Classify</button>
        <button class="btn" onclick="document.getElementById('inputText').value=''" style="background:#636e72">Clear</button>
      </div>
      <div id="result" style="margin-top:16px"></div>
    </div>
    <script>
    async function classify() {
      const text = document.getElementById('inputText').value.trim();
      if (!text) { document.getElementById('result').innerHTML = '<div class="empty">Enter some text first</div>'; return; }
      document.getElementById('result').innerHTML = '<div class="empty">Classifying...</div>';
      try {
        const res = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text}) });
        if (!res.ok) { document.getElementById('result').innerHTML = '<div class="empty">Error: ' + res.status + '</div>'; return; }
        const data = await res.json();
        const topLabel = data.label;
        const topProb = (data.probabilities[topLabel] * 100).toFixed(1);
        const probs = Object.entries(data.probabilities).sort((a,b) => b[1]-a[1]);
        const colors = ['#0984e3','#00b894','#fdcb6e','#e17055','#6c5ce7'];
        document.getElementById('result').innerHTML = `
          <div class="result-box">
            <div style="font-size:18px;font-weight:700;margin-bottom:4px">${topLabel}</div>
            <div style="font-size:13px;color:#636e72;margin-bottom:12px">Confidence: ${topProb}%</div>
            ${probs.map(([l,p],i) => `
              <div class="prob-bar">
                <span class="prob-label">${l}</span>
                <div style="flex:1;background:#dfe6e9;border-radius:4px;height:20px;margin:0 8px">
                  <div class="prob-bar-fill" style="width:${(p*100).toFixed(1)}%;background:${colors[i%colors.length]};transition:width 0.3s"></div>
                </div>
                <span class="prob-pct">${(p*100).toFixed(1)}%</span>
              </div>
            `).join('')}
          </div>
        `;
      } catch(e) {
        document.getElementById('result').innerHTML = '<div class="empty">Error: ' + e.message + '</div>';
      }
    }
    document.getElementById('inputText').addEventListener('keydown', e => { if (e.key === 'Enter' && e.ctrlKey) classify(); });
    </script>
    '''
    return page("Predict — MadeWithML", body, predict_active="active")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokens  = torch.tensor(_vocab(_tokenizer(request.text)), dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        logits = _model(tokens.to(_device), offsets.to(_device))
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        pred   = logits.argmax(1).item()

    label = _idx2label[str(pred)]
    probabilities = {_idx2label[str(i)]: float(p) for i, p in enumerate(probs)}
    log_prediction(request.text, label, probabilities)
    return PredictResponse(
        label=label,
        probabilities=probabilities,
    )


# ── Monitor HTML ────────────────────────────────────────────────────────────────
MONITOR_HTML = UI_HEAD.replace("{title}", "Monitor — MadeWithML") \
    .replace("{home_active}", "") \
    .replace("{predict_active}", "") \
    .replace("{monitor_active}", "active") + '''
<div class="header-row">
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
  <div class="card">
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

    const critical = (status.drift?.dataset_drift) || status.anomalies?.high > 0;
    const warning = status.anomalies?.medium > 0;
    const badge = document.getElementById('statusBadge');
    if (critical) { badge.textContent = 'Critical'; badge.className = 'status-badge status-critical'; }
    else if (warning) { badge.textContent = 'Warning'; badge.className = 'status-badge status-warning'; }
    else { badge.textContent = 'Healthy'; badge.className = 'status-badge status-ok'; }

    const d = status.drift || {};
    document.getElementById('driftContent').innerHTML = `
      <div class="metric-row"><span class="metric-label">Dataset Drift</span><span class="metric-value">${d.dataset_drift ? 'YES' : 'No'}</span></div>
      <div class="metric-row"><span class="metric-label">Drifted Columns</span><span class="metric-value">${d.num_drifted_columns || 0} / ${d.num_columns || 0}</span></div>
      <div class="metric-row"><span class="metric-label">Share Drifted</span><span class="metric-value">${((d.share_drifted_cols || 0) * 100).toFixed(1)}%</span></div>
      <div class="metric-row"><span class="metric-label">Threshold</span><span class="metric-value">p < ${d.alert_threshold || 0.05}</span></div>
    `;

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
''' + UI_FOOT


# ── Monitoring routes ─────────────────────────────────────────────────────────
@app.get("/monitor")
def monitor_page():
    return HTMLResponse(MONITOR_HTML)


@app.get("/monitor/status")
def monitor_status():
    drift = run_monitoring(
        reference_loc="data/projects.csv",
        current_loc="data/projects.csv",
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


@app.on_event("startup")
def startup():
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    Path("reports/").mkdir(parents=True, exist_ok=True)

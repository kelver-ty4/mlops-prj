# MLOps Pipeline — Complete Project Report

## 1. What Is This Project?

A full **MLOps pipeline** that classifies GitHub project descriptions into ML domains:

- **computer-vision**
- **natural-language-processing**
- **mlops**

Example: *"classifying images with deep neural networks"* → **computer-vision** (99.9% confidence)

---

## 2. What Is MLOps and Why Do We Need It?

**MLOps** = DevOps applied to Machine Learning. It brings the same engineering practices (automation, testing, CI/CD, monitoring) that software developers use to ML projects.

**Why it matters:**

| Without MLOps | With MLOps |
|--------------|------------|
| Train models manually on your laptop | Automated training on every code change |
| No tests — bugs slip into production | Tests + lint run automatically |
| Deploy by copying files manually | One-click deploy via Docker |
| No tracking — which model is better? | MLflow logs every experiment |
| No performance gate — bad models ship | Pipeline blocks bad models (F1 < 0.70) |

---

## 3. Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │           GitHub Repository              │
                    │   kelver-ty4/mlops-prj                   │
                    └────────────────┬────────────────────────┘
                                     │ push
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │          Jenkins Pipeline                 │
                    │                                          │
                    │  ┌──────┐  ┌──────┐  ┌──────┐           │
                    │  │Setup │─▶│ Lint │─▶│ Test │           │
                    │  └──────┘  └──────┘  └──────┘           │
                    │                        │                 │
                    │                        ▼                 │
                    │  ┌──────┐  ┌──────┐  ┌──────┐           │
                    │  │Train │─▶│ Eval │─▶│ Gate │           │
                    │  └──────┘  └──────┘  └──────┘  ← CI    │
                    │                        │                 │
                    │                        ▼                 │
                    │  ┌────────┐  ┌──────┐  ┌──────┐         │
                    │  │ Docker │─▶│Deploy│─▶│Health│         │
                    │  │ Build  │  │      │  │Check │  ← CD   │
                    │  └────────┘  └──────┘  └──────┘         │
                    └─────────────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │   FastAPI App (port 8000)                │
                    │   /health  →  {"status":"ok"}           │
                    │   /predict →  {"label":"...","probs":..}│
                    └─────────────────────────────────────────┘
```

### Key Components

| Component | Role | Why This Technology? |
|-----------|------|---------------------|
| **PyTorch + torchtext** | Train a text classifier | Industry-standard deep learning framework |
| **MLflow** | Experiment tracking | Logs every run's metrics, artifacts, params |
| **FastAPI** | Serve the model via REST API | Modern, fast, auto-docs |
| **Docker** | Containerize the app | Reproducible deployment everywhere |
| **Jenkins** | CI/CD automation | Automates the entire pipeline on every push |
| **flake8** | Code linting | Enforces code quality |
| **pytest** | Testing | 22 tests validate every component |

---

## 4. The ML Model

### What it does

Text classifier using an **EmbeddingBag** layer (learns word embeddings) → fully connected layer → softmax over 3 classes.

### Why EmbeddingBag?

`EmbeddingBag` is more efficient than `Embedding` + averaging — it handles variable-length sequences in a single operation. No padding needed.

### Training

- **Data**: 764 GitHub project descriptions → 658 after preprocessing
- **Split**: 460 train / 66 validation / 132 test
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss**: Cross-entropy

### Performance

| Metric | Score |
|--------|-------|
| Validation accuracy | 77.27% |
| Test F1 (weighted) | 0.8287 |
| Precision | 0.85 |
| Recall | 0.83 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| computer-vision | 0.96 | 0.79 | 0.87 |
| mlops | 0.75 | 0.46 | 0.57 |
| NLP | 0.77 | 0.95 | 0.85 |

---

## 5. Pipeline Stages — Explained

### Stage 1 — Setup (Why?)

Create a clean Python environment and install all dependencies. Every build starts fresh so there are no conflicts between runs.

```groovy
python3 -m venv .venv        // isolated environment
pip install -r requirements.txt  // exact versions
pip install -e .                 // install our package
```

- `pip install -e .` registers `madewithml/` as an importable package. Without this, `from madewithml.config import logger` fails with `ModuleNotFoundError`.

### Stage 2 — Lint (Why?)

Enforces consistent code style. Catches unused imports, bad formatting, etc. *before* running tests.

```groovy
flake8 madewithml/ --max-line-length=99 --ignore=E501,W503,E221,E241,E226,E402
```

We ignore E402 because `config.py` imports `mlflow` after path setup (necessary for the import to work).

### Stage 3 — Test (Why?)

22 unit tests validate:
- **data**: text cleaning, preprocessing, train/val/test split
- **models**: output shape, softmax sums to 1, different seeds give different weights
- **API**: health endpoint, predict returns 200 + label + probabilities

Tests must pass before training — no point training on broken data.

### Stage 4 — Train (Why?)

Trains the model and logs everything to **MLflow** (experiment name, run ID, metrics per epoch, model artifacts).

```bash
python madewithml/train.py \
    --experiment-name "ci_run_${BUILD_NUMBER}" \
    --dataset-loc data/dataset.csv \
    --num-epochs 10
```

Why MLflow? So we can compare runs, roll back to a previous model, and track *which code + data + hyperparams* produced each model.

### Stage 5 — Evaluate (Why?)

Loads the trained model from MLflow, runs inference on the held-out test set, computes classification metrics, and writes `metrics.json`.

The `metrics.json` file is the bridge between the ML world and the DevOps world — Jenkins reads it in the next stage.

### Stage 6 — Performance Gate (Why?)

Reads the F1 score from `metrics.json`. If F1 < 0.70, the pipeline **fails** and the bad model never gets deployed.

```python
f1 = json.load("metrics.json")["f1"]
if f1 < 0.70:
    sys.exit(1)  // pipeline fails
```

This is the **key MLOps concept**: automated quality gates prevent regressions.

### Stage 7 — Build Docker Image (Why?)

Packages the model server into a Docker container so it runs identically everywhere.

```dockerfile
FROM python:3.10-slim            // small base image
RUN pip install -r requirements.txt
COPY madewithml/ ./madewithml/
COPY data/ ./data/
CMD ["uvicorn", "madewithml.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

Why `--network host`? In our environment, Docker's default bridge network can't resolve DNS. `--network host` lets the container use the host's network stack directly.

### Stage 8 — Deploy (Why?)

Stops any old container, starts the new one with:
- `--network host` → shares host network (no port mapping issues)
- `-v` mount → shares model artifacts from CI workspace

### Stage 9 — Health Check (Why?)

Confirms the container is actually serving requests before declaring success.

```bash
curl --fail http://localhost:8000/health
```

If this fails, the pipeline fails — no broken deployments.

---

## 6. Why Jenkins?

### What is Jenkins?

Jenkins is an **automation server** — it watches a Git repo and runs a pipeline every time code changes.

### Why not GitHub Actions?

| | Jenkins | GitHub Actions |
|---|---|---|
| **Setup** | Manual (install server) | Built into GitHub |
| **Control** | Full (own server) | Limited to GitHub |
| **Cost** | Free (self-hosted) | Free for public repos |
| **Learning** | More complex | Simpler |

Jenkins was chosen for the project because:
1. It runs **locally** (no internet needed)
2. You have **full control** over the environment
3. It's the industry standard for enterprise CI/CD

### How it works here

```
You push code ──→ GitHub ──→ Jenkins (polls/trigger) ──→ Pipeline runs
```

Since Jenkins is on `localhost:8080`, GitHub can't send webhooks to it. We use **Build Now** manually. In production, you'd use a public URL or ngrok.

---

## 7. All 13 Bugs We Fixed

### Bug 1 — Typer vs Click 8.4.0
**Symptom**: `AttributeError: module 'typer' has no attribute 'Option'`  
**Root cause**: Typer 0.9.0 broke compatibility with Click ≥ 8.4.0  
**Fix**: Replaced `typer.Option()` + `app()` with `argparse` everywhere

### Bug 2 — Wrong Dataset
**Symptom**: `KeyError: 'tag'`  
**Root cause**: Default was `data/projects.csv` (no `tag` column)  
**Fix**: Changed default to `data/dataset.csv`

### Bug 3 — Sparse Embedding + Adam
**Symptom**: `ValueError: SparseAdam does not support sparse gradients`  
**Root cause**: `EmbeddingBag(sparse=True)` requires `SparseAdam`, code used `Adam`  
**Fix**: Set `EmbeddingBag(sparse=False)`

### Bug 4 — Tensor Shape Mistmatch
**Symptom**: Runtime error when making predictions  
**Root cause**: `.unsqueeze(0)` added wrong dimension  
**Fix**: Removed `.unsqueeze(0)` from `predict.py` and `serve.py`

### Bug 5 — Missing pkg_resources
**Symptom**: `ModuleNotFoundError: No module named 'pkg_resources'`  
**Root cause**: MLflow imports `pkg_resources`; setuptools ≥ 72 removed it  
**Fix**: Pinned `setuptools<72` in requirements.txt

### Bug 6 — Package Not Installed
**Symptom**: `ModuleNotFoundError: No module named 'madewithml'`  
**Root cause**: Scripts use `from madewithml.xxx import ...` but package isn't installed  
**Fix**: Added `pip install -e .` to Setup stage

### Bug 7 — Permission Denied /tmp/mlflow
**Symptom**: `PermissionError: [Errno 13] Permission denied: '/tmp/mlflow'`  
**Root cause**: Hardcoded `/tmp/mlflow` owned by `elko`, Jenkins runs as `jenkins`  
**Fix**: Made path configurable via `MODEL_DIR` env var; set to workspace in CI

### Bug 8 — Permission Denied /tmp/eval_artifacts
**Symptom**: Same as above for evaluate.py  
**Root cause**: Hardcoded `/tmp/eval_artifacts`  
**Fix**: Changed to workspace-relative `eval_artifacts/`, create dir before use

### Bug 9 — Docker DNS
**Symptom**: `Could not find a version that satisfies the requirement torch==2.2.2+cpu`  
**Root cause**: Docker can't resolve PyPI inside build container  
**Fix**: Added `--network host` to `docker build`

### Bug 10 — Port Forwarding Broken
**Symptom**: "Connection reset by peer" when hitting localhost:8000  
**Root cause**: Docker port mapping not working properly (iptables issue)  
**Fix**: Switched to `--network host` for `docker run`

### Bug 11 — Health Returns 503
**Symptom**: `curl --fail` fails because `/health` returns 503 when model not loaded  
**Root cause**: Health endpoint returned 503 for liveness check  
**Fix**: Made `/health` always return 200 (liveness ≠ model readiness)

### Bug 12 — Branch Guard Blocking CD
**Symptom**: Docker/Deploy/Health stages skip on every build  
**Root cause**: `when { branch 'main' }` — Pipeline jobs have no branch context  
**Fix**: Removed `when` blocks

### Bug 13 — Jenkins Can't Run Docker
**Symptom**: `docker: command not found` or permission error  
**Root cause**: `jenkins` user not in `docker` group  
**Fix**: `sudo usermod -aG docker jenkins`

---

## 8. How to Reproduce on Another Machine

### Prerequisites

- Ubuntu 22.04+ (or any Linux)
- Python 3.10+
- Docker
- Git

### Step-by-step

```bash
# 1. Clone the repo
git clone https://github.com/kelver-ty4/mlops-prj.git
cd mlops-prj

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Run lint
flake8 madewithml/ --max-line-length=99 --ignore=E501,W503,E221,E241,E226,E402

# 5. Run tests
pytest tests/ -v --tb=short

# 6. Train
python madewithml/train.py --experiment-name my_exp --dataset-loc data/dataset.csv --num-epochs 10

# 7. Evaluate
python madewithml/evaluate.py --experiment-name my_exp --dataset-loc data/dataset.csv

# 8. Serve locally
uvicorn madewithml.serve:app --reload --port 8000

# 9. Test prediction (in another terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"building a sentiment analysis model with transformers"}'

# 10. Docker
docker build --network host -t mlops-app .
docker run --network host --name mlops-app mlops-app:latest
```

### For Jenkins CI/CD

```bash
# Install Jenkins
sudo apt install jenkins   # or download from jenkins.io

# Add jenkins user to docker group
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins

# Access Jenkins at http://localhost:8080
# Create a Pipeline job
# Set Pipeline > Definition > Pipeline script from SCM
# SCM: Git, Repository URL: https://github.com/kelver-ty4/mlops-prj
# Script Path: Jenkinsfile
# Save → Build Now
```

---

## 9. Final Pipeline Run

**Build #15** — Full green pipeline:

```
Setup    ✅  venv + install (47s)
Lint     ✅  flake8 (3s)
Test     ✅  22/22 passed (5s)
Train    ✅  val_acc 77.27% (3s)
Evaluate ✅  F1 0.8287 (3s)
Gate     ✅  0.8287 > 0.70 (1s)
Docker   ✅  image built (cached, 0.2s)
Deploy   ✅  container running on port 8000
Health   ✅  {"status":"ok"}
```

End-to-end time: ~62 seconds (build + install takes most of it).

---

## 10. Key Takeaways

1. **MLOps = DevOps for ML** — same automation, testing, CI/CD principles applied to ML
2. **Automated gates prevent bad models** — F1 threshold blocks regressions
3. **Reproducibility matters** — Docker + MLflow + pinned deps = every run is traceable
4. **CI/CD is not just for code** — models need testing, linting, and versioning too
5. **Start simple, fix iteratively** — we hit 13 bugs but solved each one systematically

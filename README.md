# MLOps Full Project

End-to-end MLOps pipeline: **Notebooks → Scripts → Logging → Reproducibility → CI/CD → Monitoring**

---

## Project structure

```
mlops_project/
├── madewithml/
│   ├── config.py        # Paths, MLflow URI, logging config
│   ├── utils.py         # set_seeds, load_dict, save_dict
│   ├── data.py          # load_data, preprocess, split_data
│   ├── models.py        # TextClassifier (EmbeddingBag → Linear)
│   ├── train.py         # Training loop + MLflow logging
│   ├── evaluate.py      # Test-set evaluation + metrics.json
│   ├── predict.py       # Single-input inference CLI
│   ├── serve.py         # FastAPI app (GET /health, POST /predict)
│   └── monitoring.py    # EvidentlyAI drift detection
├── tests/
│   ├── test_data.py     # Unit tests for data functions
│   ├── test_models.py   # Unit tests for model architecture
│   └── test_api.py      # Integration tests for FastAPI endpoints
├── data/
│   └── projects.csv     # Dataset (title, description, tag)
├── logs/
│   ├── info.log
│   └── error.log
├── reports/             # Generated drift HTML reports
├── Dockerfile
├── Jenkinsfile          # CI/CD pipeline definition
├── requirements.txt
├── pytest.ini
└── setup.py
```

---

## 1. Local setup

### Prerequisites
- Python 3.10+
- pip
- Docker (for containerized deployment)
- Jenkins (for CI/CD)

### Install dependencies

```bash
git clone https://github.com/badrhr/MLOpsFull
cd MLOpsFull

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                   # makes madewithml importable
```

---

## 2. Run the pipeline locally

### Train
```bash
python madewithml/train.py \
    --experiment-name "baseline" \
    --dataset-loc data/projects.csv \
    --num-epochs 10
```

### Evaluate
```bash
python madewithml/evaluate.py \
    --experiment-name "baseline" \
    --dataset-loc data/projects.csv
# writes metrics.json
```

### Full pipeline in one command
```bash
python madewithml/main.py \
    --experiment-name "baseline" \
    --dataset-loc data/projects.csv
```

### View MLflow dashboard
```bash
mlflow ui --backend-store-uri file:///tmp/mlflow
# open http://localhost:5000
```

### Predict on a single input
```bash
python madewithml/predict.py \
    --text "Attention mechanism for NLP classification"
```

---

## 3. Run tests

```bash
pytest tests/ -v
```

Test results are written to `tests/results/results.xml` (used by Jenkins).

---

## 4. Run the API locally

```bash
uvicorn madewithml.serve:app --reload --port 8000
```

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Graph neural networks for molecule classification"}'
```

---

## 5. Docker

### Build
```bash
docker build -t mlops-app:latest .
```

### Run
```bash
docker run -d --name mlops-app --network host -v /tmp/eval_artifacts:/tmp/eval_artifacts mlops-app:latest

---

## 6. CI/CD with Jenkins

### Prerequisites on the Jenkins server
1. Install Jenkins: https://www.jenkins.io/doc/book/installing/
2. Install plugins: **GitHub**, **Pipeline**, **JUnit**
3. Install Docker on the Jenkins agent
4. Add your GitHub credentials in Jenkins → Manage Credentials

### Setup steps

1. **Create a Pipeline job** in Jenkins
2. Set **SCM** to your GitHub repo URL
3. Set **Script path** to `Jenkinsfile`
4. Enable **GitHub hook trigger for GITScm polling**

### What happens on pull request (CI)
```
Setup → Lint → Test → Train → Evaluate → Performance Gate
```
The PR is blocked from merging if F1 score < 0.70.

### What happens on merge to main (CD)
```
Build Docker Image → Deploy → Health Check
```
The new model is automatically deployed to production.

### Trigger a CI run manually
```bash
# Create a branch, make a change, open a PR
git checkout -b feature/new-experiment
# ... make changes ...
git push origin feature/new-experiment
# Open PR on GitHub → Jenkins triggers automatically
```

---

## 7. Monitoring

### Generate a current production window (example)
```bash
# Simulate production data by sampling from the dataset
python - <<EOF
import pandas as pd
df = pd.read_csv("data/projects.csv").sample(200, random_state=99)
df.to_csv("data/current_window.csv", index=False)
EOF
```

### Run drift detection
```bash
python madewithml/monitoring.py \
    --reference-loc data/projects.csv \
    --current-loc   data/current_window.csv \
    --report-dir    reports/
```

Open the generated HTML report in `reports/` to visualize drift.

### Automate monitoring in Jenkins
Add this stage to your Jenkinsfile after deploy:

```groovy
stage('Monitor') {
    steps {
        sh '''
            . ${VENV_DIR}/bin/activate
            python madewithml/monitoring.py \
                --reference-loc data/projects.csv \
                --current-loc   data/current_window.csv \
                --report-dir    reports/
        '''
    }
}
```

---

## 8. Logging

Logs are written to two rotating files:

| File | Content |
|---|---|
| `logs/info.log` | All INFO+ messages with timestamp and location |
| `logs/error.log` | ERROR and CRITICAL messages only |

The console shows minimal output. The files show full detail:

```
INFO 2024-01-15 10:23:01,452 [train.py:train_model:95]
Training epoch 1/10 ...
```

---

## Key concepts implemented

| Topic | Implementation |
|---|---|
| Scripts (not notebooks) | `madewithml/*.py` |
| Logging | `config.py` → `logs/info.log`, `logs/error.log` |
| Reproducibility | `set_seeds()`, MLflow run tracking, Git versioning |
| CI/CD | `Jenkinsfile` with PR gate + auto-deploy |
| Monitoring | `monitoring.py` with EvidentlyAI drift reports |

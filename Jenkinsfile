pipeline {
    agent any

    environment {
        PYTHON     = 'python3'
        VENV_DIR   = '.venv'
        MODEL_DIR  = "${WORKSPACE}/mlflow"
        APP_PORT   = '8000'
        IMAGE_NAME = 'mlops-app'
    }

    // ─────────────────────────────────────────────
    // CI  — triggered on every Pull Request to main
    // ─────────────────────────────────────────────
    triggers {
        githubPullRequests(
            spec: '',
            triggerMode: 'HEAVY_HOOKS'
        )
    }

    stages {

        // ── 1. SETUP ──────────────────────────────
        stage('Setup') {
            steps {
                echo '── Setting up Python environment ──'
                sh '''
                    ${PYTHON} -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install -e .
                '''
            }
        }

        // ── 2. LINT ───────────────────────────────
        stage('Lint') {
            steps {
                echo '── Linting with flake8 ──'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    flake8 madewithml/ \
                        --max-line-length=99 \
                        --ignore=E501,W503,E221,E241,E226,E402
                '''
            }
        }

        // ── 3. TESTS ──────────────────────────────
        stage('Test') {
            steps {
                echo '── Running pytest ──'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pytest tests/ -v --tb=short
                '''
            }
            post {
                always {
                    // Publish test results in Jenkins UI
                    junit 'tests/results/*.xml'
                }
            }
        }

        // ── 4. TRAIN ──────────────────────────────
        stage('Train') {
            steps {
                echo '── Training model ──'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    python madewithml/train.py \
                        --experiment-name "ci_run_${BUILD_NUMBER}" \
                        --dataset-loc data/dataset.csv \
                        --num-epochs 10
                '''
            }
        }

        // ── 5. EVALUATE & COMPARE ─────────────────
        // Compare new run against the current production
        // model stored in MLflow. Fail the PR if the new
        // model is worse.
        stage('Evaluate') {
            steps {
                echo '── Evaluating and comparing to production ──'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    python madewithml/evaluate.py \
                        --experiment-name "ci_run_${BUILD_NUMBER}" \
                        --dataset-loc data/dataset.csv
                '''
            }
        }

        // ── 6. PERFORMANCE GATE ───────────────────
        // Read the F1 score written by evaluate.py and
        // block the merge if it falls below the threshold.
        stage('Performance Gate') {
            steps {
                echo '── Checking performance threshold ──'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    python - <<EOF
import json, sys

with open("metrics.json") as f:
    metrics = json.load(f)

threshold = 0.70
f1 = metrics.get("f1", 0)
print(f"F1 score: {f1:.4f}  (threshold: {threshold})")

if f1 < threshold:
    print("FAILED: model did not meet the performance threshold.")
    sys.exit(1)

print("PASSED: model meets threshold. Safe to merge.")
EOF
                '''
            }
        }

        // ── 7. CD ──────────────────────────────────
        stage('Build Docker Image') {
            steps {
                echo '── Building Docker image ──'
                sh '''
                    docker build --network host -t ${IMAGE_NAME}:${BUILD_NUMBER} .
                    docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Deploy') {
            steps {
                echo '── Deploying to production ──'
                sh '''
                    docker stop ${IMAGE_NAME} || true
                    docker rm   ${IMAGE_NAME} || true
                    docker run -d \
                        --network host \
                        --name ${IMAGE_NAME} \
                        -e MODEL_DIR=${MODEL_DIR} \
                        -v $(pwd)/eval_artifacts:/tmp/eval_artifacts \
                        ${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Health Check') {
            steps {
                echo '── Verifying deployment ──'
                sh '''
                    for i in $(seq 1 10); do
                        curl --fail http://localhost:${APP_PORT}/health && exit 0
                        sleep 3
                    done
                    echo "Health check failed after 30s" && exit 1
                '''
            }
        }

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
    }

    // ─────────────────────────────────────────────
    // NOTIFICATIONS
    // ─────────────────────────────────────────────
    post {
        success {
            echo "Pipeline passed — safe to merge PR #${env.CHANGE_ID}"
        }
        failure {
            echo "Pipeline FAILED on branch ${env.BRANCH_NAME} — do not merge"
        }
        always {
            // Clean up virtual env to keep workspace tidy
            sh 'rm -rf ${VENV_DIR}'
        }
    }
}

pipeline {
    agent any

    environment {
        PYTHON     = 'python3'
        VENV_DIR   = '.venv'
        MODEL_DIR  = '/tmp/mlflow'
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
                        --dataset-loc data/projects.csv \
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
                        --dataset-loc data/projects.csv
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

        // ── 7. CD — only runs after merge to main ─
        stage('Build Docker Image') {
            when {
                branch 'main'
            }
            steps {
                echo '── Building Docker image ──'
                sh '''
                    docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .
                    docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo '── Deploying to production ──'
                sh '''
                    docker stop ${IMAGE_NAME} || true
                    docker rm   ${IMAGE_NAME} || true
                    docker run -d \
                        --name ${IMAGE_NAME} \
                        -p ${APP_PORT}:8000 \
                        -e MODEL_DIR=${MODEL_DIR} \
                        ${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Health Check') {
            when {
                branch 'main'
            }
            steps {
                echo '── Verifying deployment ──'
                sh '''
                    sleep 5
                    curl --fail http://localhost:${APP_PORT}/health \
                        || (echo "Health check failed" && exit 1)
                '''
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

# ITCS355 Lab 1
# `make reproduce` is the one command a grader runs. Keep it working.

SHELL := /bin/bash
IMAGE ?= itcs355-lab1
TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
PLATFORM ?= linux/amd64
SEED ?= 20260101

# For Lab2 at "make reload-check"
MODEL_REGISTRY_NAME ?= itcs355-6688166
VERSION ?= 2

.PHONY: help setup cloud-check data test portability-audit train image image-push reproduce verify clean teardown \
        tune compare reload-check serve serve-image loadtest drift inject-drift pipeline cost swap-check llm-eval llm-gate

help:
	@grep -E "^[a-zA-Z_-]+:.*?## .*$$" $(MAKEFILE_LIST) | awk -F":.*?## " "{printf \"  %-20s %s\\n\", \$$1, \$$2}"

setup: ## Install dependencies and print environment status
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "environment ok"

cloud-check: ## Resolve the eight capability slots
	python scripts/cloud_check.py

data: ## Generate the default dataset (deterministic)
	python scripts/make_dataset.py --seed $(SEED)

test: ## Run data contract and split property tests
	pytest -q tests/

portability-audit: ## Fail if provider strings leak into src/
	python scripts/portability_audit.py

train: ## Train locally, outside the container
	python -m src.train --seed $(SEED) --metrics-out reports/metrics.json

image: ## Build the training image for linux/amd64
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) --load .

image-push: image ## Push to CONTAINER_REGISTRY via your adapter
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	print(get_adapter(config.load()).push_image(\"$(IMAGE):$(TAG)\"))"

reproduce: data image ## THE ONE COMMAND. Grader runs this.
	@mkdir -p reports && chmod 777 reports
	docker run --rm \
	  -v "$$PWD/data:/app/data:ro" \
	  -v "$$PWD/reports:/app/reports" \
	  -e MLFLOW_TRACKING_URI=sqlite:////app/reports/mlflow.db \
	  $(IMAGE):$(TAG) --seed $(SEED) --metrics-out /app/reports/metrics.json

verify: ## Check the produced metric against the README claim
	python scripts/verify_metric.py

teardown: ## Delete every resource tagged course=itcs355 for this lab
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	cfg=config.load(); print(get_adapter(cfg).teardown(cfg.tags(1)))"

clean: ## Remove local artifacts
	rm -rf mlruns mlartifacts mlflow.db reports/metrics.json .pytest_cache

# --- Lab 2 -------------------------------------------------------------------
tune: ## Budgeted hyperparameter study (>=12 trials)
# Adding --instance Standard_DS3_v2 for azure (lab2task)
	python -m src.tune --trials 12 --budget-thb 150 --instance Standard_DS3_v2

compare: ## Rank runs by metric and by cost per point
	python scripts/compare_runs.py --experiment itcs355-lab2

reload-check: ## Load the registered model by version and score rows
	python scripts/reload_check.py --name $(MODEL_REGISTRY_NAME) --version $(VERSION)

#Created on Lab02 Task01
train-remote: ## Submit training to Azure ML and wait for completion
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	cfg = config.load(); adapter = get_adapter(cfg); \
	job_id = adapter.submit_training(f'{cfg.container_registry}:$(TAG)', {'seed': $(SEED)}); \
	print(f'Submitted training job: {job_id}'); \
	res = adapter.wait_training(job_id); \
	print(f'Job result: {res}')"

# --- Lab 3 -------------------------------------------------------------------
serve: ## Run the inference service locally on :8080
	python scripts/export_model.py --out reports/model.joblib
	MODEL_PATH=reports/model.joblib MODEL_VERSION=local uvicorn service.app:app --port 8080

serve-image: ## Build the serving image
	docker buildx build --platform $(PLATFORM) -f service/Dockerfile.serve -t itcs355-serve:$(TAG) --load .

loadtest: ## Load test at three concurrency levels
	@for vus in 1 10 50; do \
	  echo "=== $$vus VUs ==="; \
	  k6 run -e TARGET=$(TARGET) -e VUS=$$vus loadtest/k6.js || true; \
	done

deploy: ## Deploy the inference service to Azure Container Apps
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	adapter = get_adapter(config.load()); \
	url = adapter.deploy('$(VERSION)', 'itcs355-predict', 'Standard_B1s'); \
	print(f'Endpoint live at: {url}')"

smoke: ## Smoke test the live endpoint with three known payloads
	python -c "\
	from src import config; from cloudlayer.factory import get_adapter; \
	adapter = get_adapter(config.load()); \
	samples = [ \
	  {'temp_c': 78.4, 'vibration_mm_s': 3.1, 'pressure_kpa': 315.2, 'hours_since_service': 4200.0, 'load_pct': 68.0, 'ambient_humidity': 55.0}, \
	  {'temp_c': 95.0, 'vibration_mm_s': 7.5, 'pressure_kpa': 450.0, 'hours_since_service': 8500.0, 'load_pct': 92.0, 'ambient_humidity': 40.0}, \
	  {'temp_c': 65.0, 'vibration_mm_s': 1.2, 'pressure_kpa': 290.0, 'hours_since_service': 500.0, 'load_pct': 30.0, 'ambient_humidity': 60.0}, \
	]; \
	[print(f'Payload {i}: prob={r[\"probability\"]:.4f}, version={r[\"model_version\"]}') for i, s in enumerate(samples, 1) for r in [adapter.invoke('itcs355-predict', s)]]"

# --- Lab 4 -------------------------------------------------------------------
inject-drift: ## Shift a feature's distribution on purpose
	python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6

drift: ## Score drift against the reference window
	python -m monitoring.drift --current data/current.csv

# --- Lab 5 -------------------------------------------------------------------
pipeline: ## Compile pipeline/pipeline.yaml for your provider
	python -c "from cloudlayer.pipelines import compile_for; from src import config; \
	compile_for(config.load().provider)"

llm-eval: ## Run the LLM golden set against recorded responses (offline, free)
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json

llm-gate: ## Prove the gate fails on a degraded set — expected to exit non-zero
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json >/dev/null
	python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl \
	  --out reports/llm_eval.json --baseline reports/llm_eval-baseline.json

cost: ## Build the cost report
	python scripts/cost_report.py --estimate $(EST) --actual $(ACT) --rps $(RPS) --instance $(INSTANCE)

swap-check: ## Prove the portability seam against a second provider
	python scripts/portability_swap_check.py --second-provider $(SECOND)

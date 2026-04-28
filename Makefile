# HAES Makefile
# Reproducibility target: re-runs all six XD-Violence phases under default seed
# and emits a SHA256-verified results CSV.

.PHONY: help install reproduce_all reproduce_xd reproduce_ucf reproduce_shanghaitech test clean docker-build

# Default settings
SEED ?= 42
CONFIG ?= configs/default.yaml
DATA_DIR ?= ./data
OUTPUT_DIR ?= ./output

help: ## Show this help message
	@echo "HAES - Hierarchical Adaptive Expert System"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

install: ## Install conda environment
	conda env create -f environment.yaml
	@echo "Environment 'haes' created. Activate with: conda activate haes"

install-pip: ## Install with pip only
	pip install -r requirements.txt

download-xd: ## Download XD-Violence dataset
	python -c "from data.download import download_xd_violence; download_xd_violence('$(DATA_DIR)/XD-Violence')"

download-ucf: ## Download UCF-Crime dataset
	python -c "from data.download import download_ucf_crime; download_ucf_crime('$(DATA_DIR)/UCF-Crime')"

download-shanghai: ## Download ShanghaiTech dataset
	python -c "from data.download import download_shanghaitech; download_shanghaitech('$(DATA_DIR)/ShanghaiTech')"

reproduce_all: reproduce_xd reproduce_ucf ## Re-run all experiments (XD-Violence + UCF-Crime)

reproduce_xd: ## Reproduce XD-Violence results (6 phases, default seed=42)
	@echo "=== Reproducing XD-Violence Results (6 Phases) ==="
	@echo "GPU: Tesla A100 40GB | CUDA: 12.1 | Driver: 535.104.05"
	@echo "Seed: $(SEED) | Config: $(CONFIG)"
	python main_train.py \
		--dataset xd_violence \
		--config $(CONFIG) \
		--data_dir $(DATA_DIR) \
		--output_dir $(OUTPUT_DIR)/xd_violence \
		--seed $(SEED)
	@echo "=== XD-Violence SHA256 Verification ==="
	sha256sum $(OUTPUT_DIR)/xd_violence/results_xd_violence.csv

reproduce_ucf: ## Reproduce UCF-Crime results (13 phases, default seed=42)
	@echo "=== Reproducing UCF-Crime Results (13 Phases) ==="
	@echo "GPU: Tesla A100 40GB | CUDA: 12.1 | Driver: 535.104.05"
	@echo "Seed: $(SEED) | Config: $(CONFIG)"
	python main_train.py \
		--dataset ucf_crime \
		--config $(CONFIG) \
		--data_dir $(DATA_DIR) \
		--output_dir $(OUTPUT_DIR)/ucf_crime \
		--seed $(SEED)
	@echo "=== UCF-Crime SHA256 Verification ==="
	sha256sum $(OUTPUT_DIR)/ucf_crime/results_ucf_crime.csv

reproduce_shanghaitech: ## Reproduce ShanghaiTech results
	python main_train.py \
		--dataset shanghaitech \
		--config $(CONFIG) \
		--data_dir $(DATA_DIR) \
		--output_dir $(OUTPUT_DIR)/shanghaitech \
		--seed $(SEED)

evaluate: ## Evaluate a trained checkpoint
	python main_test.py \
		--checkpoint $(CHECKPOINT) \
		--dataset $(DATASET) \
		--config $(CONFIG) \
		--benchmark \
		--drift_test \
		--noise_test

test: ## Run unit tests
	python -m pytest tests/ -v

clean: ## Clean output directories
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(DATA_DIR)/features/*
	@echo "Cleaned outputs and feature caches."

docker-build: ## Build Docker image
	docker build -t haes:latest .
	@echo "Docker image built: haes:latest"
	@docker images haes:latest --format "Image SHA256: {{.ID}}"

docker-run: ## Run training in Docker container
	docker run --gpus all \
		-v $(PWD)/data:/workspace/HAES/data \
		-v $(PWD)/output:/workspace/HAES/output \
		haes:latest --dataset xd_violence --config configs/default.yaml

docker-reproduce: ## Reproduce all results in Docker
	docker run --gpus all \
		-v $(PWD)/data:/workspace/HAES/data \
		-v $(PWD)/output:/workspace/HAES/output \
		haes:latest make reproduce_all

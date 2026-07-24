# Convenience targets for the course project stages

.PHONY: setup test smoke clean-results validate

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e .
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest -q

smoke:
	.venv/bin/python scripts/run_experiment.py --config configs/smoke.yaml --synthetic --until report --skip-finetune
	.venv/bin/python scripts/00_validate_pipeline.py --config configs/smoke.yaml --stages clean distorted enhanced

validate:
	.venv/bin/python scripts/00_validate_pipeline.py

clean-results:
	rm -rf results/clean results/distorted results/enhanced results/finetuned results/figures
	mkdir -p results/figures

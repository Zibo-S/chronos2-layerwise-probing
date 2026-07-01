# Layer-wise probing of frozen Chronos-2 — entry points.
# Run targets from the repository root. Uses the active Python (activate .venv first).
PYTHON ?= python

.PHONY: help setup smoke perdataset pipeline improve harden verify audit regen-cache clean-results notebook

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install pinned dependencies
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

smoke:  ## Fast behavior-preservation check (reads cache + committed JSON, no model)
	$(PYTHON) tests/test_smoke.py

perdataset:  ## Main experiment: per-dataset ID vs OOD grids + results/perdataset_summary.json
	$(PYTHON) -m experiments.run_perdataset

pipeline:  ## Phase B-D demo (Epilepsy ID, synthetic shift, transfer)
	$(PYTHON) -m experiments.run_pipeline

improve:  ## SCP1-primary run with bootstrap CIs
	$(PYTHON) -m experiments.run_improve

harden:  ## Five hardening tests -> results/probe_harden_artifacts.json
	$(PYTHON) -m experiments.run_harden

verify:  ## Read-only: verify UEA dataset shape facts via aeon (no model)
	$(PYTHON) tools/verify_dataset_facts.py

audit:  ## Read-only: re-derive conclusions from results/perdataset_summary.json
	$(PYTHON) tools/audit_local.py

regen-cache:  ## Rebuild ALL cached features from scratch (SLOW; needs the model + MPS/GPU)
	@echo "This re-extracts every (dataset, split, corruption) into features_cache/ (~13 GB)."
	@echo "It downloads amazon/chronos-2 and runs the encoder over all datasets."
	rm -rf features_cache
	$(PYTHON) -m experiments.run_perdataset   # extracts clean + gauss/timewarp/drift for all 8 datasets
	$(PYTHON) -m experiments.run_harden       # extracts the gaussian alpha sweep used by hardening tests

clean-results:  ## Remove regenerated figures/JSON in results/ (git can restore them)
	rm -f results/*.png results/*.json

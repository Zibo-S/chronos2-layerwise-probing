# Layer-wise probing of frozen Chronos-2 — entry points.
# Run targets from the repository root. Activate the venv first (on Narval, see the
# environment recipe in README before activating).
PYTHON ?= python

.PHONY: help setup forecasting bootstrap uea clean-results

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies (laptop; on Narval use module + --no-index, see README)
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

forecasting:  ## Run the ID forecasting probes -> results/$$ID_DATASET_SET/ (default extended_v1)
	$(PYTHON) -m experiments.run_id_forecasting

bootstrap:  ## Post-hoc series-level cluster bootstrap (CPU) -> results/$$ID_DATASET_SET/bootstrap/
	$(PYTHON) -m experiments.run_bootstrap

uea:  ## UEA classification baseline (maintained baseline) -> results/uea/
	$(PYTHON) -m experiments.run_perdataset

clean-results:  ## Remove regenerated ID forecasting outputs for the active set (git can restore them)
	rm -rf results/$${ID_DATASET_SET:-extended_v1}/

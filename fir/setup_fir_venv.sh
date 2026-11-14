#!/bin/bash
# =======================================================================================
# FIR: build the Python environment of the Narval rebuttal jobs, package for package.
# LOGIN node (pip installs; Fir compute nodes have no internet). Minutes, mostly I/O.
#
#   bash fir/setup_fir_venv.sh            # creates /scratch/$USER/venvs/c2lp-py311
#
# Source of truth: fir/narval_c2lp-py311_pip_freeze.txt = `pip freeze --all` of the Narval venv
# ($SCRATCH/venvs/c2lp-py311, Python 3.11.5, StdEnv/2023) that the Narval jobs use.
#   * every "+computecanada" package comes from the Alliance wheelhouse (CVMFS, the same files on
#     every national cluster) at the SAME version: pip install --no-index;
#   * the 6 others were installed from PyPI on Narval; they are installed here from PyPI at the
#     SAME version with --no-deps (their dependencies are already pinned by the freeze).
# The venv is built fresh (never copied): compiled wheels must match this cluster's stack.
# =======================================================================================
set -euo pipefail
VENV=${VENV:-/scratch/$USER/venvs/c2lp-py311}
FREEZE=fir/narval_c2lp-py311_pip_freeze.txt
[ -f "$FREEZE" ] || { echo "run from the repo root"; exit 2; }
export PIP_CACHE_DIR=${SCRATCH:?}/.cache/pip XDG_CACHE_HOME=$SCRATCH/.cache
module load gcc python/3.11 arrow/24.0.0
python -V
[ -x "$VENV/bin/python" ] || python -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

WH=$(mktemp); PYPI=$(mktemp)
grep -v -E '^(pip|setuptools)==' "$FREEZE" | grep '+computecanada' | sed 's/+computecanada//' > "$WH"
grep -v -E '^(pip|setuptools)==' "$FREEZE" | grep -v '+computecanada' > "$PYPI"
echo "wheelhouse: $(wc -l < "$WH") pins; PyPI: $(tr '\n' ' ' < "$PYPI")"
pip install --no-index -r "$WH"
# a "PyPI" pin that the wheelhouse also carries is taken from the wheelhouse; the rest from PyPI
while read -r req; do
    pip install --no-index --no-deps "$req" 2>/dev/null || pip install --no-deps "$req"
done < "$PYPI"
rm -f "$WH" "$PYPI"

echo "--- verify: pip freeze vs Narval (version-only differences listed)"
diff <(sed 's/+computecanada//' "$FREEZE" | sort -f) <(pip freeze --all | sed 's/+computecanada//' | sort -f) \
    && echo "IDENTICAL to the Narval freeze" || echo "^ differences (must be empty for a faithful mirror)"
pip check || true
OMP_NUM_THREADS=1 python - <<'EOF'
import importlib
for m in ("numpy", "torch", "transformers", "chronos", "timesfm", "tirex", "aeon", "pyarrow",
          "datasets", "sklearn", "scipy", "matplotlib"):
    mod = importlib.import_module(m)
    print(f"  import {m:<13} {getattr(mod, '__version__', '?')}")
EOF

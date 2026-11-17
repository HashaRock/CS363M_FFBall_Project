#!/usr/bin/env bash
# Run the notebook top to bottom in a fresh kernel, then run the tests.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

NOTEBOOK="fantasy_football_rankings.ipynb"

echo "== Executing $NOTEBOOK in a fresh kernel =="
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 "$NOTEBOOK"

echo "== Running tests =="
pytest -q tests

echo "== verify.sh passed =="

#!/usr/bin/env bash
# Regenerate every table and figure in the report from the seeded simulation, and check the numbers against
# paper/results.sha256. Needs the `paper` extra (matplotlib) for the figures.
set -euo pipefail
cd "$(dirname "$0")/../paper"
python make_data.py > /dev/null
python check_results.py
python figures.py
echo "tables in paper/tables, figures in paper/figures, results in paper/data/results.json"

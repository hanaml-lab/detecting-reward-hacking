#!/usr/bin/env bash
# Regenerate every table and figure in the paper from the seeded simulation.
set -euo pipefail
cd "$(dirname "$0")/../paper"
python make_data.py
python figures.py
echo "tables in paper/tables/, figures in paper/figures/"

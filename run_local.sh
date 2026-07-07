#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export TASKS_INPUT_PATH="$(pwd)/test_input/tasks.json"
export RESULTS_OUTPUT_PATH="$(pwd)/test_output/results.json"

python3 src/main.py

echo "---"
echo "Results written to test_output/results.json:"
cat test_output/results.json

#!/usr/bin/env bash
set -euo pipefail

# Runs the golden-dataset chatbot tests end-to-end.
#
# Requires:
#   - Python 3.10+
#   - OPENAI_API_KEY exported in your environment
#
# Usage:
#   ./examples/run_golden_tests.sh

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set. Export it first, e.g.:" >&2
  echo "  export OPENAI_API_KEY=sk-..." >&2
  exit 1
fi

pip install -r requirements.txt
pytest tests/test_chatbot_responses.py -v

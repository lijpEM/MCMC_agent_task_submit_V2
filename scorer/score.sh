#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="${1:-/home/workspace/pseudo2d_mt_lab/outputs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate.py" "${SUBMISSION_DIR}"

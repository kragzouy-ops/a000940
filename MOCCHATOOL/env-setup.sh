#!/usr/bin/env bash
# env-setup.sh: Overhead environment switch and prerequisites installation for MOCHAbin Tool
# Usage: source ./env-setup.sh [python_version]

set -e

PYTHON_VERSION="${1:-3.10}"
VENV_DIR=".venv"

# Check for python version
if ! command -v python$PYTHON_VERSION &>/dev/null; then
  echo "[ERROR] python$PYTHON_VERSION not found. Please install Python $PYTHON_VERSION."
  return 1 2>/dev/null || exit 1
fi

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment with python$PYTHON_VERSION..."
  python$PYTHON_VERSION -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
python -m pip install --upgrade pip

# Install prerequisites
python -m pip install typer[all] pyserial pexpect rich

# Make main tool executable
chmod +x mochabin_tool.py

echo "[INFO] Environment ready. To activate later: source $VENV_DIR/bin/activate"

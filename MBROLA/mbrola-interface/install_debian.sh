#!/usr/bin/env bash
# Debian/Ubuntu quick installer for mbrola-interface (local source install)
# - Installs system deps (mbrola, espeak-ng, default voices en1+fr1)
# - Creates Python venv and installs this package
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "[1/3] Installing system packages via apt..."
$SUDO apt-get update -y
# Try to install both English and French MBROLA voices (ignore failures if not in repos)
$SUDO apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip \
  mbrola espeak-ng || true
# Attempt common voice packages (Debian/Ubuntu repos may differ)
$SUDO apt-get install -y --no-install-recommends mbrola-en1 mbrola-fr1 || true

echo "[2/3] Creating virtual environment (.venv) ..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "[3/3] Installing the package from source ..."
pip install .

cat << 'EOF'

Done.

Quick test (English; requires voice 'en1' installed):
  . .venv/bin/activate
  mbrola-interface --voice /usr/share/mbrola/en1/en1 --demo --out demo.wav --play

French example:
  mbrola-interface --voice /usr/share/mbrola/fr1/fr1 --demo --espeak-voice mb-fr1 --out demo_fr.wav --play

If a voice directory differs on your system, adjust the --voice path accordingly.
EOF

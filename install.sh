#!/usr/bin/env bash
# ============================================================
#   Lexis One-Click Setup (Linux / WSL / macOS)
# ------------------------------------------------------------
#   Run `./install.sh` (or `bash install.sh`) in a terminal
#   to install everything automatically:
#     1. Create/activate `.venv` (if not already in a virtualenv/conda env)
#     2. pip install -e .              lexis (editable install)
#        pip install pyinstaller pytest
#     3. Ensure spaCy `en_core_web_sm` model is installed
#     4. Optionally install `faster-whisper` for video/audio transcription
#     5. Run `lexis init .` to initialize workspace & download `pedlex.db`
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Lexis Wiki — One-Click Setup (Linux / WSL / macOS)"
echo "============================================================"

# ---- Resolve base Python 3 interpreter ----
BASE_PY=""
if command -v python3 >/dev/null 2>&1; then
    BASE_PY="python3"
elif command -v python >/dev/null 2>&1; then
    BASE_PY="python"
elif command -v apt-get >/dev/null 2>&1; then
    echo "[Env] Python 3 not found. Installing python3, python3-venv, and python3-pip via apt ..."
    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    BASE_PY="python3"
else
    echo "[ERROR] Python 3 not found. Please install Python 3.9+ first."
    exit 1
fi

# ---- Use active virtualenv/conda if present, otherwise manage `.venv` ----
USED_LOCAL_VENV=0
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
    echo "[Env] Using active virtual environment: $VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PY="$CONDA_PREFIX/bin/python"
    echo "[Env] Using active Conda environment: $CONDA_PREFIX"
else
    # Clean up if a Windows `.venv` (with Scripts/python.exe) or broken `.venv` (missing pip) exists
    if [[ -d ".venv" ]] && { [[ ! -x ".venv/bin/python" ]] || ! ".venv/bin/python" -m pip --version >/dev/null 2>&1; }; then
        echo "[Env] Detected incomplete or non-Linux .venv directory; recreating ..."
        rm -rf .venv
    fi

    if [[ ! -x ".venv/bin/python" ]]; then
        echo "[Env] Creating local virtual environment in .venv ..."
        if ! "$BASE_PY" -m venv .venv; then
            rm -rf .venv
            if command -v apt-get >/dev/null 2>&1; then
                echo "[Env] Installing missing python3-venv & python3-pip packages via apt ..."
                sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
                echo "[Env] Retrying .venv creation ..."
                "$BASE_PY" -m venv .venv
            else
                echo "[ERROR] Failed to create .venv. Please install the python3-venv package for your OS."
                exit 1
            fi
        fi
    else
        echo "[Env] Found existing virtual environment in .venv"
    fi
    PY="$SCRIPT_DIR/.venv/bin/python"
    USED_LOCAL_VENV=1
fi

# ---- 1/3 Install lexis (editable) + dev tools ----
echo ""
echo "[1/3] Installing lexis and dev tools (pyinstaller, pytest) ..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -e .
"$PY" -m pip install pyinstaller pytest

# ---- Ensure spaCy English model is ready ----
if ! "$PY" -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
    echo "      Downloading spaCy model en_core_web_sm ..."
    "$PY" -m spacy download en_core_web_sm
fi

# ---- Optional: local video/audio transcription (faster-whisper) ----
echo ""
EXTRA="n"
if [[ -t 0 ]]; then
    read -r -p "Install local video transcription (faster-whisper — large download)? [y/N] " EXTRA || EXTRA="n"
fi
if [[ "${EXTRA,,}" == "y" || "${EXTRA,,}" == "yes" ]]; then
    "$PY" -m pip install faster-whisper
    echo "  faster-whisper installed."
else
    echo "  Skipped. You can add it later with:  $PY -m pip install faster-whisper"
fi

# ---- 2/3 Initialize project & download/verify PedLex lexicon (pedlex.db) ----
echo ""
echo "[2/3] Initializing project and syncing PedLex lexicon (lexis init .) ..."
"$PY" -m librarian.cli init .

# ---- 3/3 Verify PedLex database status ----
echo ""
echo "[3/3] Verifying PedLex database ..."
"$PY" -m librarian.cli pedlex

echo ""
echo "============================================================"
echo "  Setup complete!"
if [[ "$USED_LOCAL_VENV" -eq 1 ]]; then
    echo ""
    echo "  Activate the virtual environment first in new terminals:"
    echo "    source .venv/bin/activate"
fi
echo ""
echo "  You can now run:"
echo "    lexis dashboard              # Launch the Web Dashboard"
echo "    lexis compile Rosetta_Stone  # Compile a sample unit"
echo "============================================================"

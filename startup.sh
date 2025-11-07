#!/bin/bash

# Azure App Service Startup Script for LiveKit Agent
# Fixes typing_extensions conflict by ensuring venv packages take precedence

echo "========================================="
echo "🚀 Starting LiveKit Voice Agent"
echo "========================================="

# Find virtual environment (Azure creates it in different locations)
# Method 1: Check common locations
VENV_PATH=""
if [ -d "/home/site/wwwroot/antenv" ]; then
    VENV_PATH="/home/site/wwwroot/antenv"
elif [ -d "/tmp/8de1dd7a72e565e/antenv" ]; then
    VENV_PATH="/tmp/8de1dd7a72e565e/antenv"
else
    # Method 2: Find it dynamically using find command
    echo "🔍 Searching for virtual environment..."
    VENV_PATH=$(find /tmp -maxdepth 2 -type d -name "antenv" 2>/dev/null | head -1)
fi

if [ -z "$VENV_PATH" ] || [ ! -d "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found!"
    echo "Checked locations:"
    echo "  - /home/site/wwwroot/antenv"
    echo "  - /tmp/*/antenv"
    exit 1
fi

echo "📦 Virtual environment: $VENV_PATH"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "❌ ERROR: Cannot activate virtual environment at $VENV_PATH/bin/activate"
    exit 1
fi

# CRITICAL FIX: Override PYTHONPATH to use venv FIRST
# Since we can't delete Azure's old typing_extensions, we override the import path
echo "🔧 Fixing typing_extensions conflict..."
echo "   Setting PYTHONPATH to prioritize venv packages..."

# Put venv site-packages FIRST, before /agents/python
export PYTHONPATH="$VENV_PATH/lib/python3.12/site-packages:${PYTHONPATH}"
echo "   ✅ PYTHONPATH updated (venv now has priority)"

# Upgrade typing_extensions in the venv
echo "📥 Installing typing_extensions>=4.8.0 in venv..."
pip install --upgrade --force-reinstall typing-extensions>=4.8.0 --quiet --no-warn-script-location 2>/dev/null
echo "✅ typing_extensions installation complete"

# Verify Sentinel import works
echo "🔍 Verifying Sentinel import..."
python -c "from typing_extensions import Sentinel; print('   ✅ Sentinel imported successfully!')" 2>&1 || {
    echo "   ❌ ERROR: Cannot import Sentinel from typing_extensions"
    echo "   Attempting to diagnose..."
    python -c "import typing_extensions; print(f'   typing_extensions location: {typing_extensions.__file__}')"
    python -c "import sys; print('   PYTHONPATH:', sys.path[:3])"
    exit 1
}

# Find app directory
APP_DIR="/home/site/wwwroot"
if [ ! -f "$APP_DIR/app.py" ]; then
    # Try to find it in tmp
    for dir in /tmp/*/; do
        if [ -f "${dir}app.py" ]; then
            APP_DIR="$dir"
            break
        fi
    done
fi

echo "📁 App directory: $APP_DIR"
cd "$APP_DIR" || exit 1

# Verify app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ ERROR: app.py not found in $APP_DIR"
    exit 1
fi

echo "========================================="
echo "▶️  Starting application: python app.py api"
echo "========================================="

# Start the application
exec python app.py api


#!/bin/bash

# Azure App Service Startup Script for LiveKit Agent
# Fixes typing_extensions conflict by ensuring venv packages take precedence

set -e  # Exit on error

echo "========================================="
echo "🚀 Starting LiveKit Voice Agent"
echo "========================================="

# Find virtual environment (Azure creates it in different locations)
VENV_PATH=""
for path in "/home/site/wwwroot/antenv" "/tmp/*/antenv"; do
    if [ -d "$path" ]; then
        VENV_PATH=$(echo $path)
        break
    fi
done

if [ -z "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found!"
    exit 1
fi

echo "📦 Virtual environment: $VENV_PATH"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "❌ ERROR: Cannot activate virtual environment"
    exit 1
fi

# CRITICAL FIX: Remove Azure's old typing_extensions from import path
# This forces Python to use the venv's typing_extensions
if [ -f "/agents/python/typing_extensions.py" ]; then
    echo "🔧 Backing up and removing Azure's old typing_extensions..."
    mv /agents/python/typing_extensions.py /agents/python/typing_extensions.py.bak 2>/dev/null || true
    echo "✅ Old typing_extensions removed from path"
fi

# Upgrade typing_extensions in the venv
echo "📥 Upgrading typing_extensions in venv..."
pip install --upgrade --force-reinstall typing-extensions>=4.8.0 --quiet --no-warn-script-location
echo "✅ typing_extensions upgraded"

# Verify installation
echo "🔍 Verifying typing_extensions..."
python -c "import typing_extensions; print(f'   Version: {typing_extensions.__version__}'); from typing_extensions import Sentinel; print('   Sentinel import: OK')" || {
    echo "❌ ERROR: typing_extensions verification failed"
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


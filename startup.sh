#!/bin/bash

echo "========================================"
echo "🚀 Starting LiveKit Voice Agent"
echo "========================================"

# Activate virtual environment
if [ -d "antenv" ]; then
    echo " Using existing virtual environment: antenv"
    source antenv/bin/activate
elif [ -d "/tmp/8de2831f2d38e9e/antenv" ]; then
    echo " Using Azure virtual environment"
    source /tmp/8de2831f2d38e9e/antenv/bin/activate
else
    echo " No virtual environment found!"
    exit 1
fi

# Start the application in DEFAULT API MODE
echo "  Starting application in DEFAULT API MODE"
python app.py start
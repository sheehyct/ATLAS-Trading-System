#!/bin/bash

# ATLAS Dashboard Launcher Script
# Quick start script for running the ATLAS Trading Dashboard

set -e

echo "======================================"
echo "ATLAS Trading Dashboard Launcher"
echo "======================================"
echo ""

# Check if in correct directory
if [ ! -f "dashboard/app.py" ]; then
    echo "Error: Must run from ATLAS project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found"
    echo "Creating template .env file..."
    cp .env.template .env
    echo "Please edit .env with your Alpaca API credentials"
    echo ""
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if UV is available
if command -v uv &> /dev/null; then
    echo "Using UV package manager"
    echo ""
    echo "Installing/updating dependencies..."
    uv sync
    echo ""
    echo "Starting ATLAS Dashboard..."
    echo "Access at: http://localhost:8050"
    echo "Press Ctrl+C to stop"
    echo ""
    uv run python dashboard/app.py
else
    echo "UV not found, using standard Python"
    echo ""
    echo "Installing/updating dependencies..."
    pip install -e .
    echo ""
    echo "Starting ATLAS Dashboard..."
    echo "Access at: http://localhost:8050"
    echo "Press Ctrl+C to stop"
    echo ""
    python dashboard/app.py
fi

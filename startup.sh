#!/bin/bash

# Exit on error
set -e

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Install system dependencies
log_message "Installing system dependencies..."
if ! command -v ffmpeg &> /dev/null; then
    log_message "Installing ffmpeg..."
    apt-get update -y
    apt-get install -y --no-install-recommends ffmpeg libsndfile1
    apt-get clean
    rm -rf /var/lib/apt/lists/*
fi

# Create and activate virtual environment
log_message "Setting up Python virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
log_message "Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
log_message "Installing Python requirements..."
pip install -r requirements.txt

# Set environment variables
export PORT=${PORT:-8000}
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start Streamlit
log_message "Starting Streamlit application..."
exec streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.baseUrlPath "/" \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.serverAddress "0.0.0.0" \
    --browser.gatherUsageStats false

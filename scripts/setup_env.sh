#!/usr/bin/env bash
set -e

echo "Setting up the intent-classif environment..."

# Install dependencies from pyproject.toml
echo "Installing Python dependencies..."
pip install -e .

# Setup .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please open the .env file and fill in your MLFLOW_TRACKING_URI."
else
    echo ".env file already exists. Skipping..."
fi

echo "Environment setup complete!"

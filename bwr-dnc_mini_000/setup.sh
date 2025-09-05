#!/bin/bash
# setup.sh

echo "Setting up BWR-DNC Phase 1 environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r backend/requirements.txt

# Install API dependencies
echo "Installing API dependencies..."
pip install -r api/requirements.txt

echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
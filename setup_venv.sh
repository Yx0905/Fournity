#!/bin/bash
# Script to recreate virtual environment

cd "/Users/liuyuxiang/Desktop/Rework datathon"

# Remove old broken venv
echo "Removing old virtual environment..."
rm -rf venv

# Create new virtual environment
echo "Creating new virtual environment..."
python3 -m venv venv

# Activate and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source venv/bin/activate"

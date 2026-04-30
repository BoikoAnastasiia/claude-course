#!/bin/bash
# Activate the virtual environment for this project
source .venv/bin/activate
echo "Virtual environment activated. Python version:"
python --version
echo "Available packages:"
pip list | grep -E "(anthropic|dotenv|jupyter|ipykernel)"
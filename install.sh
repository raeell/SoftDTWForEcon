#!/bin/bash

# Install dependencies in venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Create directory for plots
mkdir plots
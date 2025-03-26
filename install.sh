#!/bin/bash

# Install dependencies in venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Initialize submodule for soft-dtw divergence calculation
git submodule init
git submodule update
git apply --directory=soft-dtw-divergences/ patches/softdtw.patch
cd soft-dtw-divergences
python setup.py install
cd ..
# Create Jupyter kernel for venv
python -m ipykernel install --user --name .venv
# Create directory for plots
mkdir plots

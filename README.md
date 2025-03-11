# SoftDTWForEcon
Dépôt pour le projet final de l'évaluation du cours Mise en production (ENSAE 3A).

```
# Install dependencies in venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Clone repository with soft-dtw implementation
git clone https://github.com/google-research/soft-dtw-divergences.git
cd soft-dtw-divergences
python setup.py install
cd ..
git apply --directory=soft-dtw-divergences/ patches/softdtw.patch
# Create Jupyter kernel for venv
python -m ipykernel install --user --name .venv
```

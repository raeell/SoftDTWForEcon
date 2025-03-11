# SoftDTWForEcon
Dépôt pour le projet final de l'évaluation du cours Mise en production (ENSAE 3A).

```python
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
```

Le code du projet est pour l'instant dans un notebook (`notebooks/projet_time_series.ipynb`). Les métriques sont dans `src/metrics.py`


# ⏱️ TimeSeriesForecast

Projet final du cours **Mise en production** (ENSAE 3A).  
Ce dépôt contient une **API de prédiction** et un **site web interactif** pour explorer des modèles de séries temporelles appliqués à :
- 🌤️ la **météo** sur les prochaines 24 heures
- 🚕 les **courses de taxi** sur les 5 prochaines heures

---

## 📈 Objectifs

Ce projet propose une API permettant d’interroger deux modèles prédictifs :
- **MSE** (Mean Squared Error)
- **Soft-DTW**, une distance dynamique différentiable (Cuturi & Blondel, 2017 : [arXiv](https://arxiv.org/abs/1703.01541))

Les modèles sont entraînés avec validation croisée et enregistrés via **MLflow**.

---

## ⚙️ Installation

```bash
./install.sh                  # Installe les dépendances et crée l’environnement virtuel
source .venv/bin/activate     # Active l’environnement Python
```

---

## ✅ Tests

Pour tester le code : 

```bash
pytest tests/
```

---

## 🧪 Entraînement des modèles


`python train_taxi.py` et `python train_weather.py` démarrent respectivement des entraînements enregistrés par MLFlow en validation croisée pour les données de taxi et de météo. Ces entraînements peuvent être paramétrés en ligne de commande, par exemple :

#### Météo

```bash
python train_weather.py --epochs 1 --k_folds 5 --batch_size 512 --experiment_name training_weather --gamma 10 --hidden_size 10 --lr 1e-2
```

---

## 🧾 Évaluation

Pour évaluer les modèles enregistrés par MLFlow : 

```bash
python eval_taxi.py      # Évalue les modèles taxi
python eval_weather.py   # Évalue les modèles météo
```

---

## ☁️ Données

Les données sont accessibles à l’adresse suivante :  
📦 [MinIO S3](https://minio.lab.sspcloud.fr/tnguyen/diffusion)

---

## 🖥️ API FastAPI

Lancer l’API localement :

```bash
uvicorn api.main:app --reload
```

---

## 🌐 Interface Web (Quarto)

Lancer le site localement :

```bash
cd quarto
quarto preview
```

Voir la version en ligne 👉 [https://tuduyen-nguyen.github.io/TimeSeriesForecast](https://tuduyen-nguyen.github.io/TimeSeriesForecast)

---

## 🚀 CI/CD & Déploiement

[![prod](https://github.com/tuduyen-nguyen/TimeSeriesForecast/actions/workflows/prod.yml/badge.svg)](https://github.com/tuduyen-nguyen/TimeSeriesForecast/actions/workflows/prod.yml)

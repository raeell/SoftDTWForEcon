
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
bash install.sh               # Installe les dépendances et crée l’environnement virtuel
source .venv/bin/activate     # Active l’environnement Python
export PYTHONPATH=:$PWD/src   # Indique le chemin du module du projet
```

Le fichier `src/.env.example` contient les variables d'environnement à initialiser pour faire fonctionner le projet. 

Si vous utilisez le SSP Cloud, il suffit de lancer un service MLflow avant de lancer le service qui fait tourner ce projet pour initialiser les variables MLflow. Quant aux variables AWS, elles s'initialisent automatiquement. 

---

## ✅ Tests

Pour tester le code : 

```bash
pytest tests/
```

---

## 🧪 Entraînement des modèles


`python src/train/train_taxi.py` et `python src/train/train_weather.py` démarrent respectivement des entraînements enregistrés par MLFlow en validation croisée pour les données de taxi et de météo. Ces entraînements peuvent être paramétrés en ligne de commande, par exemple :

#### Météo

```bash
python src/train/train_weather.py --epochs 1 --k_folds 5 --batch_size 512 --experiment_name training_weather --gamma 10 --hidden_size 10 --lr 1e-2
```

---

## 🧾 Évaluation

Pour évaluer les modèles enregistrés par MLFlow aux adresses `models:/model_{MSE_ou_SDTW}_taxi/latest` et `models:/model_{MSE_ou_SDTW}_weather/latest` : 

```bash
python src/eval/eval_taxi.py      # Évalue les modèles taxi stockés dans models:/model_{MSE_ou_SDTW}_taxi/latest
python src/eval/eval_weather.py   # Évalue les modèles météo dans models:/model_{MSE_ou_SDTW}_weather/latest
```

---

## ☁️ Données

Les données sont accessibles à l’adresse suivante :  
📦 [MinIO S3](https://minio.lab.sspcloud.fr/tnguyen/diffusion)

---

## 🖥️ API FastAPI

Lancer l’API localement :

```bash
uvicorn api.api:app --reload
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

Pour déployer le projet avec Kubernetes, commencer par changer l'URL spécifiée dans le fichier `deployment/ingress.yml` en l'URL de votre choix, l'URL pré-existante étant déjà utilisée pour le déploiement du projet par les contributrices. Puis, depuis un service SSP Cloud, il suffit d'exécuter : 

```bash
kubectl apply -f deployment/
```

[![prod](https://github.com/tuduyen-nguyen/TimeSeriesForecast/actions/workflows/prod.yml/badge.svg)](https://github.com/tuduyen-nguyen/TimeSeriesForecast/actions/workflows/prod.yml)

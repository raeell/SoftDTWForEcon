# TimeSeriesForecast
Dépôt pour le projet final de l'évaluation du cours Mise en production (ENSAE 3A).

Ce projet met à disposition une API pour vous permettre de faire vos prédictions sur la météo sur les prochaines 24h ou sur le nombre de courses de taxi sur les 5 prochaines heures ! 

Les modèles de prédiction sont entraînés avec deux types de fonctions de perte : la mean squared error (MSE) et la soft-DTW introduite dans Cuturi et Blondel, 2017 [https://arxiv.org/abs/1703.01541].

Exécuter `install.sh` pour configurer l'environnement et le projet.

Exécuter `source .venv/bin/activate` pour se placer dans l'environnement du projet. 

Exécuter `pytest tests/` pour tester le code. 

`python train_taxi.py` et `python train_weather.py` démarrent respectivement des entraînements enregistrés par MLFlow en validation croisée pour les données de taxi et de météo. Ces entraînements peuvent être paramétrés en ligne de commande, par exemple : 

```python
python train_weather.py --epochs 1 --k_folds 5 --batch_size 512 --experiment_name training_weather --gamma 10 --hidden_size 10 --lr 1e-2
```

`python eval_taxi.py` et `python eval_weather.py` permettent respectivement d'évaluer les modèles enregistrés par MLFlow pour les données de taxi et de météo. 

Les données sont stockées dans le stockage externe S3 au lien suivant : https://minio.lab.sspcloud.fr/tnguyen/diffusion

📘 Voir le site ici : https://tuduyen-nguyen.github.io/TimeSeriesForecast

[![prod](https://github.com/tuduyen-nguyen/TimeSeriesForecast/actions/workflows/prod.yml/badge.svg)](https://github.com/tuduyen-nguyen/TimeSeriesForecast/actions/workflows/prod.yml)


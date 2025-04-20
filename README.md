# SoftDTWForEcon
Dépôt pour le projet final de l'évaluation du cours Mise en production (ENSAE 3A).

Ce projet met à disposition une API pour vous permettre de faire vos prédictions sur la météo sur les prochaines 24h ou sur le nombre de courses de taxi sur les 5 prochaines heures ! 

Les modèles de prédiction sont entraînés avec deux types de fonctions de perte : la mean squared error (MSE) et la soft-DTW introduite dans Cuturi et Blondel, 2017 [https://arxiv.org/abs/1703.01541].

Exécuter `install.sh` pour configurer l'environnement et le projet.
Exécuter `pytest tests/` pour tester le code. 

Les données sont stockées dans le stockage externe S3 au lien suivant : https://minio.lab.sspcloud.fr/tnguyen/diffusion

📘 Voir le site ici : https://tuduyen-nguyen.github.io/SoftDTWForEcon

[![prod](https://github.com/tuduyen-nguyen/SoftDTWForEcon/actions/workflows/prod.yml/badge.svg)](https://github.com/tuduyen-nguyen/SoftDTWForEcon/actions/workflows/prod.yml)


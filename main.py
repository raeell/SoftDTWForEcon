
"""
Main script for data processing and model training.

This script loads data from S3, preprocesses it, and trains a model.
"""
import os
import logging

import pandas as pd
import torch
import matplotlib.pyplot as plt
import s3fs
from dotenv import load_dotenv

from model.train_model import train_models_insee
from data.data_preprocessing import DataConfig
from model.eval_model import error_insee, eval_models_insee
from model.forecast_model import plot_forecasts_insee
from model.train_model import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)


load_dotenv()



data_config = DataConfig(
    split_train=0.6,
    split_val=0.2,
    input_size=20,
    output_size=5,
)
training_config = TrainingConfig(
    hidden_size=300,
    epochs=150,
    batch_size=50,
    lr=1e-2,
    gammas=[1],
    max_norm=100.0,
    divergence=False,
)


MY_BUCKET = os.getenv("MY_BUCKET", "laurinemir")
PATH = f"s3://{MY_BUCKET}/taxi_data/"




fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})
files = fs.ls(PATH)

dfs = []
for file in files:
    with fs.open(file) as f:
        df = pd.read_parquet(f)
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(df.head())

# df = pd.read_csv("DS_ICA_CSV_FR/DS_ICA_data.csv", sep=";", encoding="utf-8")
# df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")


df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)
df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
df_activity = df.groupby("hour").size().reset_index(name="num_trips")
df_activity = df_activity[df_activity["num_trips"] >= 100]
df_activity["hour"] = pd.to_datetime(df_activity["hour"])
print(df_activity.head(n=10))


# colonne = df.columns[0]  # colonne Activite
# df_activity = df[
#     (df[colonne] == "L")
#     & (df["SEASONAL_ADJUST"] == "Y")
#     & (df["IDX_TYPE"] == "ICA_SERV")
# ].sort_values(
#     by="TIME_PERIOD", ascending=True
# )  # choisir le secteur activité et indicateur

plt.figure(figsize=(12, 6))  # Définir la taille du graphique
plt.plot(df_activity["hour"], df_activity["num_trips"], marker="o", label="num_trips")
plt.xlabel("Date")
plt.ylabel("num_trips")
plt.title("Évolution de num_trips dans le temps")
plt.grid()
plt.legend()
plt.show()
plt.savefig("plots/output_taxi.png")


dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)


trainer = Trainer(df_activity, "num_trips", device, data_config, training_config)

models = trainer.train_models()

results = eval_models_insee(
    models,
    "num_trips",
    df_activity,
    device,
    data_config
)
plot_forecasts_insee(
    results,
    "num_trips",
    df_activity,
    training_config.gammas,
    data_config,

)
eval_models_insee(
    results,
    "num_trips",
    df_activity,
    device,
    data_config
)

"""Main script."""

import logging

import pandas as pd
import torch

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


df_insee = pd.read_csv("DS_ICA_CSV_FR/DS_ICA_data.csv", sep=";", encoding="utf-8")
df_insee["TIME_PERIOD"] = pd.to_datetime(df_insee["TIME_PERIOD"], format="%Y-%m")
colonne = df_insee.columns[0]  # colonne Activite
df_activity = df_insee[
    (df_insee[colonne] == "L")
    & (df_insee["SEASONAL_ADJUST"] == "Y")
    & (df_insee["IDX_TYPE"] == "ICA_SERV")
].sort_values(
    by="TIME_PERIOD",
    ascending=True,
)  # choisir le secteur activité et indicateur

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

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

trainer = Trainer(df_activity, "OBS_VALUE", device, data_config, training_config)

models = trainer.train_models()
results = eval_models_insee(
    models,
    "OBS_VALUE",
    df_activity,
    device,
    data_config,
)
plot_forecasts_insee(
    results,
    "OBS_VALUE",
    df_activity,
    training_config.gammas,
    data_config,
)
error_insee(
    results,
    "OBS_VALUE",
    df_activity,
    data_config,
)

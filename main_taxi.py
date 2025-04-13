"""Main Script."""

import logging

import torch
from dotenv import load_dotenv
from joblib import dump

from data.data_loader import DataLoaderS3
from data.data_preprocessing import DataConfig
from data.plot_figures import plot_times_series
from model.eval_model import error, eval_models
from model.plot_forecast_model import plot_forecasts
from model.train_model import Trainer, TrainingConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)

data_config = DataConfig(
    split_train=0.6,
    split_val=0.2,
    input_size=20,
    output_size=5,
    stride=1,
    input_columns = ["num_trips"],
    output_columns = ["num_trips"],
)
training_config = TrainingConfig(
    hidden_size=300,
    epochs=50,
    batch_size=50,
    lr=1e-3,
    gammas=[1],
    max_norm=100.0,
    divergence=False,
)

taxi_loader = DataLoaderS3(
    data_name="taxi",
    data_format="parquet",
    bucket_name="tnguyen",
    folder="diffusion/taxi_data",
)
df_taxi = taxi_loader.load_data()
plot_times_series(df_taxi, "hour", "num_trips")

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)

trainer = Trainer(
    df_taxi, device, data_config, training_config,
)

models = trainer.train_models()
for idx, model in enumerate(models):
    if idx != len(models) - 1:
        gamma = training_config.gammas[idx]
        dump(model, f"model_taxi_SDTW_gamma_{gamma}.joblib")
    else:
        dump(model, "model_taxi_MSE.joblib")

results = eval_models(
    models,
    df_taxi,
    device,
    data_config,
)
plot_forecasts(
    results,
    df_taxi,
    training_config.gammas,
    data_config,
)
error(
    results,
    df_taxi,
    data_config,
)

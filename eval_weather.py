"""Evaluation script for weather data."""

import logging

import mlflow
import torch
from dotenv import load_dotenv

from data.data_loader import DataLoaderS3
from data.data_preprocessing import DataConfig
from model.eval_model import error, eval_models
from model.plot_forecast_model import plot_forecasts
from model.train_model import TrainingConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)

model_mse_uri = "models:/model_MSE_weather/latest"
model_weather_mse = mlflow.pytorch.load_model(model_mse_uri, map_location=device)
model_dtw_uri = "models:/model_SDTW_weather/latest"
model_weather_dtw = mlflow.pytorch.load_model(model_dtw_uri, map_location=device)

logger.info("Weather models correctly loaded from MLFlow")

weather_loader = DataLoaderS3(
    data_name="weather",
    data_format="csv",
    bucket_name="tnguyen",
    folder="diffusion/weather_data",
)
df_weather = weather_loader.load_data().drop(columns=["Date Time"])

data_config = DataConfig(
    split_train=0.6,
    split_val=0.2,
    input_size=24,
    output_size=24,
    stride=1,
    input_columns=list(df_weather.columns),
    output_columns=["T (degC)"],
)


training_config = TrainingConfig(
    hidden_size=64,
    epochs=50,
    batch_size=50,
    lr=1e-3,
    gammas=[1],
    max_norm=100.0,
    divergence=False,
)

models = []
models.append(model_weather_dtw)
models.append(model_weather_mse)

results = eval_models(
    models,
    df_weather,
    device,
    data_config,
)
plot_forecasts(
    results,
    df_weather,
    training_config.gammas,
    data_config,
)
error(
    results,
    df_weather,
    data_config,
)

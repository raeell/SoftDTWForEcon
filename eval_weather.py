"""Evaluation script for weather data."""

import logging

import torch
from dotenv import load_dotenv
from pathlib import Path

from data.data_loader import DataLoaderS3
from data.data_preprocessing import DataConfig
from model.eval_model import error, eval_models
from model.mlp_baseline import MLP
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

weather_loader = DataLoaderS3(
    data_name="weather",
    data_format="csv",
    bucket_name="tnguyen",
    folder="diffusion/weather_data",
)
df_weather = weather_loader.load_data()

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

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)

dir_weights = Path("model_weights/weather_weights")
models = []
for path in dir_weights.glob("*gamma*.pt"):
    model = MLP(
        data_config.input_size,
        training_config.hidden_size,
        data_config.output_size,
        len(data_config.input_columns),
    )
    model.load_state_dict(torch.load(str(path)))
    models.append(model.to(device))
for path in dir_weights.glob("*MSE*.pt"):
    model = MLP(
        data_config.input_size,
        training_config.hidden_size,
        data_config.output_size,
        len(data_config.input_columns),
    )
    model.load_state_dict(torch.load(str(path)))
    models.append(model.to(device))

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

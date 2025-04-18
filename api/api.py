"""A simple API to make the prediction of time series."""

import logging

from datetime import datetime
import numpy as np
import torch
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query

from data.data_preprocessing import DataConfig, get_normalization_metrics
from data.data_loader import DataLoaderS3
from model.mlp_baseline import MLP
from data.data_preprocessing import DataConfig, get_normalization_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Prédiction des valeurs suivants de la série",
    description='Prédiction du traffic de taxi pour les 5 prochaines heures <br>Une version par API pour faciliter la réutilisation du modèle 🚀 <br><br><img src="https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg" width="200">',  # noqa: E501
)

taxi_loader = DataLoaderS3(
    data_name="taxi",
    data_format="parquet",
    bucket_name="tnguyen",
    folder="diffusion/taxi_data",
)
df_taxi = taxi_loader.load_data()

weather_loader = DataLoaderS3(
    data_name="weather",
    data_format="csv",
    bucket_name="tnguyen",
    folder="diffusion/weather_data",
)
df_weather = weather_loader.load_data()


@app.get("/", tags=["Welcome"])
def show_welcome_page() -> dict:
    """Show welcome page with model name and version."""
    return {
        "Message": "API de prédiction heures de taxi ou de météo",
        "Model_name": "Taxi ML ou Weather ML",
        "Model_version": "0.2",
    }


@app.get("/predict_taxi", tags=["Predict_taxi"])
async def predict_taxi(
    date: str = Query(
        "2023-03-12 15:00:00", description="Date format: %Y-%m-%d %H:%M:%S"
    )
):
    """Predict taxi values."""

    try:
        input_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": "Date must be in format %Y-%m-%d %H:%M:%S"}

    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=20,
        output_size=5,
        stride=1,
        input_columns=["num_trips"],
        output_columns=["num_trips"],
    )
    model_taxi = MLP(
        input_size=20,
        hidden_size=300,
        output_size=5,
        num_features=1,
    )
    model_taxi.load_state_dict(
        torch.load(
            "model_weights/taxi_weights/model_taxi_MSE.pt",
            map_location=torch.device("cpu"),
        ),
    )
    input_array = df_taxi[
        pd.to_datetime(
            df_taxi["hour"],
            format="%Y-%m-%d %H:%M:%S",
        )
        < input_date
    ].tail(data_config.input_size)

    if input_array.shape[0] < data_config.input_size:
        return {
            "error": "Not enough data before the selected date to generate prediction."
        }

    input_array = input_array[data_config.input_columns].to_numpy().astype(np.float32)
    x_test = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
    x_mean, x_std, y_mean, y_std = get_normalization_metrics(df_taxi, data_config)
    x_mean = torch.tensor(x_mean, dtype=torch.float32)
    x_std = torch.tensor(x_std, dtype=torch.float32)
    x_test = (x_test - x_mean) / x_std

    with torch.no_grad():
        y_pred = model_taxi(x_test).detach()
        prediction = (y_pred * y_std + y_mean).numpy().tolist()

    return {
        "date recue": input_date,
        "Prédiction taxi": prediction,
    }


@app.get("/predict_weather", tags=["Predict_weather"])
async def predict_weather(
    date: str = Query(
        "12.03.2023 15:00:00", description="Date format: %d.%m.%Y %H:%M:%S"
    )
):
    """Prédiction météo à partir d'une date."""

    try:
        input_date = datetime.strptime(date, "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return {"error": "Date must be in format %d.%m.%Y %H:%M:%S"}

    df_meteo = df_weather.drop(columns=["Date Time"])
    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=24,
        output_size=24,
        stride=1,
        input_columns=list(df_meteo.columns),
        output_columns=["T (degC)"],
    )
    x_mean, x_std, y_mean, y_std = get_normalization_metrics(df_meteo, data_config)
    x_mean = torch.tensor(x_mean, dtype=torch.float32)
    x_std = torch.tensor(x_std, dtype=torch.float32)
    input_array = df_weather[
        pd.to_datetime(
            df_weather["Date Time"],
            format="%d.%m.%Y %H:%M:%S",
        )
        < input_date
    ].tail(data_config.input_size)
    input_array = np.array(input_array[list(df_meteo.columns)].to_numpy())

    if input_array.shape[0] < data_config.input_size:
        return {
            "error": "Not enough data before the selected date to generate prediction."
        }

    model_weather = MLP(
        data_config.input_size,
        64,
        data_config.output_size,
        len(df_meteo.columns),
    )
    model_weather.load_state_dict(
        torch.load(
            "model_weights/weather_weights/model_weather_MSE.pt",
            map_location=torch.device("cpu"),
        )
    )

    model_weather.eval()

    x_test = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
    x_norm = (x_test - x_mean) / (x_std)

    with torch.no_grad():
        y_pred = model_weather(x_norm)
        y_pred_denorm = y_pred * y_std + y_mean
    prediction = y_pred_denorm.squeeze(0).tolist()

    return {"variables_reçues": {"annee": input_date}, "prediction": prediction}

"""A simple API to make the prediction of time series ."""

import logging
from typing import Annotated

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Query

from data.data_loader import DataLoaderS3
from model.mlp_baseline import MLP

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

weather_loader = DataLoaderS3(
    data_name="weather",
    data_format="csv",
    bucket_name="tnguyen",
    folder="diffusion/weather_data",
)
df_weather = weather_loader.load_data()

model_weather = MLP(
    input_size=24,
    hidden_size=64,
    output_size=24,
    num_features=len(list(df_weather.columns)),
)
model_weather.load_state_dict(
    torch.load(
        "model_weights/weather_weights/model_weather_MSE.pt",
        map_location=torch.device("cpu"),
    ),
)


@app.get("/", tags=["Welcome"])
def show_welcome_page() -> dict:
    """Show welcome page with model name and version."""
    return {
        "Message": "API de prédiction heures de taxi",
        "Model_name": "Taxi ML",
        "Model_version": "0.1",
    }


@app.get("/predict_taxi", tags=["Predict"])
async def predict_taxi(
    valeurs_anciennes: Annotated[list[int], Query()] = [1000] * 5
    + [400] * 2
    + [600] * 3
    + [900] * 10,
) -> dict:
    """Predict taxi values."""
    x_test = torch.Tensor(np.array(valeurs_anciennes)).unsqueeze(0)
    x_mean = x_test.mean(dim=0, keepdim=True)
    x_std = x_test.std(dim=0, keepdim=True)
    x_test = (x_test - x_mean) / x_std

    prediction = (model_taxi(x_test) * x_std + x_mean).tolist()

    return {
        "Valeurs reçues": valeurs_anciennes,
        "Prédiction taxi": prediction,
    }

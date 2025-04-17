"""A simple API to make the prediction of time series ."""

from typing import Annotated

from datetime import datetime
import numpy as np
import torch
import pandas as pd
from fastapi import FastAPI, Query
from model.mlp_baseline import MLP
from data.data_loader import DataLoaderS3
from data.data_preprocessing import DataConfig, get_normalization_metrics

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
model_taxi.load_state_dict(torch.load("model_weights/taxi_weights/model_taxi_MSE.pt"))


@app.get("/", tags=["Welcome"])
def show_welcome_page() -> dict:
    """Show welcome page with model name and version."""
    return {
        "Message": "API de prédiction heures de taxi",
        "Model_name": "Taxi ML",
        "Model_version": "0.1",
    }


@app.get("/predict_taxi", tags=["Predict_taxi"])
async def predict_taxi(
    valeurs_anciennes: Annotated[list[int], Query()] = [1000] * 5
    + [400] * 2
    + [600] * 3
    + [900] * 10,
) -> dict:
    """Predict."""
    x_test = torch.Tensor(np.array(valeurs_anciennes)).unsqueeze(0)
    x_mean = x_test.mean(dim=0, keepdim=True)
    x_std = x_test.std(dim=0, keepdim=True)
    x_test = (x_test - x_mean) / x_std

    prediction = (model_taxi(x_test) * x_std + x_mean).tolist()

    return {
        "Valeurs reçues": valeurs_anciennes,
        "Prédiction": prediction,
    }


@app.get("/predict_weather", tags=["Predict_weather"])
async def predict_weather(
        date: str = Query("12.03.2023 15:00:00", description="Date format: %d.%m.%Y %H:%M:%S")
        ):

    """Prédiction météo à partir d'une date."""

    try:
        input_date = datetime.strptime(date, "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return {"error": "Date must be in format %d.%m.%Y %H:%M:%S"}

    # Recupérer les données
    weather_loader = DataLoaderS3(
        data_name="weather",
        data_format="csv",
        bucket_name="tnguyen",
        folder="diffusion/weather_data",
    )

    df_weather = weather_loader.load_data()
    df_meteo = df_weather.drop(columns=["Date Time"])
    print(df_meteo)
    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=24,
        output_size=24,
        stride=1,
        input_columns=list(df_meteo.columns),
        output_columns=["T (degC)"],
    )
    
    x_mean, x_std, _, _ = get_normalization_metrics(df_meteo, data_config)
    x_mean = torch.tensor(x_mean, dtype=torch.float32)
    x_std = torch.tensor(x_std, dtype=torch.float32)
    input_array = df_weather[pd.to_datetime(
            df_weather["Date Time"],
            format="%d.%m.%Y %H:%M:%S",
        ) < input_date].tail(24)
    input_array = np.array(input_array[list(df_meteo.columns)].to_numpy())

    if input_array.shape[0] < data_config.input_size:
        return {"error": "Not enough data before the selected date to generate prediction."}

    model_weather = MLP(
        data_config.input_size,
        64,
        data_config.output_size,
        len(df_meteo.columns),
    )
    model_weather.load_state_dict(torch.load("model_weights/weather_weights/model_weather_MSE.pt"))
    model_weather.eval()

    x_test = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
    x_norm = (x_test - x_mean) / (x_std)
    print(x_norm.dtype)  
    y_pred = model_weather(x_norm)
    y_pred_denorm = y_pred * x_std + x_mean
    prediction = y_pred_denorm.squeeze(0).tolist()

    return {
        "variables_reçues": {
            "annee": input_date
        },
        "prediction": prediction
    }

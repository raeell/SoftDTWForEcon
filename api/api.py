"""A simple API to make the prediction of time series ."""

import json
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, Query
from model.mlp_baseline import MLP

app = FastAPI(
    title="Prédiction des valeurs suivants de la série",
    description='Prédiction du traffic de taxi pour les 5 prochaines heures <br>Une version par API pour faciliter la réutilisation du modèle 🚀 <br><br><img src="https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg" width="200">',  # noqa: E501
)
model = MLP(
    input_size=20,
    hidden_size=300,
    output_size=5,
    num_features=1,
)
model.load_state_dict(torch.load("model/model_DTW.pt"))


@app.get("/", tags=["Welcome"])
def show_welcome_page() -> dict:
    """Show welcome page with model name and version."""
    return {
        "Message": "API de prédiction heures de taxi",
        "Model_name": "Taxi ML",
        "Model_version": "0.1",
    }


@app.get("/predict", tags=["Predict"])
async def predict(
    valeurs_anciennes: Annotated[list[int], Query()] = [1000] * 5
    + [400] * 2
    + [600] * 3
    + [900] * 10,
) -> str:
    """Predict."""
    print(f"Valeurs reçues : {valeurs_anciennes}")
    x_test = torch.Tensor(np.array(valeurs_anciennes)).unsqueeze(0)
    x_mean = x_test.mean(dim=1, keepdim=True)
    x_std = x_test.std(dim=1, keepdim=True)
    x_test = (x_test - x_mean) / x_std
    print(x_test.shape)

    prediction = (model(x_test) * x_std + x_mean).tolist()

    return json.dumps(prediction)

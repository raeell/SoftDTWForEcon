"""A simple API to make the prediction of time series ."""
from fastapi import FastAPI, Query
from joblib import load
import torch
from data.data_preprocessing import to_tensor_and_normalize

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)
app = FastAPI(
    title="Prédiction des valeurs suivants de la série",
    description= "Prédiction du traffic de taxi pour les 5 prochaines heures <br>Une version par API pour faciliter la réutilisation du modèle 🚀" +\
        "<br><br><img src=\"https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg\" width=\"200\">"
    )


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API de prédiction heures de taxi",
        "Model_name": 'Taxi ML',
        "Model_version": "0.1",
    }

model = load('model_DTW.joblib').to(device)
@app.get("/predict", tags=["Predict"])
async def predict(
    valeurs_anciennes:  list[int] = Query(default=[1000] * 20)
) -> str:
    """
    """
    print(f"Valeurs reçues : {valeurs_anciennes}")
    x = to_tensor_and_normalize(valeurs_anciennes).to(device).unsqueeze(-1)

    prediction = model(x).tolist()

    return prediction
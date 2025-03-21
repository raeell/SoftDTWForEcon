import pandas as pd
import torch
from model.train_model import train_models_insee
from model.eval_model import eval_models_insee, error_insee
from model.forecast_model import plot_forecasts_insee

df = pd.read_csv("DS_ICA_CSV_FR/DS_ICA_data.csv", sep=";", encoding="utf-8")
print(df.columns)
df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
colonne = df.columns[0]  # colonne Activite
df_activity = df[
    (df[colonne] == "L")
    & (df["SEASONAL_ADJUST"] == "Y")
    & (df["IDX_TYPE"] == "ICA_SERV")
].sort_values(
    by="TIME_PERIOD", ascending=True
)  # choisir le secteur activité et indicateur


input_size = 20
output_size = 5

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


gammas = [1]


models = train_models_insee(
    "OBS_VALUE",
    df_activity,
    device=device,
    input_size=input_size,
    output_size=output_size,
    gammas=gammas,
    divergence=False,
)
results = eval_models_insee(
    models,
    "OBS_VALUE",
    df_activity,
    device,
    input_size=input_size,
    output_size=output_size,
)
plot_forecasts_insee(
    results,
    "OBS_VALUE",
    df_activity,
    gammas,
    input_size=input_size,
    output_size=output_size,
)
error_insee(
    results,
    "OBS_VALUE",
    df_activity,
    device,
    input_size=input_size,
    output_size=output_size,
)

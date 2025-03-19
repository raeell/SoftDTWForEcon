import pandas as pd
import torch
from data/data_preprocessing import train_models_insee
from models/eval_model import eval_models_insee,error_insee
from models/forecast_model import plot_forecast_insee

df = pd.read_csv("../DS_ICA_CSV_FR/DS_ICA_data.csv", sep=";", encoding="utf-8")
print(df.columns)
df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
colonne = df.columns[0]  #colonne Activite
df_activity = df[(df[colonne] == "L") & (df["SEASONAL_ADJUST"] == "Y") & (df["IDX_TYPE"]=="ICA_SERV")].sort_values(by="TIME_PERIOD", ascending=True)  #choisir le secteur activité et indicateur


input_size = 20
output_size = 5

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)


gammas = [1]


models = train_models_insee("OBS_VALUE",df_activity, gammas=gammas,divergence=False)
results = eval_models_insee(models,"OBS_VALUE",df_activity)
plot_forecasts_insee(results, "OBS_VALUE",df_activity,gammas)
error_insee(results, "OBS_VALUE",df_activity)
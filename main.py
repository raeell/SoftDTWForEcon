import pandas as pd
import torch
from model.train_model import train_models_insee
from model.eval_model import eval_models_insee, error_insee
from model.forecast_model import plot_forecasts_insee
import matplotlib.pyplot as plt


file_list = ["yellow_tripdata_2025-01.parquet","yellow_tripdata_2024-12.parquet",
"yellow_tripdata_2024-11.parquet",
"yellow_tripdata_2024-10.parquet",
"yellow_tripdata_2024-09.parquet",
"yellow_tripdata_2024-08.parquet",
"yellow_tripdata_2024-07.parquet",
"yellow_tripdata_2024-06.parquet",
"yellow_tripdata_2024-05.parquet",
"yellow_tripdata_2024-04.parquet",
"yellow_tripdata_2024-03.parquet",
"yellow_tripdata_2024-02.parquet",
"yellow_tripdata_2024-01.parquet"]


df = pd.concat([pd.read_parquet("taxi_data/"+file) for file in file_list], ignore_index=True)
print(df.head())
#df = pd.read_csv("DS_ICA_CSV_FR/DS_ICA_data.csv", sep=";", encoding="utf-8")
# df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], format="%Y-%m-%d %H:%M:%S")
df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
df_activity = df.groupby("hour").size().reset_index(name="num_trips")
df_activity = df_activity[df_activity["num_trips"] >= 100]
df_activity["hour"] = pd.to_datetime(df_activity["hour"])
print(df_activity.head(n=10)) 
# colonne = df.columns[0]  # colonne Activite
# df_activity = df[
#     (df[colonne] == "L")
#     & (df["SEASONAL_ADJUST"] == "Y")
#     & (df["IDX_TYPE"] == "ICA_SERV")
# ].sort_values(
#     by="TIME_PERIOD", ascending=True
# )  # choisir le secteur activité et indicateur

plt.figure(figsize=(12, 6))  # Définir la taille du graphique
plt.plot(df_activity["hour"], df_activity["num_trips"],marker='o', label="num_trips")  
plt.xlabel("Date")
plt.ylabel("num_trips")
plt.title("Évolution de num_trips dans le temps")
plt.grid()
plt.legend()
plt.show()
plt.savefig("plots/output_taxi.png") 


input_size = 100
output_size = 5

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


gammas = [1]


models = train_models_insee(
    "num_trips",
    df_activity,
    device=device,
    input_size=input_size,
    output_size=output_size,
    gammas=gammas,
    divergence=False,
)
results = eval_models_insee(
    models,
    "num_trips",
    df_activity,
    device,
    input_size=input_size,
    output_size=output_size,
)
plot_forecasts_insee(
    results,
    "num_trips",
    df_activity,
    gammas,
    input_size=input_size,
    output_size=output_size,
)
error_insee(
    results,
    "num_trips",
    df_activity,
    device,
    input_size=input_size,
    output_size=output_size,
)

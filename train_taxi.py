"""Training script for taxi data."""

import argparse
import logging
import os

import mlflow
import torch
from dotenv import load_dotenv

from data.data_loader import DataLoaderS3
from data.data_preprocessing import DataConfig
from data.plot_figures import plot_times_series
from model.train_model import Trainer, TrainingConfig
from model.eval_model import error, eval_models

load_dotenv()

parser = argparse.ArgumentParser(description="Paramètres d'entraînement taxi")
parser.add_argument("--epochs", type=int, default=1, help="Nombre d'epochs")
parser.add_argument(
    "--experiment_name", type=str, default="taximl", help="MLFlow experiment name"
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)

# LOGGING IN MLFLOW -----------------
mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
logger.info(f"Saving experiment in {mlflow_server}")
mlflow.set_tracking_uri(mlflow_server)
mlflow.set_experiment(args.experiment_name)

data_config = DataConfig(
    split_train=0.6,
    split_val=0.2,
    input_size=20,
    output_size=5,
    stride=1,
    input_columns=["num_trips"],
    output_columns=["num_trips"],
)
training_config = TrainingConfig(
    hidden_size=300,
    epochs=args.epochs,
    batch_size=50,
    lr=1e-3,
    gammas=[1],
    max_norm=100.0,
    divergence=False,
)

taxi_loader = DataLoaderS3(
    data_name="taxi",
    data_format="parquet",
    bucket_name="tnguyen",
    folder="diffusion/taxi_data",
)
df_taxi = taxi_loader.load_data()
plot_times_series(df_taxi, "hour", "num_trips")

# Log input data
input_data_mlflow = mlflow.data.from_pandas(
    df_taxi, source="s3://tnguyen/diffusion/taxi_data", name="Raw dataset"
)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)
logger.info("Using device %s", device)

trainer = Trainer(
    df_taxi,
    device,
    data_config,
    training_config,
)

with mlflow.start_run() as parent_run:
    # Log datasets
    mlflow.log_input(input_data_mlflow, context="raw")

    # Log parameters
    mlflow.log_param("hidden_size", training_config.hidden_size)
    mlflow.log_param("epochs", training_config.epochs)
    mlflow.log_param("batch_size", training_config.batch_size)
    mlflow.log_param("lr", training_config.lr)
    mlflow.log_param("max_norm", training_config.max_norm)
    mlflow.log_param("divergence", training_config.divergence)

    models = trainer.train_models()
    for idx, model in enumerate(models):
        if idx != len(models) - 1:
            gamma = training_config.gammas[idx]
            model_path = f"model_weights/taxi_weights/model_taxi_SDTW_gamma_{gamma}.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
        else:
            model_path = "model_weights/taxi_weights/model_taxi_MSE.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)

    results = eval_models(
        models,
        df_taxi,
        device,
        data_config,
    )
    mean_mse, std_mse, mean_dtw, std_dtw = error(
        results,
        df_taxi,
        data_config,
    )
    for idx_score in range(len(mean_mse)):
        mlflow.log_metric(f"mean_mse_{idx_score}", mean_mse[idx_score])
        mlflow.log_metric(f"std_mse_{idx_score}", std_mse[idx_score])
        mlflow.log_metric(f"mean_dtw_{idx_score}", mean_dtw[idx_score])
        mlflow.log_metric(f"std_dtw_{idx_score}", std_dtw[idx_score])

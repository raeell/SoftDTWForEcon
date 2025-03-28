"""Main Script."""

import logging

import torch
from dotenv import load_dotenv

from data.data_preprocessing import DataConfig, DataLoaderS3
from model.eval_model import error_insee, eval_models_insee
from model.forecast_model import plot_forecasts_insee
from model.train_model import Trainer, TrainingConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ],
)
logger = logging.getLogger(__name__)

data_config = DataConfig(
    split_train=0.6,
    split_val=0.2,
    input_size=20,
    output_size=5,
)
training_config = TrainingConfig(
    hidden_size=300,
    epochs=1,
    batch_size=50,
    lr=1e-2,
    gammas=[1],
    max_norm=100.0,
    divergence=False,
)

taxi_loader = DataLoaderS3(
    data_name="taxi",
    data_format="parquet",
    bucket_name="laurinemir",
    folder="diffusion",
)
df_taxi = taxi_loader.load_data()

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)

trainer = Trainer(df_taxi, "num_trips", device, data_config, training_config)

models = trainer.train_models()

results = eval_models_insee(models, "num_trips", df_taxi, device, data_config)
plot_forecasts_insee(
    results,
    "num_trips",
    df_taxi,
    training_config.gammas,
    data_config,
)
error_insee(
    results,
    "num_trips",
    df_taxi,
    data_config,
)

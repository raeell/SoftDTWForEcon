"""Main Script."""

import logging

import torch
from dotenv import load_dotenv

from data.data_preprocessing import DataConfig, DataLoaderS3
from model.eval_model import eval_models_insee, error_insee
from model.forecast_model import plot_forecasts_insee
from model.train_model import Trainer, TrainingConfig

from joblib import dump

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
df = taxi_loader.load_data()
var = "num_trips"

# insee_loader = DataLoaderS3(data_name="insee", data_format="csv",bucket="tudyen")
# df = insee_loader.load_data()
# var = "OBS_VALUE"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEV)

trainer = Trainer(df, var, device, data_config, training_config)

models = trainer.train_models()
dump(models[0], 'model_DTW.joblib')

results = eval_models_insee(models, var, df, device, data_config)
plot_forecasts_insee(
    results,
    var,
    df,
    training_config.gammas,
    data_config,
)
error_insee(
    results,
    var,
    df,
    data_config,
)

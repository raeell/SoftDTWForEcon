"""Training tests."""

import os

import mlflow
import pandas as pd
import torch

from data_processing.data_preprocessing import DataConfig
from model.train_model import Trainer, TrainingConfig

NB_MODELS_TRAINED = 2


def test_training() -> None:
    """Test the training with dummy data."""
    mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_server)
    mlflow.set_experiment("pytest dummy experiment")
    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=20,
        output_size=5,
        stride=1,
        input_columns=["num_trips"],
        output_columns=["num_trips"],
        k_folds=2,
    )
    training_config = TrainingConfig(
        hidden_size=10,
        epochs=1,
        batch_size=512,
        lr=1e-3,
        gammas=[1],
        max_norm=100.0,
        divergence=False,
    )

    dummy_data = pd.DataFrame(
        {
            "num_trips": [10, 20, 30, 40, 50] * 100,
            "date": pd.date_range(start="2023-01-01", periods=500, freq="D"),
        },
    )

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    trainer = Trainer(dummy_data, device, data_config, training_config)

    with mlflow.start_run():
        models = trainer.train_models()

    assert models is not None, "Les modèles n'ont pas été entraînés correctement."
    assert len(models) == NB_MODELS_TRAINED, "Il n'y a pas le bon nombre de modèles."

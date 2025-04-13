"""Training tests."""

import pandas as pd
import torch

from data.data_preprocessing import DataConfig
from model.train_model import Trainer, TrainingConfig

NB_MODELS_TRAINED = 2


def test_training() -> None:
    """Test the training with dummy data."""
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
        hidden_size=10,
        epochs=1,
        batch_size=50,
        lr=1e-3,
        gammas=[1],
        max_norm=100.0,
        divergence=False,
    )

    dummy_data = pd.DataFrame(
        {
            "num_trips": [10, 20, 30, 40, 50] * 1000,
            "date": pd.date_range(start="2023-01-01", periods=5000, freq="D"),
        },
    )

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    trainer = Trainer(dummy_data, device, data_config, training_config)

    models = trainer.train_models()

    assert models is not None, "Les modèles n'ont pas été entraînés correctement."
    assert len(models) == NB_MODELS_TRAINED, "Il n'y a pas le bon nombre de modèles."

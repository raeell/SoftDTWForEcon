"""Global tests."""
import pandas as pd
import torch

from data.data_preprocessing import DataConfig
from model.eval_model import error_insee, eval_models_insee
from model.forecast_model import plot_forecasts_insee
from model.train_model import Trainer, TrainingConfig


def test_main_script() -> None:
    """Test the main script with dummy data."""
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
        lr=1e-3,
        gammas=[1],
        max_norm=100.0,
        divergence=False,
    )

    dummy_data = pd.DataFrame({
        "num_trips": [10, 20, 30, 40, 50]*1000,
        "date": pd.date_range(start="2023-01-01", periods=5000, freq="D"),
    })

    var = "num_trips"

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    trainer = Trainer(dummy_data, var, device, data_config, training_config)

    models = trainer.train_models()

    results = eval_models_insee(models, var, dummy_data, device, data_config)
    plot_forecasts_insee(
        results,
        var,
        dummy_data,
        training_config.gammas,
        data_config,
    )
    error_insee(
        results,
        var,
        dummy_data,
        data_config,
    )

    assert models is not None, "Les modèles n'ont pas été entraînés correctement."
    assert results is not None, "Les résultats d'évaluation sont manquants."

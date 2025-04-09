"""Plot forecast tests."""

import pandas as pd
import torch

from data.data_preprocessing import DataConfig
from model.plot_forecast_model import plot_forecasts_insee
from model.train_model import TrainingConfig


def test_plot_forecast() -> None:
    """Test the plotting of forecasts with dummy data."""
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

    dummy_data = pd.DataFrame(
        {
            "num_trips": [10, 20, 30, 40, 50] * 1000,
            "date": pd.date_range(start="2023-01-01", periods=5000, freq="D"),
        },
    )

    var = "num_trips"

    results = [
        torch.zeros(1, 5),
        torch.zeros(1, 5),
    ]

    plot_forecasts_insee(
        results,
        var,
        dummy_data,
        training_config.gammas,
        data_config,
        nb_plots=1,
    )

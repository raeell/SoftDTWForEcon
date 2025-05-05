"""Plot forecast tests."""

import numpy as np
import pandas as pd
import torch

from data_processing.data_preprocessing import DataConfig
from model.plot_forecast_model import plot_forecasts
from model.train_model import TrainingConfig


def test_plot_forecast() -> None:
    """Test the plotting of forecasts with dummy data."""
    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=20,
        output_size=5,
        stride=25,
        input_columns=["num_trips"],
        output_columns=["num_trips"],
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
            "num_trips": [10 + np.random.randint(5) for i in range(5000)],
            "date": pd.date_range(start="2023-01-01", periods=5000, freq="D"),
        },
    )

    results = [
        torch.rand(1, 5, 1),
        torch.rand(1, 5, 1),
    ]

    plot_forecasts(
        results,
        dummy_data,
        training_config.gammas,
        data_config,
        nb_plots=1,
    )

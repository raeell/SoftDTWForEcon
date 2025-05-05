"""Evaluation tests."""

import pandas as pd
import torch

from data_processing.data_preprocessing import DataConfig
from model.eval_model import eval_models
from model.mlp_baseline import MLP
from model.train_model import TrainingConfig


def test_eval_model() -> None:
    """Test the evaluation with dummy data and dummy models."""
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
            "num_trips": [10, 20, 30, 40, 50] * 1000,
            "date": pd.date_range(start="2023-01-01", periods=5000, freq="D"),
        },
    )

    models = [
        MLP(
            data_config.input_size,
            training_config.hidden_size,
            data_config.output_size,
            num_features=1,
        ),
        MLP(
            data_config.input_size,
            training_config.hidden_size,
            data_config.output_size,
            num_features=1,
        ),
    ]

    results = eval_models(
        models,
        dummy_data,
        torch.device("cpu"),
        data_config,
    )

    assert results is not None, "Les résultats d'évaluation sont manquants."

    assert (
        results[0].shape[1] == data_config.output_size
    ), "Les résultats ne sont pas de la bonne taille."

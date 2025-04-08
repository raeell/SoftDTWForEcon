"""Data preprocessing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Config parameters for train/val/test split and window size."""

    split_train: float
    split_val: float
    input_size: int
    output_size: int


def create_time_series_window(
    values: list,
    input_size: int,
    output_size: int,
) -> tuple[np.array]:
    """Split time series in equal size windows."""
    x = []
    y = []
    for i in range(len(values) - input_size - output_size):
        x.append(values[i : i + input_size])
        y.append(values[i + input_size : i + input_size + output_size])
    return np.array(x), np.array(y)


def get_normalization_metrics(training_data: list | np.array) -> tuple[float]:
    """Get mean and std of training data."""
    return np.array(training_data).mean(), np.array(training_data).std()


def to_tensor_and_normalize(data: list | np.array) -> torch.Tensor:
    """Convert to tensor and normalize data along axis 0."""
    x = torch.Tensor(np.array(data))
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)


def to_array_and_normalize(data: list | np.array) -> np.array:
    """Convert to numpy array and normalize along axis 0."""
    x = np.array(data)
    return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)


def train_test_val_split(
    df: pd.DataFrame,
    column: str,
    data_config: DataConfig,
) -> tuple[np.array]:
    """Create train/test/val split of data."""
    values = df[column].to_numpy()
    split_train = int(len(values) * data_config.split_train)
    split_val = int(len(values) * data_config.split_val)
    train_data = values[:split_train]
    val_data = values[split_train : split_train + split_val]
    test_data = values[split_train + split_val :]
    x_train, y_train = create_time_series_window(
        train_data,
        data_config.input_size,
        data_config.output_size,
    )
    x_val, y_val = create_time_series_window(
        val_data,
        data_config.input_size,
        data_config.output_size,
    )
    x_test, y_test = create_time_series_window(
        test_data,
        data_config.input_size,
        data_config.output_size,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test

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
    stride: int
    input_columns: list[str]
    output_columns: list[str]


def create_time_series_window(
    train_input: np.array,
    train_output: np.array,
    input_size: int,
    output_size: int,
    stride: int = 1,
) -> tuple[np.array]:
    """Split time series in equal size windows with a specified stride."""
    x = []
    y = []
    for i in range(0, len(train_input) - input_size - output_size + 1, stride):
        x.append(train_input[i : i + input_size])
        y.append(train_output[i + input_size : i + input_size + output_size])
    return np.array(x), np.array(y)


def get_normalization_metrics(
    df: pd.DataFrame,
    data_config: DataConfig,
) -> tuple[float]:
    """Get mean and std of training data."""
    input_data = df[data_config.input_columns].to_numpy()
    training_input = input_data[: int(len(input_data) * data_config.split_train)]
    output_data = df[data_config.output_columns].to_numpy()
    training_output = output_data[: int(len(output_data) * data_config.split_train)]
    return (
        np.array(training_input).mean(axis=0),
        np.array(training_input).std(axis=0),
        np.array(training_output).mean(axis=0),
        np.array(training_output).std(axis=0),
    )


def to_tensor_and_normalize(
    data: list | np.array,
    normalization_metrics: (np.array, np.array) | None = None,
) -> torch.Tensor:
    """Convert to tensor and normalize data along axis 0."""
    x = torch.Tensor(np.array(data))
    if normalization_metrics is None:
        return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    return x - normalization_metrics[0] / normalization_metrics[1]


def to_array_and_normalize(
    data: list | np.array,
    normalization_metrics: (np.array, np.array) | None = None,
) -> np.array:
    """Convert to numpy array and normalize data along axis 0."""
    x = np.array(data)
    if normalization_metrics is None:
        return (x - x.mean(axis=0, keepdims=True)) / x.std(
            axis=0,
            keepdims=True,
            ddof=1,
        )
    return x - normalization_metrics[0] / normalization_metrics[1]


def train_test_val_split(
    df: pd.DataFrame,
    data_config: DataConfig,
) -> tuple[np.array]:
    """Create train/test/val split of data with input and output columns."""
    input_values = df[data_config.input_columns].to_numpy()
    output_values = df[data_config.output_columns].to_numpy()

    split_train = int(len(input_values) * data_config.split_train)
    split_val = int(len(input_values) * data_config.split_val)

    train_input = input_values[:split_train]
    val_input = input_values[split_train : split_train + split_val]
    test_input = input_values[split_train + split_val :]

    train_output = output_values[:split_train]
    val_output = output_values[split_train : split_train + split_val]
    test_output = output_values[split_train + split_val :]

    x_train, y_train = create_time_series_window(
        train_input,
        train_output,
        data_config.input_size,
        data_config.output_size,
        data_config.stride,
    )
    x_val, y_val = create_time_series_window(
        val_input,
        val_output,
        data_config.input_size,
        data_config.output_size,
        data_config.stride,
    )
    x_test, y_test = create_time_series_window(
        test_input,
        test_output,
        data_config.input_size,
        data_config.output_size,
        data_config.stride,
    )

    return x_train, y_train, x_val, y_val, x_test, y_test

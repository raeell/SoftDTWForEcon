"""Data preprocessing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.model_selection import KFold

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
    k_folds: int = 5


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
) -> list:
    """Get mean and std of training data."""
    splits, (x_test, y_test) = train_test_val_split(
        df,
        data_config,
    )
    normalization_metrics = []
    for split in splits:
        x_train, y_train, _, _ = split
        normalization_metrics.append(
            [
                x_train.mean(axis=0, keepdims=True),
                x_train.std(axis=0, keepdims=True, ddof=1),
                y_train.mean(axis=0, keepdims=True),
                y_train.std(axis=0, keepdims=True, ddof=1),
            ]
        )
    return normalization_metrics


def to_tensor_and_normalize(
    data: list | np.array,
    normalization_metrics: (np.array, np.array) | None = None,
) -> torch.Tensor:
    """Convert to tensor and normalize data along axis 0."""
    x = torch.Tensor(np.array(data))
    if normalization_metrics is None:
        return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    return (x - torch.tensor(normalization_metrics[0])) / torch.tensor(
        normalization_metrics[1],
    )


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
    return (x - normalization_metrics[0]) / normalization_metrics[1]


def train_test_val_split(
    df: pd.DataFrame, data_config: DataConfig
) -> tuple[
    list[tuple[np.array, np.array, np.array, np.array]], tuple[np.array, np.array]
]:
    """Create train/test/val split of data with input and output columns using cross-validation."""
    input_values = df[data_config.input_columns].to_numpy()
    output_values = df[data_config.output_columns].to_numpy()
    x, y = create_time_series_window(
        input_values,
        output_values,
        input_size=data_config.input_size,
        output_size=data_config.output_size,
        stride=data_config.stride,
    )

    # Split into train and test sets
    split_train_val = int(len(x) * (data_config.split_train + data_config.split_val))
    train_val_input = x[:split_train_val]
    train_val_output = y[:split_train_val]
    x_test = x[split_train_val:]
    y_test = y[split_train_val:]

    # Create KFold splits for the train set
    kf = KFold(n_splits=data_config.k_folds, shuffle=True, random_state=42)
    splits = []

    for train_index, val_index in kf.split(train_val_input):
        train_input, val_input = (
            train_val_input[train_index],
            train_val_input[val_index],
        )
        train_output, val_output = (
            train_val_output[train_index],
            train_val_output[val_index],
        )

        splits.append((train_input, train_output, val_input, val_output))

    return splits, (x_test, y_test)

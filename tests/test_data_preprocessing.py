"""Data preprocessing tests."""

import numpy as np
import pandas as pd
import torch

from data.data_preprocessing import (
    DataConfig,
    create_time_series_window,
    get_normalization_metrics,
    to_array_and_normalize,
    to_tensor_and_normalize,
    train_test_val_split,
)

TOL = 1e-6


def test_create_time_series_window() -> None:
    """Test creation of time series windows."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    input_size = 3
    output_size = 2
    x, y = create_time_series_window(
        values,
        values,
        input_size,
        output_size,
        stride=5,
    )

    expected_x = np.array(
        [
            [1, 2, 3],
            [6, 7, 8],
        ],
    )
    expected_y = np.array(
        [
            [4, 5],
            [9, 10],
        ],
    )

    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)


def test_get_normalization_metrics() -> None:
    """Test getting normalization metrics."""
    df_test = pd.DataFrame(
        {
            "column_name": [1, 1, 1, 1, 1] * 100,
        },
    )
    data_config = DataConfig(
        split_train=1.0,
        split_val=0.0,
        input_size=3,
        output_size=3,
        stride=1,
        input_columns=["column_name"],
        output_columns=["column_name"],
        k_folds=10,
    )
    normalization_metrics = get_normalization_metrics(df_test, data_config)

    expected_mean = 1
    expected_std = 0

    for fold in normalization_metrics:
        mean_x, std_x, mean_y, std_y = (
            fold[0].squeeze(),
            fold[1].squeeze(),
            fold[2].squeeze(),
            fold[3].squeeze(),
        )
        for value in mean_x:
            assert np.abs(value - expected_mean) < TOL
        for value in mean_y:
            assert np.abs(value - expected_mean) < TOL
        for value in std_x:
            assert np.abs(value - expected_std) < TOL
        for value in std_y:
            assert np.abs(value - expected_std) < TOL


def test_to_tensor_and_normalize() -> None:
    """Test normalizing and converting to tensor."""
    data = [1, 2, 3, 4, 5]
    normalized_tensor = to_tensor_and_normalize(data)

    expected_tensor = torch.tensor([-1.2649, -0.6325, 0.0, 0.6325, 1.2649])

    torch.testing.assert_close(
        normalized_tensor,
        expected_tensor,
        rtol=1.3e-6,
        atol=1e-4,
    )


def test_to_array_and_normalize() -> None:
    """Test normalizing and converting to array."""
    data = [1, 2, 3, 4, 5]
    normalized_array = to_array_and_normalize(data)

    expected_array = np.array([-1.2649, -0.6325, 0.0, 0.6325, 1.2649])

    np.testing.assert_array_almost_equal(normalized_array, expected_array, decimal=4)


def test_train_test_val_split() -> None:
    """Test splitting into train, test, val subsets."""
    df_test = pd.DataFrame(
        {
            "column_name": [1, 2, 3, 4, 5] * 10,
        },
    )
    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=3,
        output_size=2,
        stride=5,
        input_columns=["column_name"],
        output_columns=["column_name"],
        k_folds=4,
    )
    splits, (x_test, y_test) = train_test_val_split(
        df_test,
        data_config,
    )

    for x_train, y_train, x_val, y_val in splits:
        assert x_train.shape == (6, 3, 1)
        assert x_val.shape == (2, 3, 1)
        assert x_test.shape == (2, 3, 1)
        assert y_train.shape == (6, 2, 1)
        assert y_val.shape == (2, 2, 1)
        assert y_test.shape == (2, 2, 1)

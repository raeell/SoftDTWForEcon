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


def test_create_time_series_window() -> None:
    """Test creation of time series windows."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    input_size = 3
    output_size = 2
    x, y = create_time_series_window(values, input_size, output_size)

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
    training_data = [1, 2, 3, 4, 5]
    mean, std = get_normalization_metrics(training_data)

    expected_mean = 3.0
    expected_std = np.sqrt(2.0)

    assert mean == expected_mean
    assert std == expected_std


def test_to_tensor_and_normalize() -> None:
    """Test normalizing and converting to tensor."""
    data = [1, 2, 3, 4, 5]
    normalized_tensor = to_tensor_and_normalize(data)

    expected_tensor = torch.tensor([-1.2649, -0.6325, 0.0, 0.6325, 1.2649])

    torch.testing.assert_close(
        normalized_tensor, expected_tensor, rtol=1.3e-6, atol=1e-4,
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
            "column_name": [1, 2, 3, 4, 5]*10,
        },
    )
    data_config = DataConfig(
        split_train=0.6,
        split_val=0.2,
        input_size=3,
        output_size=2,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(
        df_test,
        "column_name",
        data_config,
    )

    assert x_train.shape == (6, 3)
    assert x_val.shape == (2, 3)
    assert x_test.shape == (2, 3)
    assert y_train.shape == (6, 2)
    assert y_val.shape == (2, 2)
    assert y_test.shape == (2, 2)

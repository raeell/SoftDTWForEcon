"""Evaluate model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from tslearn.metrics import dtw

from data.data_preprocessing import (
    to_array_and_normalize,
    to_tensor_and_normalize,
    train_test_val_split,
)

if TYPE_CHECKING:
    import pandas as pd


def eval_models_insee(
    models: list,
    column: str,
    df: pd.DataFrame,
    device: str,
    split_train: float = 0.6,
    split_val: float = 0.2,
    input_size: int = 20,
    output_size: int = 5,
) -> list:
    """Inference for models."""
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(
        df, column, split_train, split_val, input_size, output_size
    )
    x_test = to_tensor_and_normalize(x_test).to(device)
    res = []
    for m in range(len(models)):
        result = models[m](x_test)
        res.append(result)
    return res


def error_insee(
    res: list,
    column: str,
    df: pd.DataFrame,
    split_train: float = 0.6,
    split_val: float = 0.2,
    input_size: int = 20,
    output_size: int = 5,
) -> None:
    """Compute error for model inferences."""
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(
        df,
        column,
        split_train,
        split_val,
        input_size,
        output_size,
    )
    gt = to_array_and_normalize(y_test)
    res = np.array(
        [r.cpu().detach().numpy() if isinstance(r, torch.Tensor) else r for r in res],
    )

    # MSE
    mse = np.mean((gt - res[0].squeeze(-1)) ** 2, axis=1)
    std_mse = np.std((gt - res[0].squeeze(-1)) ** 2)
    mse = np.mean(mse)

    # DTW
    dtw_models = np.zeros((len(res), gt.shape[1]))
    for m in range(len(res)):
        for ts in range(gt.shape[1]):
            dist = dtw(gt[0, ts], res[m][ts])
            dtw_models[m][ts] = dist
    std_dtw = np.std(dtw_models, axis=1)
    dtws = np.mean(dtw_models, axis=1)
    print(f"MSE: {np.round(mse, 2)} +- {np.round(std_mse, 2)}")
    print(f"DTW: {np.round(dtws, 2)} +- {np.round(std_dtw, 2)}")

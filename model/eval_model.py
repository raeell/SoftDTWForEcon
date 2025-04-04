"""Evaluate model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from tslearn.metrics import dtw

from data.data_preprocessing import (
    DataConfig,
    to_array_and_normalize,
    to_tensor_and_normalize,
    train_test_val_split,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def eval_models_insee(
    models: list,
    column: str,
    df: pd.DataFrame,
    device: str,
    data_config: DataConfig,
) -> list:
    """Inference for models."""
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(
        df,
        column,
        data_config,
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
    data_config: DataConfig,
) -> None:
    """Compute error for model inferences."""
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(
        df,
        column,
        data_config,
    )
    gt = to_array_and_normalize(y_test)
    res = np.array(
        [r.cpu().detach().numpy() if isinstance(r, torch.Tensor) else r for r in res],
    )

    # MSE
    mses = np.zeros((len(res), gt.shape[0]))
    for model in range(len(res)):
        for ts in range(gt.shape[0]):
            mses[model][ts] = np.mean((gt[ts] - res[model][ts].squeeze(-1)) ** 2)
    std_mse = np.std(mses, axis=1)
    mean_mse = np.mean(mses, axis=1)

    # DTW
    dtws = np.zeros((len(res), gt.shape[0]))
    for model in range(len(res)):
        for ts in range(gt.shape[0]):
            dist = dtw(gt[ts], res[model][ts])
            dtws[model][ts] = dist
    std_dtw = np.std(dtws, axis=1)
    mean_dtw = np.mean(dtws, axis=1)

    logger.info("MSE: %s +- %s", np.round(mean_mse, 2), np.round(std_mse, 2))
    logger.info("DTW: %s +- %s", np.round(mean_dtw, 2), np.round(std_dtw, 2))

"""Train models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch import nn
from tslearn.metrics import SoftDTWLossPyTorch

from data.data_preprocessing import (
    DataConfig,
    get_normalization_metrics,
    to_tensor_and_normalize,
    train_test_val_split,
)

from .mlp_baseline import MLP

if TYPE_CHECKING:
    import pandas as pd


import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training config parameters."""

    hidden_size: int = 300
    epochs: int = 150
    batch_size: int = 50
    lr: float = 1e-2
    max_norm: float = 100.0
    divergence: bool = True
    gammas: list[float] = field(default_factory=lambda: [1e-2, 1e-1, 1, 10, 100])
    patience: int = 10


class Trainer:
    """Trainer class which performs training and plotting."""

    def __init__(
        self,
        df: pd.DataFrame,
        device: torch.device,
        data_config: DataConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Initialize Trainer class."""
        self.df = df
        self.input_columns = data_config.input_columns
        self.output_columns = data_config.output_columns
        self.device = device
        self.data_config = data_config
        self.training_config = training_config
        self.models_sdtw = []
        self.models_mse = []

        self.splits, (self.x_test, self.y_test) = train_test_val_split(
            self.df,
            self.data_config,
        )
        self.best_models = dict(
            {gamma: None for gamma in self.training_config.gammas}, **{"mse": None}
        )
        self.best_val_losses = dict(
            {gamma: float("inf") for gamma in self.training_config.gammas},
            **{"mse": float("inf")},
        )

    def train_model_softdtw(self, gamma: float) -> None:
        """Train model with SoftDTW loss."""
        fold_no = 0
        for x_train, y_train, x_val, y_val in self.splits:
            with mlflow.start_run(nested=True) as child_run:
                mlflow.log_param("fold", fold_no)
                logger.info(
                    f"Training fold {fold_no+1}/{self.data_config.k_folds} for SoftDTW with gamma={gamma}"
                )

                normalization_metrics = get_normalization_metrics(
                    self.df, self.data_config
                )

                x_train = to_tensor_and_normalize(
                    x_train,
                    (
                        normalization_metrics[fold_no][0],
                        normalization_metrics[fold_no][1],
                    ),
                ).float()
                y_train = to_tensor_and_normalize(
                    y_train,
                    (
                        normalization_metrics[fold_no][2],
                        normalization_metrics[fold_no][3],
                    ),
                ).float()
                x_val = (
                    to_tensor_and_normalize(
                        x_val,
                        (
                            normalization_metrics[fold_no][0],
                            normalization_metrics[fold_no][1],
                        ),
                    )
                    .to(self.device)
                    .float()
                )
                y_val = (
                    to_tensor_and_normalize(
                        y_val,
                        (
                            normalization_metrics[fold_no][2],
                            normalization_metrics[fold_no][3],
                        ),
                    )
                    .to(self.device)
                    .float()
                )

                model = MLP(
                    input_size=self.data_config.input_size,
                    hidden_size=self.training_config.hidden_size,
                    output_size=self.data_config.output_size,
                    num_features=x_train.shape[-1],
                ).to(self.device)
                loss_fn = SoftDTWLossPyTorch(
                    gamma=gamma,
                    normalize=self.training_config.divergence,
                ).to(self.device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=self.training_config.lr
                )

                losses, val_losses, model = self._train_model(
                    model, loss_fn, optimizer, x_train, y_train, x_val, y_val
                )

                self._plot_losses(
                    losses, val_losses, f"gamma = {gamma}, fold = {fold_no}"
                )
                self._update_best_model(
                    model,
                    val_losses[-1],
                    gamma,
                    f"SoftDTW_gamma_{gamma}_fold_{fold_no}",
                )

                fold_no += 1

    def train_model_mse(self) -> None:
        """Train model with MSE loss."""
        fold_no = 0
        for x_train, y_train, x_val, y_val in self.splits:
            with mlflow.start_run(nested=True) as child_run:
                mlflow.log_param("fold", fold_no)
                logger.info(
                    f"Training fold {fold_no+1}/{self.data_config.k_folds} for MSE"
                )

                normalization_metrics = get_normalization_metrics(
                    self.df, self.data_config
                )

                x_train = to_tensor_and_normalize(
                    x_train,
                    (
                        normalization_metrics[fold_no][0],
                        normalization_metrics[fold_no][1],
                    ),
                ).float()
                y_train = to_tensor_and_normalize(
                    y_train,
                    (
                        normalization_metrics[fold_no][2],
                        normalization_metrics[fold_no][3],
                    ),
                ).float()
                x_val = (
                    to_tensor_and_normalize(
                        x_val,
                        (
                            normalization_metrics[fold_no][0],
                            normalization_metrics[fold_no][1],
                        ),
                    )
                    .to(self.device)
                    .float()
                )
                y_val = (
                    to_tensor_and_normalize(
                        y_val,
                        (
                            normalization_metrics[fold_no][2],
                            normalization_metrics[fold_no][3],
                        ),
                    )
                    .to(self.device)
                    .float()
                )

                model = MLP(
                    input_size=self.data_config.input_size,
                    hidden_size=self.training_config.hidden_size,
                    output_size=self.data_config.output_size,
                    num_features=x_train.shape[-1],
                ).to(self.device)
                loss_fn = nn.MSELoss().to(self.device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=self.training_config.lr
                )

                losses, val_losses, model = self._train_model(
                    model, loss_fn, optimizer, x_train, y_train, x_val, y_val
                )

                self._plot_losses(losses, val_losses, f"MSE, fold = {fold_no}")
                self._update_best_model(
                    model, val_losses[-1], None, f"MSE_fold_{fold_no}"
                )

                fold_no += 1

    def _train_model(
        self,
        model: MLP,
        loss_fn: any,
        optimizer: torch.optim,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> tuple[list, list, MLP]:
        losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.training_config.epochs):
            shuffled_idxs = torch.randperm(x_train.size(0))
            for batch_idx in range(
                0,
                x_train.size(0),
                self.training_config.batch_size,
            ):
                idxs = shuffled_idxs[
                    batch_idx : batch_idx + self.training_config.batch_size
                ]
                x_batch = x_train[idxs].to(self.device)
                y_batch = y_train[idxs].to(self.device)
                pred = model(x_batch)
                optimizer.zero_grad()
                loss = loss_fn(pred, y_batch).mean()
                loss.backward()
                mlflow.log_metric("training_loss", loss.detach().cpu().numpy())
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_config.max_norm,
                )
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())
                # Validation step
                with torch.no_grad():
                    pred = model(x_val)
                    val_loss = loss_fn(pred, y_val).mean()
                    mlflow.log_metric(
                        "validation_loss", val_loss.detach().cpu().numpy()
                    )
                val_losses.append(val_loss.detach().cpu().numpy())
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    "Epoch: %d, Train loss: %f, Validation loss: %f",
                    epoch,
                    loss,
                    val_loss,
                )

            if patience_counter >= self.training_config.patience:
                logger.info("Early stopping triggered")
                break

        model.load_state_dict(best_model)
        return losses, val_losses, model

    def _plot_losses(self, losses: list, val_losses: list, title: str) -> None:
        plt.plot(np.array(losses), label="training loss")
        plt.plot(np.array(val_losses), label="validation loss")
        plt.legend()
        plt.title(title)
        plt.savefig(f"plots/loss_curves_{title}.png")
        plt.clf()

    def _update_best_model(
        self, model: MLP, val_loss: float, gamma: float | None, model_name: str
    ) -> None:
        if gamma is not None:
            if val_loss < self.best_val_losses[gamma]:
                self.best_val_losses[gamma] = val_loss
                self.best_models[gamma] = model.state_dict()
                logger.info(
                    f"New best model for gamma={gamma}: {model_name} with val loss: {val_loss}"
                )
        else:
            if val_loss < self.best_val_losses["mse"]:
                self.best_val_losses["mse"] = val_loss
                self.best_models["mse"] = model.state_dict()
                logger.info(
                    f"New best model for MSE: {model_name} with val loss: {val_loss}"
                )

    def train_models(self) -> list[MLP]:
        """Train and return models using cross-validation."""
        for gamma in self.training_config.gammas:
            with mlflow.start_run(nested=True) as child_run:
                mlflow.log_param("sdtw", True)
                mlflow.log_param("gamma", gamma)
                self.train_model_softdtw(gamma)
        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_param("sdtw", False)
            self.train_model_mse()

        # Load the best models
        best_models = []
        for gamma in self.training_config.gammas:
            model = MLP(
                input_size=self.data_config.input_size,
                hidden_size=self.training_config.hidden_size,
                output_size=self.data_config.output_size,
                num_features=len(self.data_config.input_columns),
            ).to(self.device)
            model.load_state_dict(self.best_models[gamma])
            best_models.append(model)

        # Load the best MSE model
        mse_model = MLP(
            input_size=self.data_config.input_size,
            hidden_size=self.training_config.hidden_size,
            output_size=self.data_config.output_size,
            num_features=len(self.data_config.input_columns),
        ).to(self.device)
        mse_model.load_state_dict(self.best_models["mse"])
        best_models.append(mse_model)

        return best_models

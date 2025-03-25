"""Train models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tslearn.metrics import SoftDTWLossPyTorch

from data.data_preprocessing import (
    DataConfig,
    to_tensor_and_normalize,
    train_test_val_split,
)

from .mlp_baseline import MLP

if TYPE_CHECKING:
    import pandas as pd


from dataclasses import dataclass, field


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


class Trainer:
    """Trainer class which performs training and plotting."""

    def __init__(
        self,
        df: pd.DataFrame,
        column: str,
        device: torch.device,
        data_config: DataConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Initialize Trainer class."""
        self.df = df
        self.column = column
        self.device = device
        self.data_config = data_config
        self.training_config = training_config
        self.models_sdtw = []
        self.models_mse = []

        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = (
            train_test_val_split(self.df, self.column, self.data_config)
        )

        self.x_train = to_tensor_and_normalize(self.x_train).unsqueeze(-1)
        self.y_train = to_tensor_and_normalize(self.y_train).unsqueeze(-1)
        self.x_val = to_tensor_and_normalize(self.x_val).to(self.device).unsqueeze(-1)
        self.y_val = to_tensor_and_normalize(self.y_val).to(self.device).unsqueeze(-1)

    def train_model_softdtw(self, gamma: float) -> None:
        """Train model with SoftDTW loss."""
        model = MLP(
            input_size=self.data_config.input_size,
            hidden_size=self.training_config.hidden_size,
            output_size=self.data_config.output_size,
        ).to(self.device)
        loss_fn = SoftDTWLossPyTorch(
            gamma=gamma,
            normalize=self.training_config.divergence,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config.lr)

        losses, val_losses = self._train_model(model, loss_fn, optimizer)

        self._plot_losses(losses, val_losses, f"gamma = {gamma}")
        self.models_sdtw.append(model)

    def train_model_mse(self) -> None:
        """Train model with MSE loss."""
        model = MLP(
            input_size=self.data_config.input_size,
            hidden_size=self.training_config.hidden_size,
            output_size=self.data_config.output_size,
        ).to(self.device)
        loss_fn = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config.lr)

        losses, val_losses = self._train_model(model, loss_fn, optimizer)

        self._plot_losses(losses, val_losses, "MSE")
        self.models_mse.append(model)

    def _train_model(
        self, model: MLP, loss_fn: any, optimizer: torch.optim
    ) -> tuple[list]:
        losses = []
        val_losses = []

        for epoch in range(self.training_config.epochs):
            shuffled_idxs = torch.randperm(self.x_train.size(0))
            for batch_idx in range(
                0,
                self.x_train.size(0),
                self.training_config.batch_size,
            ):
                idxs = shuffled_idxs[
                    batch_idx : batch_idx + self.training_config.batch_size
                ]
                x_batch = self.x_train[idxs].to(self.device)
                y_batch = self.y_train[idxs].to(self.device)
                pred = model(x_batch)
                optimizer.zero_grad()
                loss = loss_fn(pred, y_batch).mean()
                losses.append(loss.detach().cpu().numpy())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.training_config.max_norm
                )
                optimizer.step()
            if epoch % 10 == 0:
                pred = model(self.x_val)
                val_loss = loss_fn(pred, self.y_val).mean()
                val_losses.extend([val_loss.detach().cpu().numpy()] * 10)
                print(
                    f"Epoch: {epoch}, Train loss: {loss}, Validation loss: {val_loss}"
                )

        return losses, val_losses

    def _plot_losses(self, losses: list, val_losses: list, title: str) -> None:
        plt.plot(np.array(losses), label="training loss")
        plt.plot(np.array(val_losses), label="validation loss")
        plt.legend()
        plt.title(title)
        plt.savefig(f"plots/loss_curves_{title}.png")
        plt.clf()

    def train_models(self) -> list[MLP]:
        """Train and return models."""
        for gamma in self.training_config.gammas:
            self.train_model_softdtw(gamma)
        self.train_model_mse()
        return self.models_sdtw + self.models_mse


# def train_models_insee(
#     column: str,
#     df: pd.DataFrame,
#     device: str,
#     data_config: DataConfig,
#     training_config: TrainingConfig,
# ) -> list:
#     """Train models, plot losses, return models."""
#     x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(
#         df,
#         column,
#         data_config,
#     )

#     x_train = to_tensor_and_normalize(x_train).unsqueeze(-1)
#     y_train = to_tensor_and_normalize(y_train).unsqueeze(-1)

#     x_val = to_tensor_and_normalize(x_val).to(device).unsqueeze(-1)
#     y_val = to_tensor_and_normalize(y_val).to(device).unsqueeze(-1)

#     models = []
#     for gamma in training_config.gammas:
#         model = MLP(
#             input_size=data_config.input_size,
#             hidden_size=training_config.hidden_size,
#             output_size=data_config.output_size,
#         ).to(device)
#         loss_fn = SoftDTWLossPyTorch(
#             gamma=gamma, normalize=training_config.divergence
#         ).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)

#         losses = []
#         val_losses = []

#         for epoch in range(training_config.epochs):
#             shuffled_idxs = torch.randperm(x_train.size(0))
#             for batch_idx in range(0, x_train.size(0), training_config.batch_size):
#                 idxs = shuffled_idxs[batch_idx : batch_idx + training_config.batch_size]
#                 x_batch = x_train[idxs].to(device)
#                 y_batch = y_train[idxs].to(device)
#                 pred = model(x_batch).to(device)
#                 optimizer.zero_grad()
#                 loss = loss_fn(pred, y_batch).mean()
#                 losses.append(loss.detach().cpu().numpy())
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), training_config.max_norm
#                 )
#                 optimizer.step()
#             if epoch % 10 == 0:
#                 # validation loss
#                 pred = model(x_val).to(device)
#                 val_loss = loss_fn(pred, y_val).mean()
#                 val_losses.extend([val_loss.detach().cpu().numpy()] * 10)
#                 print(
#                     f"Epoch: {epoch}, Train loss: {loss}, Validation loss: {val_loss}"
#                 )

#         plt.plot(np.array(losses), label="training loss")
#         plt.plot(np.array(val_losses), label="validation loss")
#         plt.legend()
#         plt.title(f"gamma = {gamma}")
#         plt.savefig("plots/loss_curves_softdtw.png")
#         plt.clf()

#         models.append(model)

#     # add model with MSE loss

#     model = MLP(
#         input_size=data_config.input_size,
#         hidden_size=training_config.hidden_size,
#         output_size=data_config.output_size,
#     ).to(device)
#     loss_fn = nn.MSELoss().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)

#     losses = []
#     val_losses = []

#     for epoch in range(training_config.epochs):
#         shuffled_idxs = torch.randperm(x_train.size(0))
#         for batch_idx in range(0, x_train.size(0), training_config.batch_size):
#             idxs = shuffled_idxs[batch_idx : batch_idx + training_config.batch_size]
#             x_batch = x_train[idxs].to(device)
#             y_batch = y_train[idxs].to(device)
#             pred = model(x_batch)
#             optimizer.zero_grad()
#             loss = loss_fn(pred, y_batch).mean()
#             losses.append(loss.cpu().detach().numpy())
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_norm)
#             optimizer.step()
#         if epoch % 10 == 0:
#             # validation loss
#             pred = model(x_val).to(device)
#             val_loss = loss_fn(pred, y_val).mean()
#             val_losses.extend([val_loss.detach().cpu().numpy()] * 10)
#             print(
#                 f"Epoch: {epoch}, Train loss: {loss}, Validation loss: {val_loss}",
#             )

#     plt.plot(np.array(losses), label="training loss")
#     plt.plot(np.array(val_losses), label="validation loss")
#     plt.legend()
#     plt.title("MSE")
#     plt.savefig("plots/loss_curves_MSE.png")
#     plt.clf()

#     models.append(model)

#     return models

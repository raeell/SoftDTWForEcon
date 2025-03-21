import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tslearn.metrics import SoftDTWLossPyTorch

from data.data_preprocessing import train_test_split
from .MLP_baseline import MLP


def train_models_insee(
    value,
    df,
    device,
    input_size=20,
    output_size=5,
    hidden_size=300,
    epochs=150,
    batch_size=50,
    lr=1e-2,
    gammas=[1e-2, 1e-1, 1, 10, 100],
    max_norm=100.0,
    divergence=True,
):
    X_train, y_train, X_test, y_test = train_test_split(
        df, value, 0.6, input_size, output_size
    )

    x = torch.Tensor(np.array(X_train)).unsqueeze(-1)
    x = (x - x.mean(dim=0)) / x.std(dim=0)
    y = torch.Tensor(np.array(y_train)).unsqueeze(-1)
    y = (y - y.mean(dim=0)) / y.std(dim=0)

    X_val = x[-120:].to(device)
    y_val = y[-120:].to(device)
    X_tensor = x[:-120]
    Y_tensor = y[:-120]

    models = []
    for gamma in gammas:
        model = MLP(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        ).to(device)
        loss_fn = SoftDTWLossPyTorch(gamma=gamma, normalize=divergence).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []
        val_losses = []

        for epoch in range(epochs):
            shuffled_idxs = torch.randperm(X_tensor.size(0))
            for batch_idx in range(0, X_tensor.size(0), batch_size):
                # select batch
                idxs = shuffled_idxs[batch_idx : batch_idx + batch_size]
                x_batch = X_tensor[idxs].to(device)
                y_batch = Y_tensor[idxs].to(device)
                pred = model(x_batch).to(device)
                optimizer.zero_grad()
                loss = loss_fn(pred, y_batch).mean()
                losses.append(loss.detach().cpu().numpy())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            if epoch % 10 == 0:
                # validation loss
                pred = model(X_val).to(device)
                val_loss = loss_fn(pred, y_val).mean()
                val_losses.extend([val_loss.detach().cpu().numpy()] * 10)
                print(
                    "Epoch: {}, Train loss: {}, Validation loss: {}".format(
                        epoch, loss, val_loss
                    )
                )

        plt.plot(np.array(losses), label="training loss")
        plt.plot(np.array(val_losses), label="validation loss")
        plt.legend()
        plt.title("gamma = {}".format(gamma))
        plt.savefig("plots/loss_curves_softdtw.png")
        plt.clf()

        models.append(model)

    # add model with MSE loss

    model = MLP(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    ).to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    val_losses = []

    for epoch in range(epochs):
        shuffled_idxs = torch.randperm(X_tensor.size(0))
        for batch_idx in range(0, X_tensor.size(0), batch_size):
            # select batch
            idxs = shuffled_idxs[batch_idx : batch_idx + batch_size]
            x_batch = X_tensor[idxs].to(device)
            y_batch = Y_tensor[idxs].to(device)
            pred = model(x_batch)
            optimizer.zero_grad()
            loss = loss_fn(pred, y_batch).mean()
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        if epoch % 10 == 0:
            # validation loss
            pred = model(X_val).to(device)
            val_loss = loss_fn(pred, y_val).mean()
            val_losses.extend([val_loss.detach().cpu().numpy()] * 10)
            print(
                "Epoch: {}, Train loss: {}, Validation loss: {}".format(
                    epoch, loss, val_loss
                )
            )

    plt.plot(np.array(losses), label="training loss")
    plt.plot(np.array(val_losses), label="validation loss")
    plt.legend()
    plt.title("MSE")
    plt.savefig("plots/loss_curves_MSE.png")
    plt.clf()

    models.append(model)

    return models

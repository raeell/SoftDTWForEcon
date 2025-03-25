"""Plot forecasts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_preprocessing import to_array_and_normalize, train_test_val_split


def plot_forecasts_insee(
    res: list,
    column: str,
    df: pd.DataFrame,
    gammas: list,
    split_train: float = 0.6,
    split_val: float = 0.2,
    input_size: int = 20,
    output_size: int = 5,
) -> None:
    """Plot forecasts."""
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(
        df, column, split_train, split_val, input_size, output_size
    )
    x_test = to_array_and_normalize(x_test)
    y_test = to_array_and_normalize(y_test)
    for i in range(10):
        for m in range(len(res)):
            if m < len(res) - 1:
                plt.plot(
                    np.arange(input_size, input_size + output_size),
                    y_test[i],
                    color="grey",
                    label="Ground truth",
                )
                plt.plot(
                    np.arange(input_size, input_size + output_size),
                    res[m][i].cpu().detach().squeeze(-1),
                    color="red",
                    label=f"gamma = {gammas[m]}",
                    alpha=0.6,
                )
                plt.axvline(x=input_size, linestyle="dashed", color="k")
                plt.title(f"{gammas[m]}")
                plt.legend()
                plt.grid()
                plt.savefig(f"plots/forecasts_softdtw_{i}_{m}.png")
                plt.clf()

            else:
                plt.plot(
                    np.arange(input_size, input_size + output_size),
                    y_test[i],
                    color="grey",
                    label="Ground truth",
                )
                plt.plot(
                    np.arange(input_size, input_size + output_size),
                    res[m][i].cpu().detach().squeeze(-1),
                    color="red",
                    label="MSE",
                    alpha=0.6,
                )
                plt.axvline(x=input_size, linestyle="dashed", color="k")
                plt.grid()

                plt.title(f"{i}")
                plt.legend()
                plt.savefig(f"plots/forecast_MSE_{i}_{m}.png")
                plt.clf()

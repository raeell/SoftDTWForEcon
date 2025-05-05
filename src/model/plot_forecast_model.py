"""Plot forecasts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.data_preprocessing import (
    DataConfig,
    to_array_and_normalize,
    train_test_val_split,
)


def plot_forecasts(
    res: list,
    df: pd.DataFrame,
    gammas: list,
    data_config: DataConfig,
    nb_plots: int = 10,
) -> None:
    """Plot forecasts."""
    _, (x_test, y_test) = train_test_val_split(
        df,
        data_config,
    )
    x_test = to_array_and_normalize(x_test)
    y_test = to_array_and_normalize(y_test)
    for i in range(0, nb_plots * data_config.input_size, data_config.input_size):
        for m in range(len(res)):
            for column in range(len(data_config.output_columns)):
                if m < len(res) - 1:  # First models are DTW models
                    plt.plot(
                        np.arange(
                            data_config.input_size,
                            data_config.input_size + data_config.output_size,
                        ),
                        y_test[i, :, column],
                        color="grey",
                        label="Ground truth",
                    )
                    plt.plot(
                        np.arange(
                            data_config.input_size,
                            data_config.input_size + data_config.output_size,
                        ),
                        res[m][i, :, column].cpu().detach(),  # .squeeze(-1),
                        color="red",
                        label=f"gamma = {gammas[m]}",
                        alpha=0.6,
                    )
                    plt.axvline(x=data_config.input_size, linestyle="dashed", color="k")
                    plt.title(f"{gammas[m], column}")
                    plt.legend()
                    plt.grid()
                    plt.savefig(f"plots/forecasts_softdtw_{i}_{column}_{m}.png")
                    plt.clf()

                else:  # Last model is MSE Model
                    plt.plot(
                        np.arange(
                            data_config.input_size,
                            data_config.input_size + data_config.output_size,
                        ),
                        y_test[i, :, column],
                        color="grey",
                        label="Ground truth",
                    )
                    plt.plot(
                        np.arange(
                            data_config.input_size,
                            data_config.input_size + data_config.output_size,
                        ),
                        res[m][i, :, column].cpu().detach(),  # .squeeze(-1),
                        color="red",
                        label="MSE",
                        alpha=0.6,
                    )
                    plt.axvline(x=data_config.input_size, linestyle="dashed", color="k")
                    plt.grid()

                    plt.title(f"{i, column}")
                    plt.legend()
                    plt.savefig(f"plots/forecast_MSE_{i}_{column}_{m}.png")
                    plt.clf()

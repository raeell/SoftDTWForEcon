import numpy as np
import matplotlib.pyplot as plt

from data.data_preprocessing import train_test_split


def plot_forecasts_insee(res, value, df, gammas, input_size=20, output_size=5):
    X_train, y_train, X_test, y_test = train_test_split(
        df, value, 0.6, input_size, output_size
    )
    X_test = np.array(X_test)
    input_size = int(X_test.shape[1])
    output_size = int(y_test.shape[1])

    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    y_test = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)
    for i in range(0, 10):
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
                    label="gamma = {}".format(gammas[m]),
                    alpha=0.6,
                )
                plt.axvline(x=input_size, linestyle="dashed", color="k")
                plt.title("{}".format(gammas[m]))
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

                plt.title("{}".format(i))
                plt.legend()
                plt.savefig(f"plots/forecast_MSE_{i}_{m}.png")
                plt.clf()

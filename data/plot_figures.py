"""Plot time series."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_times_series(
    df_activity: pd.DataFrame,
    time_period_column: str,
    column: str,
) -> None:
    """Plot time series column."""
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_activity[time_period_column],
        df_activity[column],
        marker="o",
        label=column,
    )
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title("Évolution de" + column + "dans le temps")
    plt.grid()
    plt.legend()

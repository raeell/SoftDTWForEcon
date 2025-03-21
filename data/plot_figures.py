import matplotlib.pyplot as plt


def plot_times_series(df_activity, time_period, value):
    plt.figure(figsize=(12, 6))  # Définir la taille du graphique
    plt.plot(df_activity[time_period], df_activity[value], marker="o", label=value)
    plt.xlabel("Date")
    plt.ylabel(value)
    plt.title("Évolution de" + value + "dans le temps")
    plt.grid()
    plt.legend()

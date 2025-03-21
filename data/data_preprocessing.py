import numpy as np


def create_time_series_window(values, input_size, output_size):
    X = []
    y = []
    for i in range(len(values) - input_size - output_size):
        X.append(values[i : i + input_size])
        y.append(values[i + input_size : i + input_size + output_size])
    return np.array(X), np.array(y)


def train_test_split(df, value, split_train, input_size, output_size):
    values = df[value].values
    split_idx = int(len(values) * split_train)
    train_data, test_data = values[:split_idx], values[split_idx:]
    X_train, y_train = create_time_series_window(train_data, input_size, output_size)
    X_test, y_test = create_time_series_window(test_data, input_size, output_size)
    return X_train, y_train, X_test, y_test

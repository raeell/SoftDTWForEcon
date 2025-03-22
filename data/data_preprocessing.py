import numpy as np
import torch


def create_time_series_window(values, input_size, output_size):
    X = []
    y = []
    for i in range(len(values) - input_size - output_size):
        X.append(values[i : i + input_size])
        y.append(values[i + input_size : i + input_size + output_size])
    return np.array(X), np.array(y)


def get_normalization_metrics(training_data):
    return np.array(training_data).mean(), np.array(training_data).std()


def to_tensor_and_normalize(data):
    x = torch.Tensor(np.array(data))
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    return x


def to_array_and_normalize(data):
    x = np.array(data)
    x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    return x


def train_test_val_split(df, value, split_train, split_val, input_size, output_size):
    values = df[value].values
    split_train = int(len(values) * split_train)
    split_val = int(len(values) * split_val)
    train_data = values[:split_train]
    val_data = values[split_train : split_train + split_val]
    test_data = values[split_train + split_val :]
    X_train, y_train = create_time_series_window(train_data, input_size, output_size)
    X_val, y_val = create_time_series_window(val_data, input_size, output_size)
    X_test, y_test = create_time_series_window(test_data, input_size, output_size)
    return X_train, y_train, X_val, y_val, X_test, y_test

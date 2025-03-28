"""Data preprocessing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import s3fs
import numpy as np
import torch
import pandas as pd

from dataclasses import dataclass


@dataclass
class DataConfig:
    """Config parameters for train/val/test split and window size."""

    split_train: float
    split_val: float
    input_size: int
    output_size: int


def create_time_series_window(
    values: list,
    input_size: int,
    output_size: int,
) -> tuple[np.array]:
    """Split time series in equal size windows."""
    x = []
    y = []
    for i in range(len(values) - input_size - output_size):
        x.append(values[i : i + input_size])
        y.append(values[i + input_size : i + input_size + output_size])
    return np.array(x), np.array(y)


def get_normalization_metrics(training_data: list | np.array) -> tuple[float]:
    """Get mean and std of training data."""
    return np.array(training_data).mean(), np.array(training_data).std()


def to_tensor_and_normalize(data: list | np.array) -> torch.Tensor:
    """Convert to tensor and normalize data along axis 0."""
    x = torch.Tensor(np.array(data))
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)


def to_array_and_normalize(data: list | np.array) -> np.array:
    """Convert to numpy array and normalize along axis 0."""
    x = np.array(data)
    return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)


def train_test_val_split(
    df: pd.DataFrame,
    column: str,
    data_config: DataConfig,
) -> tuple[np.array]:
    """Create train/test/val split of data."""
    values = df[column].to_numpy()
    split_train = int(len(values) * data_config.split_train)
    split_val = int(len(values) * data_config.split_val)
    train_data = values[:split_train]
    val_data = values[split_train : split_train + split_val]
    test_data = values[split_train + split_val :]
    x_train, y_train = create_time_series_window(
        train_data,
        data_config.input_size,
        data_config.output_size,
    )
    x_val, y_val = create_time_series_window(
        val_data,
        data_config.input_size,
        data_config.output_size,
    )
    x_test, y_test = create_time_series_window(
        test_data,
        data_config.input_size,
        data_config.output_size,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


class DataLoaderS3:
    def __init__(self, data="taxi", data_type="parquet", bucket_name=None):
        """
        Initialise le DataLoaderS3.

        :param data_type: donne le type de données que on veut utiliser (taxi,insee)
        :param bucket_name: Nom du bucket S3 (par défaut, récupéré des variables d'environnement)
        """
        self.data = data.lower()
        self.bucket = bucket_name or os.getenv("MY_BUCKET", "laurinemir")
        self.path = f"s3://{self.bucket}/diffusion"
        self.data_type = data_type
        # Connexion à S3
        self.fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
        )

    def list_files(self):
        """Liste les fichiers .parquet disponibles dans le dossier S3"""
        files = self.fs.ls(self.path)
        return [
            file
            for file in files
            if file.endswith("." + self.data_type) and self.fs.isfile(file)
        ]

    def load_data(self):
        """Charge les fichiers .parquet depuis S3 et applique le bon traitement"""
        files = self.list_files()
        if not files:
            raise ValueError(f"Aucun fichier .parquet trouvé dans {self.path}")
        dfs = []
        for file in files:
            with self.fs.open(file) as f:
                if self.data_type == "parquet":
                    df = pd.read_parquet(f)
                elif self.data_type == "csv":
                    df = pd.read_csv(f)
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        return self.process_data(df)

    def process_data(self, df):
        """Applique un pré-traitement spécifique selon le type de données"""
        if self.data == "taxi":
            return self.process_taxi_data(df)
        elif self.data == "insee":
            return self.process_insee_data(df)
        else:
            raise ValueError("Type de données non reconnu. Utilise 'taxi' ou 'insee'.")

    def process_taxi_data(self, df):
        """Traitement spécifique pour les données taxi"""
        df["tpep_pickup_datetime"] = pd.to_datetime(
            df["tpep_pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
        )
        df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
        df_activity = df.groupby("hour").size().reset_index(name="num_trips")
        df_activity = df_activity[df_activity["num_trips"] >= 100]
        df_activity["hour"] = pd.to_datetime(df_activity["hour"])
        return df_activity

    def process_insee_data(self, df):
        """Traitement spécifique pour les données INSEE"""
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
        colonne = df.columns[0]  # colonne Activite
        df_activity = df[
            (df[colonne] == "L")
            & (df["SEASONAL_ADJUST"] == "Y")
            & (df["IDX_TYPE"] == "ICA_SERV")
        ].sort_values(by="TIME_PERIOD", ascending=True)
        return df_activity

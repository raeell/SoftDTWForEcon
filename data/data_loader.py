"""Data loading utilities."""

from __future__ import annotations

import os

import pandas as pd
import s3fs

LOWER_BOUND_TAXI_TRIPS = 100


class DataLoaderS3:
    """Classe qui load les données et applique un preprocessing éventuel."""

    def __init__(
        self,
        data_name: str = "taxi",
        data_format: str = "parquet",
        bucket_name: str | None = None,
        folder: str | None = None,
    ) -> None:
        """Initialise le DataLoaderS3.

        :param data_name: donne le type de données que on veut utiliser (taxi,insee)
        :param data_format: donne le format des données (parquet, csv)
        :param bucket_name: Nom du bucket S3 (par défaut, récupéré des variables d'env)
        """
        self.data_name = data_name.lower()
        self.bucket = bucket_name or os.getenv("MY_BUCKET", "laurinemir")
        self.path = f"s3://{self.bucket}/{folder}" or f"s3://{self.bucket}/diffusion"
        if data_name == "insee":
            self.path = self.path + "/insee_data"
        self.data_format = data_format
        # Connexion à S3
        self.fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
        )

    def list_files(self) -> list:
        """Liste les fichiers .parquet disponibles dans le dossier S3."""
        files = self.fs.ls(self.path)
        return [
            file
            for file in files
            if file.endswith("." + self.data_format) and self.fs.isfile(file)
        ]

    def load_data(self) -> pd.DataFrame | None:
        """Charge les fichiers .parquet depuis S3 et applique le bon traitement."""
        files = self.list_files()
        if not files:
            msg = f"Aucun fichier .parquet trouvé dans {self.path}"
            raise ValueError(msg)
        dfs = []
        for file in files:
            with self.fs.open(file) as f:
                if self.data_format == "parquet":
                    df_file = pd.read_parquet(f)
                elif self.data_format == "csv":
                    df_file = pd.read_csv(f)
                dfs.append(df_file)

        df_data = pd.concat(dfs, ignore_index=True)
        return self.process_data(df_data)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Applique un pré-traitement spécifique selon le type de données."""
        if self.data_name == "taxi":
            return self.process_taxi_data(df)
        if self.data_name == "insee":
            return self.process_insee_data(df)
        msg = "Type de données non reconnu. Utilise 'taxi' ou 'insee'."
        raise ValueError(msg)

    def process_taxi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traitement spécifique pour les données taxi."""
        df["tpep_pickup_datetime"] = pd.to_datetime(
            df["tpep_pickup_datetime"],
            format="%Y-%m-%d %H:%M:%S",
        )
        df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")
        df_activity = df.groupby("hour").size().reset_index(name="num_trips")
        df_activity = df_activity[df_activity["num_trips"] >= LOWER_BOUND_TAXI_TRIPS]
        df_activity["hour"] = pd.to_datetime(df_activity["hour"])
        return df_activity

    def process_insee_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traitement spécifique pour les données INSEE."""
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
        colonne = df.columns[0]  # colonne Activite
        return df[
            (df[colonne] == "L")
            & (df["SEASONAL_ADJUST"] == "Y")
            & (df["IDX_TYPE"] == "ICA_SERV")
        ].sort_values(by="TIME_PERIOD", ascending=True)

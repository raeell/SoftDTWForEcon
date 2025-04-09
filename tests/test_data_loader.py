"""Data loader tests."""

from dotenv import load_dotenv

from data.data_loader import DataLoaderS3

NUM_TRIPS_0 = 6596


def test_data_loader_s3_taxi() -> None:
    """Test the S3 data loader for taxi data."""
    load_dotenv()
    taxi_loader = DataLoaderS3(
        data_name="taxi",
        data_format="parquet",
        bucket_name="laurinemir",
        folder="diffusion",
    )
    df_taxi = taxi_loader.load_data()
    assert df_taxi.iloc[0]["num_trips"] == NUM_TRIPS_0

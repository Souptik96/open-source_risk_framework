# risk_framework/ingestion/gcp_loader.py

from google.cloud import storage
import pandas as pd
from io import StringIO
import os

class GCPLoader:
    def __init__(self, bucket_name: str, blob_name: str, credentials_path: str = None):
        """
        Initialize the GCPLoader.

        :param bucket_name: GCS bucket name
        :param blob_name: Path to the CSV blob in the bucket
        :param credentials_path: Optional path to service account credentials JSON
        """
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self.bucket_name = bucket_name
        self.blob_name = blob_name
        self.client = storage.Client()

    def load_csv(self) -> pd.DataFrame:
        """
        Load a CSV file from GCS into a pandas DataFrame.

        :return: pandas DataFrame
        :raises Exception: if loading fails
        """
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(self.blob_name)
            csv_data = blob.download_as_text()
            df = pd.read_csv(StringIO(csv_data))
            print(f"Loaded CSV from GCS with shape: {df.shape}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV from gs://{self.bucket_name}/{self.blob_name}") from e


# Example usage
if __name__ == "__main__":
    loader = GCPLoader(bucket_name="my-risk-bucket", blob_name="financial/transactions.csv", credentials_path="path/to/credentials.json")
    df = loader.load_csv()
    print(df.head())

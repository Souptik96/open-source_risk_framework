# risk_framework/ingestion/azure_loader.py

from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO

class AzureLoader:
    def __init__(self, connection_string: str, container_name: str, blob_name: str):
        """
        Initialize the AzureLoader.

        :param connection_string: Azure Blob Storage connection string
        :param container_name: Name of the blob container
        :param blob_name: Path to the CSV blob within the container
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_name = blob_name
        self.client = BlobServiceClient.from_connection_string(self.connection_string)

    def load_csv(self) -> pd.DataFrame:
        """
        Load a CSV file from Azure Blob Storage into a pandas DataFrame.

        :return: pandas DataFrame
        :raises RuntimeError: if loading fails
        """
        try:
            container_client = self.client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)
            csv_data = blob_client.download_blob().readall().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
            print(f"Loaded CSV from Azure Blob Storage with shape: {df.shape}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV from Azure container '{self.container_name}', blob '{self.blob_name}'") from e


# Example usage
if __name__ == "__main__":
    loader = AzureLoader(
        connection_string="your_connection_string",
        container_name="risk-data",
        blob_name="financial/transactions.csv"
    )
    df = loader.load_csv()
    print(df.head())

# risk_framework/ingestion/aws_loader.py

import boto3
import pandas as pd
from io import StringIO

class AWSLoader:
    def __init__(self, bucket_name: str, object_key: str, region_name: str = "us-east-1"):
        """
        Initialize the AWSLoader.

        :param bucket_name: S3 bucket name
        :param object_key: Path to the CSV file in the bucket
        :param region_name: AWS region (default: us-east-1)
        """
        self.bucket_name = bucket_name
        self.object_key = object_key
        self.region_name = region_name
        self.s3_client = boto3.client("s3", region_name=self.region_name)

    def load_csv(self) -> pd.DataFrame:
        """
        Load a CSV file from S3 into a pandas DataFrame.

        :return: pandas DataFrame
        :raises Exception: if loading fails
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.object_key)
            csv_data = response["Body"].read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_data))
            print(f"Loaded CSV from S3 with shape: {df.shape}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV from s3://{self.bucket_name}/{self.object_key}") from e


# Example usage
if __name__ == "__main__":
    loader = AWSLoader(bucket_name="my-risk-data", object_key="financial/transactions.csv")
    df = loader.load_csv()
    print(df.head())

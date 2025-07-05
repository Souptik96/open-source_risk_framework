# risk_framework/ingestion/csv_loader.py

import pandas as pd
from typing import List, Optional

class CSVLoader:
    def __init__(self, filepath: str, required_columns: Optional[List[str]] = None):
        """
        Initialize the CSVLoader with file path and optional schema.

        :param filepath: Path to the CSV file
        :param required_columns: List of required column names
        """
        self.filepath = filepath
        self.required_columns = required_columns
        self.data = None

    def load(self) -> pd.DataFrame:
        """
        Load the CSV file into a DataFrame.
        
        :return: pandas DataFrame
        :raises FileNotFoundError: if the file does not exist
        :raises ValueError: if required columns are missing
        """
        try:
            df = pd.read_csv(self.filepath)
            print(f"Loaded data with shape: {df.shape}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found at: {self.filepath}") from e

        if self.required_columns:
            missing = [col for col in self.required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        self.data = df
        return df

    def preview(self, rows: int = 5):
        """Print a preview of the loaded data."""
        if self.data is not None:
            print(self.data.head(rows))
        else:
            print("No data loaded. Please call `load()` first.")


# Example usage
if __name__ == "__main__":
    loader = CSVLoader("example_data/transactions.csv", required_columns=["id", "amount", "country"])
    df = loader.load()
    loader.preview()

# risk_framework/ingestion/__init__.py

from .csv_loader import CSVLoader
from .aws_loader import AWSLoader
from .gcp_loader import GCPLoader
from .azure_loader import AzureLoader

__all__ = [
    "CSVLoader",
    "AWSLoader",
    "GCPLoader",
    "AzureLoader"
]

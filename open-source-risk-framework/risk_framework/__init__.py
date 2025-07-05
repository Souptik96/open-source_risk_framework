# risk_framework/__init__.py

__version__ = "0.1.0"

# Expose submodules for ease of access
from . import ingestion
from . import models
from . import evaluation
from . import reporting
from . import mcp

__all__ = [
    "ingestion",
    "models",
    "evaluation",
    "reporting",
    "mcp"
]

"""
Risk Framework Utilities Module

This module provides essential utilities for data preprocessing and risk metrics calculation.
It serves as the foundation for the Open-Source Risk Framework.

Exported functionality:
- DataPreprocessing: Comprehensive data cleaning and transformation tools
- RiskMetrics: Advanced risk measurement and performance metrics
- ScalingMethod: Enum for feature scaling methods
- MissingValueStrategy: Enum for missing data handling strategies
- VaRMethod: Enum for Value-at-Risk calculation approaches
"""

from .data_preprocessing import (
    DataPreprocessing,
    ScalingMethod,
    MissingValueStrategy
)
from .risk_metrics import (
    RiskMetrics,
    VaRMethod
)

__all__ = [
    'DataPreprocessing',
    'RiskMetrics',
    'ScalingMethod',
    'MissingValueStrategy',
    'VaRMethod'
]

# Package version
__version__ = '0.1.0'

# Module documentation
__doc__ = """
Open-Source Risk Framework - Utilities Module

The utils module provides core functionality for:

1. Data Preprocessing:
   - Missing value handling (mean, median, mode, drop, fill)
   - Feature scaling (standard, minmax, robust)
   - Outlier detection (IQR, z-score)
   - Time feature engineering
   - Data validation

2. Risk Metrics:
   - Value-at-Risk (VaR) - historical, parametric, Monte Carlo
   - Expected Shortfall (CVaR)
   - Risk-adjusted performance (Sharpe, Sortino ratios)
   - Drawdown analysis (max drawdown, ulcer index)
   - Tail risk measures

Example Usage:
--------------
from risk_framework.models.utils import DataPreprocessing, RiskMetrics

# Data preprocessing
preprocessor = DataPreprocessing()
clean_data = preprocessor.clean_missing(raw_data, strategy='median')
scaled_data = preprocessor.scale_features(clean_data, method='standard')

# Risk metrics
returns = scaled_data['portfolio_returns']
var = RiskMetrics.value_at_risk(returns, method='historical')
es = RiskMetrics.expected_shortfall(returns)
sharpe = RiskMetrics.sharpe_ratio(returns, annualize=True)
"""

# Add module-level convenience functions
def get_version() -> str:
    """Return the current version of the utils module"""
    return __version__

def list_supported_methods() -> dict:
    """Return a dictionary of all supported methods in the utils module"""
    return {
        'scaling_methods': [method.value for method in ScalingMethod],
        'missing_value_strategies': [strategy.value for strategy in MissingValueStrategy],
        'var_methods': [method.value for method in VaRMethod]
    }

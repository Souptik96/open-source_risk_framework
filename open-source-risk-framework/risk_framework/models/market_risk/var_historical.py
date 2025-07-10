# var_historical.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class HistoricalVaR:
    """
    Enhanced Historical Value-at-Risk (VaR) Calculator with:
    - Multi-asset portfolio support
    - Rolling window backtesting
    - VaR breach analysis
    - Serialization capabilities
    """

    def __init__(self, 
                confidence_level: float = 0.95,
                lookback_window: int = 252,
                method: str = 'percentile'):
        """
        Args:
            confidence_level: VaR confidence level (0-1)
            lookback_window: Historical window size (days)
            method: Calculation method ('percentile' or 'empirical')
        """
        self.confidence_level = confidence_level
        self.lookback_window = lookback_window
        self.method = method
        self._validate_params()
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None
        }

    def _validate_params(self):
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")

    def calculate(self,
                 returns: pd.DataFrame,
                portfolio_weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate VaR for portfolio
        
        Args:
            returns: DataFrame of asset returns (columns=assets)
            portfolio_weights: Dict of asset weights (default equal-weighted)
            
        Returns:
            Dictionary containing VaR metrics
        """
        if not portfolio_weights:
            portfolio_weights = {col: 1/len(returns.columns) for col in returns.columns}
        
        # Calculate portfolio returns
        port_returns = (returns * pd.Series(portfolio_weights)).sum(axis=1)
        
        # Select lookback window
        recent_returns = port_returns.iloc[-self.lookback_window:]
        
        if self.method == 'percentile':
            var = self._percentile_method(recent_returns)
        else:
            var = self._empirical_method(recent_returns)
        
        self.metadata.update({
            'last_updated': datetime.now().isoformat(),
            'calculation': {
                'method': self.method,
                'assets': list(portfolio_weights.keys()),
                'window_used': len(recent_returns)
            }
        })
        
        return {
            'var': var,
            'confidence_level': self.confidence_level,
            'lookback_period': self.lookback_window
        }

    def _percentile_method(self, returns: pd.Series) -> float:
        """Standard percentile-based VaR"""
        return np.percentile(returns, 100 * (1 - self.confidence_level))

    def _empirical_method(self, returns: pd.Series) -> float:
        """Empirical distribution VaR"""
        sorted_returns = returns.sort_values()
        index = int((1 - self.confidence_level) * len(sorted_returns))
        return sorted_returns.iloc[index]

    def backtest(self,
                returns: pd.Series,
                window: int = 252) -> Dict:
        """
        Perform rolling window backtest
        
        Args:
            returns: Series of portfolio returns
            window: Backtesting window size
            
        Returns:
            Dictionary with backtest results
        """
        var_series = []
        breaches = 0
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            var = self._percentile_method(window_returns)
            var_series.append(var)
            
            if returns.iloc[i] < var:
                breaches += 1
        
        breach_ratio = breaches / (len(returns) - window)
        
        self.metadata['backtest'] = {
            'breaches': breaches,
            'breach_ratio': breach_ratio,
            'expected_breaches': (1 - self.confidence_level) * (len(returns) - window)
        }
        
        return {
            'var_series': pd.Series(var_series, index=returns.index[window:]),
            'breach_ratio': breach_ratio
        }

    def save_config(self, file_path: str):
        """Save model configuration"""
        with open(file_path, 'w') as f:
            json.dump({
                'params': {
                    'confidence_level': self.confidence_level,
                    'lookback_window': self.lookback_window,
                    'method': self.method
                },
                'metadata': self.metadata
            }, f, indent=2)

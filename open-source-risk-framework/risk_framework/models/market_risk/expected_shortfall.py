# expected_shortfall.py
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
from scipy.stats import genpareto
from datetime import datetime

logger = logging.getLogger(__name__)

class ExpectedShortfall:
    """
    Advanced Expected Shortfall Calculator with:
    - Extreme Value Theory (EVT) integration
    - Conditional risk estimation
    - Multi-period forecasting
    - Backtesting framework
    """

    def __init__(self,
                confidence_level: float = 0.975,
                evt_threshold: float = 0.95):
        """
        Args:
            confidence_level: ES confidence level
            evt_threshold: Threshold for EVT tail estimation
        """
        self.confidence_level = confidence_level
        self.evt_threshold = evt_threshold
        self._validate_params()

    def _validate_params(self):
        if not 0 < self.evt_threshold < self.confidence_level < 1:
            raise ValueError("Invalid threshold/confidence level combination")

    def calculate(self,
                returns: pd.Series,
                method: str = 'historical') -> Dict:
        """
        Calculate Expected Shortfall
        
        Args:
            returns: Series of returns
            method: Calculation method ('historical' or 'evt')
            
        Returns:
            Dictionary with ES metrics
        """
        if method == 'historical':
            return self._historical_es(returns)
        elif method == 'evt':
            return self._evt_es(returns)
        else:
            raise ValueError("Invalid method")

    def _historical_es(self, returns: pd.Series) -> Dict:
        """Standard historical ES"""
        var = np.percentile(returns, 100 * (1 - self.confidence_level))
        tail_losses = returns[returns <= var]
        es = tail_losses.mean()
        
        return {
            'es': es,
            'var': var,
            'tail_observations': len(tail_losses),
            'method': 'historical'
        }

    def _evt_es(self, returns: pd.Series) -> Dict:
        """EVT-based ES using Generalized Pareto Distribution"""
        var = np.percentile(returns, 100 * (1 - self.confidence_level))
        threshold = np.percentile(returns, 100 * (1 - self.evt_threshold))
        excess = returns[returns <= threshold] - threshold
        
        if len(excess) < 10:
            logger.warning("Insufficient tail observations for EVT")
            return self._historical_es(returns)
            
        shape, loc, scale = genpareto.fit(excess, floc=0)
        es = threshold + (scale + shape * (threshold - var)) / (1 - shape)
        
        return {
            'es': es,
            'var': var,
            'gpd_params': {'shape': shape, 'scale': scale},
            'tail_observations': len(excess),
            'method': 'evt'
        }

    def backtest(self,
                returns: pd.Series,
                var_model: HistoricalVaR,
                window: int = 252) -> Dict:
        """
        Backtest ES against VaR breaches
        
        Args:
            returns: Series of returns
            var_model: HistoricalVaR instance
            window: Backtesting window size
            
        Returns:
            Dictionary with backtest results
        """
        var_results = var_model.backtest(returns, window)
        breach_indices = returns[window:] < var_results['var_series']
        breach_returns = returns[window:][breach_indices]
        
        if len(breach_returns) == 0:
            return {'status': 'no_breaches'}
            
        avg_breach = breach_returns.mean()
        es_series = var_results['var_series'].apply(
            lambda x: self.calculate(returns[returns <= x])['es']
        )
        
        return {
            'average_breach_loss': avg_breach,
            'es_series': es_series,
            'breach_count': len(breach_returns)
        }

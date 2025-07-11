# merton_model.py
import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MertonCreditRisk:
    """
    Enhanced Merton Structural Model with:
    - Distance-to-default calculation
    - Term structure modeling
    - Asset volatility estimation
    - Credit spread derivation
    """

    def __init__(self,
                equity_value: float,
                equity_volatility: float,
                debt: float,
                risk_free_rate: float = 0.05,
                time_horizons: List[float] = [1, 3, 5]):
        """
        Args:
            equity_value: Market value of equity
            equity_volatility: Annualized equity volatility
            debt: Total debt (face value)
            risk_free_rate: Annual risk-free rate
            time_horizons: List of time horizons in years
        """
        self.E = equity_value
        self.sigma_E = equity_volatility
        self.D = debt
        self.r = risk_free_rate
        self.T_list = sorted(time_horizons)
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': 'Merton Structural'
        }

    def calculate_asset_volatility(self, max_iter=100, tol=1e-6) -> float:
        """
        Estimate asset volatility using iterative method
        """
        V = self.E + self.D  # Initial guess
        sigma = self.sigma_E * self.E / V
        
        for _ in range(max_iter):
            d1 = (np.log(V/self.D) + (self.r + 0.5*sigma**2)*1) / (sigma*np.sqrt(1))
            new_V = self.E + self.D * np.exp(-self.r*1) * norm.cdf(d1 - sigma*np.sqrt(1))
            
            if abs(new_V - V) < tol:
                break
                
            V = new_V
            sigma = self.sigma_E * self.E / V
        
        self.V = V
        self.sigma = sigma
        return sigma

    def distance_to_default(self, T: float) -> float:
        """
        Calculate distance-to-default for given horizon
        """
        if not hasattr(self, 'V'):
            self.calculate_asset_volatility()
            
        d2 = (np.log(self.V/self.D) + (self.r - 0.5*self.sigma**2)*T) / (self.sigma*np.sqrt(T))
        return d2

    def probability_of_default(self) -> Dict[float, float]:
        """
        Calculate term structure of default probabilities
        """
        return {
            T: norm.cdf(-self.distance_to_default(T))
            for T in self.T_list
        }

    def credit_spread(self, T: float) -> float:
        """
        Calculate credit spread for given maturity
        """
        pd = self.probability_of_default()[T]
        return -np.log(1 - pd) / T - self.r

    def save_state(self, file_path: str):
        """Save model state"""
        with open(file_path, 'w') as f:
            json.dump({
                'parameters': {
                    'equity_value': self.E,
                    'equity_volatility': self.sigma_E,
                    'debt': self.D,
                    'risk_free_rate': self.r
                },
                'calculated': {
                    'asset_value': getattr(self, 'V', None),
                    'asset_volatility': getattr(self, 'sigma', None)
                },
                'metadata': self.metadata
            }, f, indent=2)

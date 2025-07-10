# garch_volatility.py
import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, Optional
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class GARCHModel:
    """
    Enhanced GARCH Volatility Model with:
    - Multiple GARCH variants (GARCH, EGARCH, GJRGARCH)
    - Volatility clustering analysis
    - Conditional forecasting
    - Model diagnostics
    """

    def __init__(self,
                model_type: str = 'GARCH',
                p: int = 1,
                q: int = 1,
                dist: str = 'normal'):
        """
        Args:
            model_type: GARCH variant ('GARCH', 'EGARCH', 'GJRGARCH')
            p: Lag order of volatility
            q: Lag order of shocks
            dist: Error distribution ('normal', 't', 'skewt')
        """
        self.model_type = model_type
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.results = None
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_fit': None
        }

    def fit(self,
           returns: pd.Series,
           mean: str = 'constant') -> Dict:
        """
        Fit GARCH model to returns
        
        Args:
            returns: Series of asset returns
            mean: Mean model ('constant', 'zero', 'AR', 'LS')
            
        Returns:
            Dictionary with model summary
        """
        self.model = arch_model(
            returns,
            mean=mean,
            vol=self.model_type,
            p=self.p,
            q=self.q,
            dist=self.dist
        )
        
        self.results = self.model.fit(disp='off')
        self.metadata.update({
            'last_fit': datetime.now().isoformat(),
            'model_params': self.results.params.to_dict(),
            'diagnostics': {
                'log_likelihood': self.results.loglikelihood,
                'aic': self.results.aic,
                'bic': self.results.bic
            }
        })
        
        return self.results.summary()

    def forecast(self,
               horizon: int = 5,
               start_date: Optional[str] = None) -> Dict:
        """
        Generate volatility forecasts
        
        Args:
            horizon: Forecast horizon (days)
            start_date: Optional anchor date
            
        Returns:
            Dictionary with forecast results
        """
        if not self.results:
            raise ValueError("Model not fitted")
            
        forecast = self.results.forecast(
            horizon=horizon,
            start=start_date
        )
        
        return {
            'variance': forecast.variance.iloc[-1].values,
            'mean': forecast.mean.iloc[-1].values,
            'residuals': forecast.residuals.iloc[-1].values
        }

    def plot_volatility(self) -> plt.Figure:
        """Plot conditional volatility"""
        if not self.results:
            raise ValueError("Model not fitted")
            
        fig, ax = plt.subplots(figsize=(12, 6))
        self.results.plot(annualize='D', ax=ax)
        ax.set_title(f"{self.model_type}({self.p},{self.q}) Conditional Volatility")
        return fig

    def save_model(self, file_path: str):
        """Save model configuration"""
        with open(file_path, 'w') as f:
            json.dump({
                'spec': {
                    'model_type': self.model_type,
                    'p': self.p,
                    'q': self.q,
                    'dist': self.dist
                },
                'params': self.results.params.to_dict(),
                'metadata': self.metadata
            }, f, indent=2)

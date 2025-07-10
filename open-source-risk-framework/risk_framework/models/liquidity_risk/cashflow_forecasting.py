# cashflow_forecasting.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from statsmodels.tsa.api import ExponentialSmoothing, VAR
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class CashflowForecaster:
    """
    Advanced Cash Flow Forecasting with:
    - Multiple forecasting methods
    - Anomaly detection
    - Scenario analysis
    - Liquidity gap analysis
    """

    def __init__(self):
        self.models = {}
        self.anomaly_detector = None
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_trained': None
        }

    def fit(self,
           cashflows: pd.DataFrame,
           method: str = 'holt-winters',
           freq: str = 'D') -> Dict:
        """
        Train forecasting model
        
        Args:
            cashflows: DataFrame with 'date' and 'amount' columns
            method: Forecasting method ('holt-winters', 'var')
            freq: Data frequency ('D', 'W', 'M')
        """
        cashflows = cashflows.set_index('date').asfreq(freq)
        
        if method == 'holt-winters':
            model = ExponentialSmoothing(
                cashflows['amount'],
                trend='add',
                seasonal='add' if len(cashflows) > 2*365 else None,
                initialization_method="estimated"
            ).fit()
        elif method == 'var':
            model = VAR(cashflows).fit()
        else:
            raise ValueError("Unsupported method")
        
        self.models[method] = model
        self.metadata.update({
            'last_trained': datetime.now().isoformat(),
            'active_model': method,
            'stats': {
                'mean': cashflows['amount'].mean(),
                'volatility': cashflows['amount'].std()
            }
        })
        
        # Train anomaly detector
        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.anomaly_detector.fit(cashflows.values.reshape(-1, 1))
        
        return model.summary()

    def forecast(self,
               periods: int = 30,
               scenario: Optional[Dict] = None) -> Dict:
        """
        Generate cash flow forecast
        
        Args:
            periods: Number of periods to forecast
            scenario: Optional scenario adjustments
            
        Returns:
            Dictionary with forecast results
        """
        if not self.models:
            raise ValueError("No model trained")
            
        model = self.models[self.metadata['active_model']]
        
        if isinstance(model, ExponentialSmoothing):
            point_forecast = model.forecast(periods)
            conf_int = model.get_prediction(
                start=point_forecast.index[0],
                end=point_forecast.index[-1]
            ).conf_int()
        else:
            point_forecast = model.forecast(model.y, steps=periods)
            # VAR-specific processing
        
        # Apply scenario adjustments
        if scenario:
            point_forecast = self._apply_scenario(point_forecast, scenario)
        
        return {
            'forecast': point_forecast,
            'confidence_interval': conf_int if 'conf_int' in locals() else None,
            'anomaly_score': self._detect_anomalies(point_forecast)
        }

    def _apply_scenario(self, forecast: pd.Series, scenario: Dict) -> pd.Series:
        """Apply stress scenario adjustments"""
        if scenario['type'] == 'percentage':
            return forecast * (1 + scenario['adjustment'])
        elif scenario['type'] == 'absolute':
            return forecast + scenario['adjustment']
        else:
            raise ValueError("Unknown scenario type")

    def _detect_anomalies(self, forecast: pd.Series) -> pd.Series:
        """Flag anomalous forecasts"""
        scores = self.anomaly_detector.decision_function(
            forecast.values.reshape(-1, 1)
        )
        return pd.Series(scores, index=forecast.index, name='anomaly_score')

    def liquidity_gap_analysis(self,
                             forecast: Dict,
                             liabilities: pd.Series) -> Dict:
        """
        Calculate liquidity gaps
        
        Args:
            forecast: Output from forecast() method
            liabilities: Expected liabilities
            
        Returns:
            Dictionary with gap analysis
        """
        net_cash = forecast['forecast'] - liabilities
        return {
            'daily_gap': net_cash,
            'cumulative_gap': net_cash.cumsum(),
            'stress_periods': net_cash[net_cash < 0].index
        }

    def plot_forecast(self, 
                     forecast: Dict, 
                     historical: Optional[pd.DataFrame] = None) -> plt.Figure:
        """Visualize cash flow forecast"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if historical is not None:
            historical['amount'].plot(ax=ax, label='Historical')
        
        forecast['forecast'].plot(ax=ax, label='Forecast')
        
        if 'confidence_interval' in forecast:
            ax.fill_between(
                forecast['forecast'].index,
                forecast['confidence_interval'].iloc[:, 0],
                forecast['confidence_interval'].iloc[:, 1],
                alpha=0.2
            )
        
        ax.set_title("Cash Flow Forecast")
        ax.legend()
        return fig

    def save_model(self, file_path: str):
        """Save forecasting model"""
        with open(file_path, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'model_type': self.metadata['active_model'],
                'anomaly_threshold': -0.5  # Example threshold
            }, f, indent=2)

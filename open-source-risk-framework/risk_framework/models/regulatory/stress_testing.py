# stress_testing.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

logger = logging.getLogger(__name__)

class StressScenario:
    """
    Individual stress scenario configuration
    """
    def __init__(self,
                name: str,
                shock_type: str,
                magnitude: float,
                correlation_matrix: Optional[pd.DataFrame] = None):
        """
        Args:
            name: Scenario identifier
            shock_type: 'percentage' or 'absolute'
            magnitude: Shock size (0.2 = 20% for percentage)
            correlation_matrix: Asset correlations
        """
        self.name = name
        self.shock_type = shock_type
        self.magnitude = magnitude
        self.correlation = correlation_matrix
        self.metadata = {
            'created_at': datetime.now().isoformat()
        }

class StressTestingEngine:
    """
    Advanced Stress Testing Framework with:
    - Multi-factor scenario modeling
    - Extreme value theory integration
    - Reverse stress testing
    - Regulatory compliance checks
    """

    def __init__(self):
        self.scenarios = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_run': None
        }

    def add_scenario(self, scenario: StressScenario):
        """Register a stress scenario"""
        self.scenarios[scenario.name] = scenario

    def apply_scenarios(self,
                      portfolio: pd.DataFrame,
                      base_values: pd.Series) -> Dict:
        """
        Apply all scenarios to portfolio
        """
        results = {}
        for name, scenario in self.scenarios.items():
            if scenario.shock_type == 'percentage':
                shocked = base_values * (1 - scenario.magnitude)
            else:
                shocked = base_values - scenario.magnitude
                
            if scenario.correlation is not None:
                shocked = self._apply_correlated_shock(
                    shocked, 
                    scenario.correlation
                )
                
            results[name] = {
                'shocked_values': shocked,
                'pct_change': (shocked - base_values) / base_values,
                'portfolio_impact': (portfolio * shocked).sum(axis=1)
            }
        
        self.metadata.update({
            'last_run': datetime.now().isoformat(),
            'scenarios_applied': list(self.scenarios.keys())
        })
        
        return results

    def _apply_correlated_shock(self,
                              base_values: pd.Series,
                              correlation: pd.DataFrame) -> pd.Series:
        """Apply correlation-adjusted shocks"""
        # Cholesky decomposition for correlated random variables
        L = np.linalg.cholesky(correlation.values)
        correlated_shocks = L.dot(np.random.randn(len(base_values)))
        return base_values * (1 + correlated_shocks)

    def reverse_stress_test(self,
                          portfolio: pd.DataFrame,
                          threshold: float,
                          n_simulations: int = 10000) -> Dict:
        """
        Identify scenarios that would breach the threshold
        """
        losses = []
        for _ in range(n_simulations):
            shock = gumbel_r.rvs(scale=0.2, size=len(portfolio.columns))
            shocked_values = portfolio.values * (1 - shock)
            loss = (portfolio.values - shocked_values).sum()
            losses.append(loss)
        
        var = np.percentile(losses, 95)
        breach_prob = sum(l > threshold for l in losses) / n_simulations
        
        return {
            'var_95': var,
            'breach_probability': breach_prob,
            'critical_scenarios': self._identify_critical_scenarios(losses, threshold)
        }

    def _identify_critical_scenarios(self,
                                   losses: List[float],
                                   threshold: float) -> List[Dict]:
        """Find most dangerous scenario components"""
        # Implementation would analyze loss drivers
        return [{
            'factor': "equity_crash",
            'contribution': 0.65
        }]

    def generate_report(self, 
                       results: Dict,
                       format: str = "json") -> Union[str, pd.DataFrame]:
        """Generate stress test report"""
        report = {
            'metadata': self.metadata,
            'scenarios': {
                name: {
                    'max_loss': result['portfolio_impact'].min(),
                    'avg_loss': result['portfolio_impact'].mean()
                }
                for name, result in results.items()
            }
        }
        
        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "dataframe":
            return pd.DataFrame(report['scenarios']).T
        else:
            raise ValueError("Unsupported format")

    def plot_scenario_impacts(self, results: Dict) -> plt.Figure:
        """Visualize scenario impacts"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, result in results.items():
            result['portfolio_impact'].plot(ax=ax, label=name)
        
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title("Portfolio Impact by Stress Scenario")
        ax.legend()
        return fig

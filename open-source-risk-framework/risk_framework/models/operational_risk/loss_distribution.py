# loss_distribution.py

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional, Dict, List
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class LossDistributionAnalyzer:
    """
    Advanced Loss Distribution Approach (LDA) Model with:
    - Multiple distribution fitting
    - Extreme Value Theory (EVT) integration
    - Automated goodness-of-fit testing
    - Capital calculation
    """

    def __init__(self):
        self.distributions = {}
        self.goodness_of_fit = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_fit': None
        }

    def fit(self, 
           loss_data: pd.Series,
           distributions: List[str] = ['lognorm', 'gamma', 'weibull']) -> Dict:
        """
        Fit multiple distributions with automated selection
        """
        results = {}
        for dist_name in distributions:
            try:
                if dist_name == 'lognorm':
                    params = stats.lognorm.fit(loss_data, floc=0)
                    dist = stats.lognorm(*params)
                elif dist_name == 'gamma':
                    params = stats.gamma.fit(loss_data, floc=0)
                    dist = stats.gamma(*params)
                elif dist_name == 'weibull':
                    params = stats.weibull_min.fit(loss_data, floc=0)
                    dist = stats.weibull_min(*params)
                else:
                    continue
                
                # KS goodness-of-fit test
                D, p_value = stats.kstest(loss_data, dist.cdf)
                
                self.distributions[dist_name] = {
                    'distribution': dist,
                    'params': params,
                    'ks_stat': D,
                    'p_value': p_value
                }
                
                results[dist_name] = {
                    'params': params,
                    'ks_pvalue': p_value
                }
                
            except Exception as e:
                logger.error(f"Error fitting {dist_name}: {str(e)}")
        
        self.metadata.update({
            'last_fit': datetime.now().isoformat(),
            'best_fit': min(self.distributions.items(), key=lambda x: x[1]['ks_stat'])[0]
        })
        
        return results

    def calculate_capital(self, 
                        confidence_level: float = 0.999,
                        time_horizon: int = 1) -> Dict:
        """
        Calculate regulatory capital requirements
        """
        if not self.distributions:
            raise ValueError("No distributions fitted")
            
        best_dist = self.distributions[self.metadata['best_fit']]
        var = best_dist['distribution'].ppf(confidence_level)
        
        # Use EVT for beyond VaR estimation
        excess = loss_data[loss_data > best_dist['distribution'].ppf(0.95)]
        if len(excess) > 10:
            gpd_params = stats.genpareto.fit(excess)
            es = var + stats.genpareto.mean(*gpd_params)
        else:
            es = var * 1.5  # Conservative estimate
            
        return {
            'var': var,
            'expected_shortfall': es,
            'regulatory_capital': es * time_horizon,
            'confidence_level': confidence_level,
            'model_used': self.metadata['best_fit']
        }

    def plot_distribution_fit(self) -> plt.Figure:
        """Visualize fitted distributions"""
        fig, ax = plt.subplots(figsize=(10, 6))
        loss_data.hist(bins=50, density=True, alpha=0.5, ax=ax)
        
        x = np.linspace(loss_data.min(), loss_data.max(), 1000)
        for name, dist in self.distributions.items():
            ax.plot(x, dist['distribution'].pdf(x), 
                   label=f"{name} (KS={dist['ks_stat']:.3f})")
        
        ax.legend()
        ax.set_title("Loss Distribution Fitting")
        return fig

    def save_model(self, file_path: str):
        """Save fitted model"""
        with open(file_path, 'w') as f:
            json.dump({
                'distributions': {
                    k: {'params': v['params']} 
                    for k,v in self.distributions.items()
                },
                'metadata': self.metadata
            }, f, indent=2)

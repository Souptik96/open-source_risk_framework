# scenario_analysis.py

import pandas as pd
from typing import Dict, List, Optional
import numpy as np
import logging
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

class ScenarioAnalyzer:
    """
    Advanced Scenario Analysis with:
    - Probabilistic scenario modeling
    - Dependency mapping between risks
    - Impact cascading
    - Regulatory capital integration
    """

    def __init__(self):
        self.scenarios = {}
        self.dependencies = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_run': None
        }

    def add_scenario(self, 
                   name: str,
                   base_impact: float,
                   likelihood: float,
                   triggers: Optional[List[str]] = None):
        """
        Add scenario with probabilistic parameters
        """
        self.scenarios[name] = {
            'base_impact': base_impact,
            'likelihood': likelihood,
            'triggers': triggers or []
        }
        
        # Build dependency graph
        for trigger in triggers or []:
            self.dependencies.setdefault(trigger, []).append(name)

    def evaluate(self, 
               monte_carlo_runs: int = 10000,
               correlation_matrix: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run probabilistic scenario analysis
        """
        results = {}
        scenario_names = list(self.scenarios.keys())
        n_scenarios = len(scenario_names)
        
        # Generate correlated random numbers if matrix provided
        if correlation_matrix is not None:
            rng = np.random.default_rng()
            uniforms = rng.multivariate_normal(
                mean=np.zeros(n_scenarios),
                cov=correlation_matrix,
                size=monte_carlo_runs
            )
            uniforms = stats.norm.cdf(uniforms)
        else:
            uniforms = np.random.uniform(size=(monte_carlo_runs, n_scenarios))
        
        # Monte Carlo simulation
        impacts = np.zeros((monte_carlo_runs, n_scenarios))
        for i, name in enumerate(scenario_names):
            scenario = self.scenarios[name]
            occurs = uniforms[:,i] < scenario['likelihood']
            impacts[occurs,i] = scenario['base_impact']
            
            # Handle triggered scenarios
            for trigger in scenario['triggers']:
                if trigger in self.scenarios:
                    j = scenario_names.index(trigger)
                    impacts[occurs,j] += self.scenarios[trigger]['base_impact'] * 0.5  # Partial amplification
        
        # Aggregate results
        total_impacts = impacts.sum(axis=1)
        results['total_loss_distribution'] = {
            'mean': np.mean(total_impacts),
            'var_95': np.percentile(total_impacts, 95),
            'var_99': np.percentile(total_impacts, 99)
        }
        
        # Scenario-specific stats
        for i, name in enumerate(scenario_names):
            scenario_impacts = impacts[:,i]
            active_runs = scenario_impacts > 0
            results[name] = {
                'activation_rate': np.mean(active_runs),
                'conditional_impact': np.mean(scenario_impacts[active_runs]) if any(active_runs) else 0,
                'contribution_to_total': np.mean(scenario_impacts) / results['total_loss_distribution']['mean']
            }
        
        self.metadata['last_run'] = datetime.now().isoformat()
        return results

    def load_scenarios(self, config_path: str):
        """Load scenarios from YAML config"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            for scenario in config['scenarios']:
                self.add_scenario(**scenario)
            self.metadata.update({
                'config_source': config_path,
                'scenario_count': len(self.scenarios)
            })

    def plot_impact_distribution(self, results: Dict) -> plt.Figure:
        """Visualize loss distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(results['total_loss_distribution'], bins=50)
        ax.axvline(results['var_95'], color='orange', label='95% VaR')
        ax.axvline(results['var_99'], color='red', label='99% VaR')
        ax.legend()
        return fig

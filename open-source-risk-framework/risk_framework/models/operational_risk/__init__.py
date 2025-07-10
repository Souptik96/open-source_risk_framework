"""
Operational Risk Modeling Module

Provides:
- Rule-based operational risk flagging
- Loss Distribution Approach (LDA) modeling
- Scenario analysis for stress testing
"""

from .rule_based import OperationalRiskEngine
from .loss_distribution import LossDistributionAnalyzer
from .scenario_analysis import ScenarioAnalyzer
from typing import Dict, List, Optional
import logging
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "2.1.0"
__all__ = [
    'OperationalRiskEngine',
    'LossDistributionAnalyzer',
    'ScenarioAnalyzer',
    'load_standard_scenarios'
]

def load_standard_scenarios(config_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load pre-configured operational risk scenarios from YAML
    
    Args:
        config_path: Optional custom path to scenario config
        
    Returns:
        Dictionary of scenario configurations
        
    Example:
        >>> scenarios = load_standard_scenarios()
        >>> analyzer = ScenarioAnalyzer()
        >>> for name, params in scenarios.items():
        ...     analyzer.add_scenario(name, **params)
    """
    default_config = Path(__file__).parent / "config/standard_scenarios.yaml"
    path = Path(config_path) if config_path else default_config
    
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load scenarios: {str(e)}")
        raise

# Initialize default rule set
_DEFAULT_RULES = [
    {
        'name': 'high_value_transaction',
        'function': lambda df: df['amount'] > 100000,
        'category': 'financial',
        'weight': 1.5,
        'severity': 'HIGH'
    },
    {
        'name': 'after_hours_activity',
        'function': lambda df: ~df['timestamp'].dt.time.between(pd.Time(9,0), pd.Time(17,0)),
        'category': 'temporal',
        'weight': 0.8,
        'severity': 'MEDIUM'
    }
]

def create_default_engine() -> OperationalRiskEngine:
    """
    Factory method for pre-configured risk engine
    
    Returns:
        OperationalRiskEngine with common rules pre-loaded
    """
    engine = OperationalRiskEngine()
    for rule in _DEFAULT_RULES:
        engine.add_rule(**rule)
    return engine

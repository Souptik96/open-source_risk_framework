# rule_based.py

import pandas as pd
from typing import List, Dict, Callable, Optional
import yaml
from pathlib import Path
import logging
from enum import Enum
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class OperationalRiskEngine:
    """
    Advanced Rule-Based Operational Risk Detection with:
    - Dynamic rule weighting
    - Temporal rule evaluation
    - Multi-level severity classification
    - Rule version control
    """

    def __init__(self, rules: Optional[List[Dict]] = None, config_path: Optional[str] = None):
        """
        Args:
            rules: List of rule dictionaries
            config_path: Path to YAML config file
        """
        self.rules = rules or []
        self.rule_weights = {}
        self.rule_categories = {}
        self._initialize_metadata()

        if config_path:
            self.load_rules(config_path)

    def _initialize_metadata(self):
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'rule_counts': {
                'total': 0,
                'by_category': {}
            }
        }

    def add_rule(self, 
                rule_fn: Callable, 
                name: str,
                category: str = "general",
                weight: float = 1.0,
                severity: RiskSeverity = RiskSeverity.MEDIUM):
        """
        Add a rule with metadata
        """
        self.rules.append({
            'function': rule_fn,
            'name': name,
            'category': category,
            'weight': weight,
            'severity': severity
        })
        self._update_metadata()

    def _update_metadata(self):
        self.metadata.update({
            'last_updated': datetime.now().isoformat(),
            'rule_counts': {
                'total': len(self.rules),
                'by_category': pd.Series([r['category'] for r in self.rules]).value_counts().to_dict()
            }
        })

    def apply(self, 
             df: pd.DataFrame,
             temporal_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply rules with temporal awareness
        """
        df = df.copy()
        df['operational_risk_score'] = 0
        df['operational_risk_flags'] = ""
        
        for rule in self.rules:
            try:
                mask = rule['function'](df, temporal_context)
                df.loc[mask, 'operational_risk_score'] += rule['weight']
                
                # Append rule name to flags
                df.loc[mask, 'operational_risk_flags'] += f"{rule['name']};"
                
            except Exception as e:
                logger.error(f"Error applying rule {rule['name']}: {str(e)}")
        
        # Normalize score 0-1000
        max_score = sum(r['weight'] for r in self.rules)
        if max_score > 0:
            df['operational_risk_score'] = (df['operational_risk_score'] / max_score * 1000).clip(0, 1000)
        
        # Add severity classification
        df['operational_risk_severity'] = pd.cut(
            df['operational_risk_score'],
            bins=[0, 300, 500, 700, 1000],
            labels=[s.name for s in RiskSeverity]
        )
        
        return df

    def save_rules(self, file_path: str):
        """Save rules to disk with metadata"""
        with open(file_path, 'w') as f:
            json.dump({
                'rules': self.rules,
                'metadata': self.metadata
            }, f, indent=2)

    def load_rules(self, config_path: str):
        """Load rules from YAML config"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.rules = config.get('rules', [])
            self._update_metadata()

# Example rule definitions
def high_value_transaction_rule(df: pd.DataFrame, _) -> pd.Series:
    return df['amount'] > 100000

def rapid_sequence_rule(df: pd.DataFrame, temporal: pd.DataFrame) -> pd.Series:
    if temporal is None:
        return pd.Series(False, index=df.index)
    return temporal['time_diff'] < pd.Timedelta(minutes=5)

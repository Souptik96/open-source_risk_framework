# risk_framework/models/rule_based.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
import re
from enum import Enum
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Standardized risk levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class RuleOperator(Enum):
    """Supported rule operators"""
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not in"
    CONTAINS = "contains"
    REGEX = "regex"
    BETWEEN = "between"

class RuleBasedRiskEngine:
    """
    Enhanced rule-based risk detection system with:
    - Complex rule evaluation
    - Risk scoring
    - Rule weights and priorities
    - Temporal rules
    - Bulk and streaming support
    - Rule persistence
    - Detailed auditing
    """
    
    def __init__(self, 
                 rules: Optional[List[Dict[str, Any]]] = None,
                 config_path: Optional[str] = None,
                 default_risk_level: str = "LOW"):
        """
        Initialize the risk engine.
        
        Args:
            rules: List of rule dictionaries
            config_path: Path to JSON/YAML config file
            default_risk_level: Default risk level
        """
        self.rules = rules or []
        self.default_risk_level = default_risk_level
        self.rule_weights = {}
        self.rule_categories = {}
        self.evaluation_history = []
        
        if config_path:
            self.load_rules(config_path)
            
        self._validate_rules()
        
    def _validate_rules(self) -> None:
        """Validate all rules for correct structure."""
        required_fields = {'column', 'operator', 'value', 'label'}
        for rule in self.rules:
            if not all(field in rule for field in required_fields):
                raise ValueError(f"Rule missing required fields: {rule}")
                
            try:
                RuleOperator(rule['operator'])
            except ValueError:
                raise ValueError(f"Invalid operator in rule: {rule['operator']}")
    
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add a single rule with validation."""
        self._validate_rule(rule)
        self.rules.append(rule)
        
    def _validate_rule(self, rule: Dict[str, Any]) -> None:
        """Validate a single rule."""
        required_fields = {'column', 'operator', 'value', 'label'}
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Rule missing required fields: {rule}")
            
        try:
            RuleOperator(rule['operator'])
        except ValueError:
            raise ValueError(f"Invalid operator in rule: {rule['operator']}")
    
    def load_rules(self, config_path: str) -> None:
        """
        Load rules from JSON or YAML file.
        
        Args:
            config_path: Path to config file
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                if path.suffix.lower() == '.json':
                    config = json.load(f)
                elif path.suffix.lower() in ('.yaml', '.yml'):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported config file format")
                
            self.rules = config.get('rules', [])
            self.default_risk_level = config.get('default_risk_level', 'LOW')
            self.rule_weights = config.get('rule_weights', {})
            self.rule_categories = config.get('rule_categories', {})
            
            logger.info(f"Loaded {len(self.rules)} rules from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading rules: {str(e)}")
            raise
    
    def save_rules(self, config_path: str, format: str = 'json') -> None:
        """
        Save rules to JSON or YAML file.
        
        Args:
            config_path: Path to save config
            format: 'json' or 'yaml'
        """
        config = {
            'rules': self.rules,
            'default_risk_level': self.default_risk_level,
            'rule_weights': self.rule_weights,
            'rule_categories': self.rule_categories,
            'updated_at': datetime.now().isoformat()
        }
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() == 'json':
                    json.dump(config, f, indent=2)
                elif format.lower() in ('yaml', 'yml'):
                    yaml.dump(config, f)
                else:
                    raise ValueError("Unsupported format")
                    
            logger.info(f"Saved {len(self.rules)} rules to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving rules: {str(e)}")
            raise
    
    def _evaluate_rule(self, 
                      df: pd.DataFrame, 
                      rule: Dict[str, Any],
                      current_row: Optional[Dict[str, Any]] = None) -> Union[pd.Series, bool]:
        """
        Evaluate a rule against data.
        
        Args:
            df: DataFrame for bulk evaluation
            rule: Rule definition
            current_row: Single row dict for streaming
            
        Returns:
            Boolean mask (bulk) or boolean result (streaming)
        """
        if current_row is None:
            # Bulk evaluation
            col = rule['column']
            op = rule['operator']
            val = rule['value']
            
            try:
                if op == RuleOperator.EQUAL.value:
                    return df[col] == val
                elif op == RuleOperator.NOT_EQUAL.value:
                    return df[col] != val
                elif op == RuleOperator.GREATER.value:
                    return df[col] > val
                elif op == RuleOperator.GREATER_EQUAL.value:
                    return df[col] >= val
                elif op == RuleOperator.LESS.value:
                    return df[col] < val
                elif op == RuleOperator.LESS_EQUAL.value:
                    return df[col] <= val
                elif op == RuleOperator.IN.value:
                    return df[col].isin(val)
                elif op == RuleOperator.NOT_IN.value:
                    return ~df[col].isin(val)
                elif op == RuleOperator.CONTAINS.value:
                    return df[col].str.contains(val, na=False)
                elif op == RuleOperator.REGEX.value:
                    return df[col].str.match(val, na=False)
                elif op == RuleOperator.BETWEEN.value:
                    return df[col].between(val[0], val[1])
                else:
                    raise ValueError(f"Unsupported operator: {op}")
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule}: {str(e)}")
                return pd.Series(False, index=df.index)
        else:
            # Streaming evaluation
            col = rule['column']
            op = rule['operator']
            val = rule['value']
            
            try:
                cell_value = current_row[col]
                if op == RuleOperator.EQUAL.value:
                    return cell_value == val
                elif op == RuleOperator.NOT_EQUAL.value:
                    return cell_value != val
                elif op == RuleOperator.GREATER.value:
                    return cell_value > val
                elif op == RuleOperator.GREATER_EQUAL.value:
                    return cell_value >= val
                elif op == RuleOperator.LESS.value:
                    return cell_value < val
                elif op == RuleOperator.LESS_EQUAL.value:
                    return cell_value <= val
                elif op == RuleOperator.IN.value:
                    return cell_value in val
                elif op == RuleOperator.NOT_IN.value:
                    return cell_value not in val
                elif op == RuleOperator.CONTAINS.value:
                    return val in str(cell_value)
                elif op == RuleOperator.REGEX.value:
                    return bool(re.match(val, str(cell_value)))
                elif op == RuleOperator.BETWEEN.value:
                    return val[0] <= cell_value <= val[1]
                else:
                    raise ValueError(f"Unsupported operator: {op}")
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule}: {str(e)}")
                return False
    
    def evaluate(self, 
                 df: pd.DataFrame,
                 calculate_score: bool = True,
                 detailed_results: bool = False) -> pd.DataFrame:
        """
        Evaluate all rules against a DataFrame.
        
        Args:
            df: Input DataFrame
            calculate_score: Whether to calculate risk score
            detailed_results: Whether to include rule-by-rule results
            
        Returns:
            DataFrame with risk flags and scores
        """
        df = df.copy()
        df['risk_level'] = self.default_risk_level
        df['risk_score'] = 0
        
        rule_results = {}
        
        for i, rule in enumerate(self.rules):
            rule_name = rule.get('name', f'rule_{i}')
            rule_weight = self.rule_weights.get(rule_name, 1)
            
            mask = self._evaluate_rule(df, rule)
            df.loc[mask, 'risk_level'] = rule['label']
            df.loc[mask, 'risk_score'] += rule_weight
            
            if detailed_results:
                rule_results[rule_name] = mask.astype(int)
        
        # Convert risk_level to categorical for better processing
        risk_levels = [r['label'] for r in self.rules] + [self.default_risk_level]
        df['risk_level'] = pd.Categorical(
            df['risk_level'],
            categories=risk_levels,
            ordered=True
        )
        
        # Normalize risk score (0-1000)
        if calculate_score:
            max_score = sum(self.rule_weights.values()) if self.rule_weights else len(self.rules)
            df['risk_score'] = (df['risk_score'] / max_score * 1000).clip(0, 1000).astype(int)
        
        # Add detailed results if requested
        if detailed_results:
            for rule_name, result in rule_results.items():
                df[f'rule_{rule_name}'] = result
        
        # Log evaluation
        self._log_evaluation(df)
        
        return df
    
    def evaluate_single(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate rules against a single row (for streaming).
        
        Args:
            row: Input data row
            
        Returns:
            Dict with risk assessment
        """
        result = {
            'risk_level': self.default_risk_level,
            'risk_score': 0,
            'triggered_rules': []
        }
        
        for i, rule in enumerate(self.rules):
            rule_name = rule.get('name', f'rule_{i}')
            rule_weight = self.rule_weights.get(rule_name, 1)
            
            if self._evaluate_rule(None, rule, row):
                result['risk_level'] = max(
                    result['risk_level'],
                    rule['label'],
                    key=lambda x: RiskLevel[x].value if x in RiskLevel.__members__ else 0
                )
                result['risk_score'] += rule_weight
                result['triggered_rules'].append(rule_name)
        
        # Normalize risk score (0-1000)
        max_score = sum(self.rule_weights.values()) if self.rule_weights else len(self.rules)
        result['risk_score'] = int((result['risk_score'] / max_score * 1000))
        
        # Log evaluation
        self._log_single_evaluation(row, result)
        
        return result
    
    def _log_evaluation(self, df: pd.DataFrame) -> None:
        """Log evaluation statistics."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'records_processed': len(df),
            'risk_distribution': df['risk_level'].value_counts().to_dict(),
            'avg_risk_score': df['risk_score'].mean(),
            'high_risk_count': len(df[df['risk_level'] != self.default_risk_level])
        }
        
        self.evaluation_history.append(stats)
        logger.info(f"Evaluation stats: {stats}")
    
    def _log_single_evaluation(self, row: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log single evaluation."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_data': {k: v for k, v in row.items() if not isinstance(v, (list, dict))},
            'result': result
        }
        
        self.evaluation_history.append(log_entry)
        logger.debug(f"Evaluated record: {log_entry}")
    
    def get_evaluation_stats(self) -> pd.DataFrame:
        """Get evaluation statistics as DataFrame."""
        return pd.DataFrame(self.evaluation_history)
    
    def save_model(self, file_path: str) -> None:
        """Save the entire model object."""
        joblib.dump(self, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str):
        """Load saved model."""
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model


# Example usage with enhanced features
if __name__ == "__main__":
    # Sample transaction data
    transactions = pd.DataFrame({
        "transaction_id": ["T1001", "T1002", "T1003", "T1004", "T1005"],
        "amount": [5000, 15000, 2000, 50000, 750],
        "currency": ["USD", "USD", "EUR", "USD", "GBP"],
        "country": ["US", "PK", "UK", "RU", "CA"],
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "is_high_risk_occupation": [False, True, False, True, False],
        "transaction_time": [
            "2023-01-15 09:30:00",
            "2023-01-15 23:45:00",
            "2023-01-16 10:15:00",
            "2023-01-16 03:20:00",
            "2023-01-17 14:10:00"
        ]
    })

    # Define comprehensive rules
    risk_rules = [
        {
            "name": "high_amount_usd",
            "column": "amount",
            "operator": ">",
            "value": 10000,
            "label": "MEDIUM",
            "category": "amount"
        },
        {
            "name": "high_risk_country",
            "column": "country",
            "operator": "in",
            "value": ["PK", "RU", "IR", "KP"],
            "label": "HIGH",
            "category": "geo"
        },
        {
            "name": "unusual_time",
            "column": "transaction_time",
            "operator": "regex",
            "value": r"^(00|01|02|03|04|05):",
            "label": "MEDIUM",
            "category": "temporal"
        },
        {
            "name": "high_risk_occupation",
            "column": "is_high_risk_occupation",
            "operator": "==",
            "value": True,
            "label": "HIGH",
            "category": "customer"
        },
        {
            "name": "very_high_amount",
            "column": "amount",
            "operator": ">",
            "value": 30000,
            "label": "CRITICAL",
            "category": "amount"
        }
    ]

    # Rule weights (higher = more severe)
    rule_weights = {
        "high_amount_usd": 2,
        "high_risk_country": 3,
        "unusual_time": 1,
        "high_risk_occupation": 2,
        "very_high_amount": 5
    }

    # Initialize engine
    engine = RuleBasedRiskEngine(
        rules=risk_rules,
        default_risk_level="LOW"
    )
    engine.rule_weights = rule_weights

    # Evaluate transactions
    results = engine.evaluate(
        transactions,
        calculate_score=True,
        detailed_results=True
    )

    print("\nRisk Evaluation Results:")
    print(results[['transaction_id', 'amount', 'country', 'risk_level', 'risk_score']])

    # Example of saving and loading rules
    engine.save_rules("risk_rules.yaml", format="yaml")
    
    # Example of streaming evaluation
    single_transaction = {
        "transaction_id": "T1006",
        "amount": 45000,
        "currency": "USD",
        "country": "IR",
        "customer_id": "C006",
        "is_high_risk_occupation": True,
        "transaction_time": "2023-01-18 02:30:00"
    }
    
    single_result = engine.evaluate_single(single_transaction)
    print("\nSingle Transaction Evaluation:")
    print(single_result)

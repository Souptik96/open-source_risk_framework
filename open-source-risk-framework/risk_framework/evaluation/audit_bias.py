# risk_framework/evaluation/audit_bias.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)

class FairnessMetric(Enum):
    """Supported fairness metrics"""
    STATISTICAL_PARITY = "statistical_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    DISPARATE_IMPACT = "disparate_impact"

@dataclass
class BiasReport:
    """Structured bias audit results"""
    group_metrics: Dict[str, Dict[str, float]]
    fairness_metrics: Dict[str, float]
    statistical_tests: Dict[str, float]
    recommendations: List[str]

class BiasAuditor:
    """
    Comprehensive bias and fairness evaluation with:
    - Multiple fairness definitions
    - Statistical testing
    - Automated recommendations
    """
    
    def __init__(self,
                 sensitive_attributes: List[str],
                 positive_label: int = 1):
        """
        Args:
            sensitive_attributes: Columns containing sensitive groups
            positive_label: Which label represents the positive class
        """
        self.sensitive_attributes = sensitive_attributes
        self.positive_label = positive_label
        
    def evaluate(self,
                 data: pd.DataFrame,
                 y_true: str,
                 y_pred: str,
                 y_prob: Optional[str] = None) -> BiasReport:
        """
        Run comprehensive bias evaluation.
        
        Args:
            data: DataFrame containing predictions and sensitive attributes
            y_true: Column with true labels
            y_pred: Column with predicted labels
            y_prob: Column with predicted probabilities (optional)
            
        Returns:
            BiasReport with detailed findings
        """
        report = {
            'group_metrics': {},
            'fairness_metrics': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Calculate group-wise metrics
        for attr in self.sensitive_attributes:
            report['group_metrics'][attr] = self._group_metrics(
                data, attr, y_true, y_pred
            )
            
            # Calculate fairness metrics
            report['fairness_metrics'].update(
                self._calculate_fairness(
                    data, attr, y_true, y_pred, y_prob
                )
            )
            
            # Run statistical tests
            report['statistical_tests'].update(
                self._statistical_tests(
                    data, attr, y_true, y_pred
                )
            )
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return BiasReport(**report)
    
    def _group_metrics(self,
                      data: pd.DataFrame,
                      attribute: str,
                      y_true: str,
                      y_pred: str) -> Dict[str, float]:
        """Calculate performance metrics by group"""
        metrics = {}
        groups = data[attribute].unique()
        
        for group in groups:
            group_data = data[data[attribute] == group]
            y_g = group_data[y_true]
            p_g = group_data[y_pred]
            
            tn, fp, fn, tp = confusion_matrix(
                y_g, p_g, labels=[0, 1]
            ).ravel()
            
            metrics[group] = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            }
            
        return metrics
    
    def _calculate_fairness(self,
                           data: pd.DataFrame,
                           attribute: str,
                           y_true: str,
                           y_pred: str,
                           y_prob: str) -> Dict[str, float]:
        """Calculate various fairness metrics"""
        fairness = {}
        groups = data[attribute].unique()
        
        # Statistical parity difference
        pos_rate = [
            (data[(data[attribute] == g)][y_pred] == self.positive_label).mean()
            for g in groups
        ]
        fairness[f'{attribute}_statistical_parity_diff'] = max(pos_rate) - min(pos_rate)
        
        # Add other fairness metrics...
        
        return fairness
    
    def _statistical_tests(self,
                          data: pd.DataFrame,
                          attribute: str,
                          y_true: str,
                          y_pred: str) -> Dict[str, float]:
        """Run statistical significance tests"""
        tests = {}
        groups = data[attribute].unique()
        
        if len(groups) == 2:
            group_a, group_b = groups
            a_preds = data[data[attribute] == group_a][y_pred]
            b_preds = data[data[attribute] == group_b][y_pred]
            
            # Chi-square test for independence
            contingency = pd.crosstab(data[attribute], data[y_pred])
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            tests[f'{attribute}_chi2_pvalue'] = p
            
            # T-test for mean differences
            t_stat, p_val = stats.ttest_ind(a_preds, b_preds)
            tests[f'{attribute}_ttest_pvalue'] = p_val
            
        return tests
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recs = []
        
        # Example recommendation logic
        for attr, metrics in report['fairness_metrics'].items():
            if 'statistical_parity_diff' in attr and metrics > 0.1:
                recs.append(
                    f"Large statistical parity difference detected for {attr}. "
                    "Consider debiasing techniques or threshold adjustment."
                )
                
        return recs

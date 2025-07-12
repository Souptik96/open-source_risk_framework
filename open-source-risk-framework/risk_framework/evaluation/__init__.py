"""
Risk Model Evaluation Module

Provides:
- Model explainability via SHAP
- Fairness/bias auditing
- Performance metrics tracking
"""

from .shap_explain import SHAPExplainer, ExplanationType
from .audit_bias import BiasAuditor, FairnessMetric, BiasReport
from typing import List, Dict, Optional, Union
import pandas as pd
import logging

__version__ = "1.0.0"
__all__ = [
    'SHAPExplainer',
    'ExplanationType',
    'BiasAuditor', 
    'FairnessMetric',
    'BiasReport',
    'evaluate_model'
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(
    model,
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_attributes: Optional[List[str]] = None,
    model_type: str = 'tree'
) -> Dict[str, Union[Dict, plt.Figure]]:
    """
    Run comprehensive model evaluation including:
    - SHAP explainability
    - Bias/fairness audit (if sensitive attributes provided)
    
    Args:
        model: Trained model object
        X: Feature dataframe
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attributes: Columns for bias analysis
        model_type: Model type for SHAP ('tree', 'linear', etc.)
    
    Returns:
        Dictionary containing:
        - 'shap_explanation': SHAP values and plots
        - 'bias_report': Bias metrics (if sensitive_attributes provided)
    """
    results = {}
    
    try:
        # SHAP Explanation
        logger.info("Generating SHAP explanation...")
        shap_exp = SHAPExplainer(model, model_type=model_type)
        shap_exp.fit(X)
        results['shap'] = {
            'values': shap_exp.shap_values,
            'summary_plot': shap_exp.explain(ExplanationType.SUMMARY),
            'expected_value': shap_exp.expected_value
        }
        
        # Bias Audit if sensitive attributes provided
        if sensitive_attributes:
            logger.info("Running bias audit...")
            data = X.copy()
            data['y_true'] = y_true
            data['y_pred'] = y_pred
            
            auditor = BiasAuditor(sensitive_attributes)
            bias_report = auditor.evaluate(data, 'y_true', 'y_pred')
            results['bias'] = bias_report.__dict__
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    
    return results

# Example CLI Interface
if __name__ == "__main__":
    import argparse
    import joblib
    import json
    
    parser = argparse.ArgumentParser(description="Model Evaluation CLI")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("data_path", help="Path to evaluation data")
    parser.add_argument("--sensitive_attrs", nargs='+', help="Sensitive attributes for bias audit")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Load model and data
    model = joblib.load(args.model_path)
    data = pd.read_csv(args.data_path)
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        X=data.drop(columns=['target']),
        y_true=data['target'],
        y_pred=model.predict(data.drop(columns=['target'])),
        sensitive_attributes=args.sensitive_attrs
    )
    
    # Save or display results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

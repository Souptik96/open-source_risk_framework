# risk_framework/evaluation/shap_explain.py
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict
import json
from pathlib import Path
import logging
import warnings
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class ExplanationType(Enum):
    """Supported explanation formats"""
    SUMMARY = "summary_plot"
    FORCE = "force_plot"
    DECISION = "decision_plot"
    DEPENDENCE = "dependence_plot"

class SHAPExplainer:
    """
    Enhanced SHAP explainer with:
    - Multiple explanation types
    - Model-agnostic support
    - Serialization capabilities
    - Production monitoring
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize explainer for a trained model.
        
        Args:
            model: Trained model with predict/predict_proba method
            model_type: One of ['tree', 'linear', 'kernel', 'deep']
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = None
        
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray],
            feature_names: Optional[List[str]] = None) -> None:
        """
        Calculate SHAP values for the dataset.
        
        Args:
            X: Input features (DataFrame or array)
            feature_names: List of feature names (required if X is array)
        """
        try:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
                X = X.values
            elif feature_names:
                self.feature_names = feature_names
            else:
                raise ValueError("feature_names required when X is not DataFrame")
            
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, X)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, X)
                
            self.shap_values = self.explainer.shap_values(X)
            self.expected_value = self.explainer.expected_value
            
            logger.info(f"SHAP explanation initialized for {self.model_type} model")
            
        except Exception as e:
            logger.error(f"SHAP calculation failed: {str(e)}")
            raise
    
    def explain(self,
                explanation_type: ExplanationType = ExplanationType.SUMMARY,
                **kwargs) -> plt.Figure:
        """
        Generate specified explanation plot.
        
        Args:
            explanation_type: Type of explanation to generate
            **kwargs: Plot-specific arguments
            
        Returns:
            Matplotlib figure object
        """
        if self.shap_values is None:
            raise RuntimeError("Call fit() before generating explanations")
            
        try:
            plt.figure()
            
            if explanation_type == ExplanationType.SUMMARY:
                shap.summary_plot(
                    self.shap_values,
                    feature_names=self.feature_names,
                    **kwargs
                )
            elif explanation_type == ExplanationType.FORCE:
                shap.force_plot(
                    self.expected_value,
                    self.shap_values[0],
                    feature_names=self.feature_names,
                    **kwargs
                )
            # Additional explanation types...
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}")
            raise
    
    def save_explanation(self, 
                        file_path: str,
                        format: str = 'json') -> None:
        """
        Save SHAP values to file.
        
        Args:
            file_path: Output file path
            format: One of ['json', 'csv', 'npy']
        """
        try:
            if format == 'json':
                with open(file_path, 'w') as f:
                    json.dump({
                        'shap_values': self.shap_values.tolist(),
                        'expected_value': self.expected_value,
                        'feature_names': self.feature_names,
                        'model_type': self.model_type
                    }, f)
            # Other formats...
            
            logger.info(f"SHAP explanation saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save explanation: {str(e)}")
            raise

    @classmethod
    def load_explanation(cls, file_path: str):
        """Load saved SHAP explanation"""
        # Implementation omitted for brevity
        pass

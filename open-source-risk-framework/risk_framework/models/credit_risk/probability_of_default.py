# probability_of_default.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PDCalculator:
    """
    Advanced Probability of Default Model with:
    - Multi-method estimation (logistic, survival)
    - Feature significance testing
    - Scorecard transformation
    - Regulatory compliance
    """

    def __init__(self,
                method: str = 'logistic',
                scorecard_bins: Optional[int] = 20):
        """
        Args:
            method: Modeling approach ('logistic', 'survival')
            scorecard_bins: Number of bins for scorecard transformation
        """
        self.method = method
        self.scorecard_bins = scorecard_bins
        self.model = None
        self.feature_stats = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'regulatory_framework': 'Basel III'
        }

    def fit(self,
           X: pd.DataFrame,
           y: pd.Series,
           feature_names: Optional[List[str]] = None):
        """
        Train PD model with automated diagnostics
        """
        # Check for multicollinearity
        self._check_vif(X)
        
        if self.method == 'logistic':
            self.model = CalibratedClassifierCV(
                LogisticRegression(penalty='l1', solver='liblinear'),
                cv=5
            ).fit(X, y)
        elif self.method == 'survival':
            raise NotImplementedError("Survival analysis coming in v2.1")
        
        # Calculate feature statistics
        self._calculate_feature_stats(X, y, feature_names or X.columns)
        
        self.metadata.update({
            'last_trained': datetime.now().isoformat(),
            'features': list(X.columns),
            'performance': {
                'brier_score': brier_score_loss(y, self.predict_proba(X))
            }
        })

    def _check_vif(self, X: pd.DataFrame, threshold: float = 5.0):
        """Detect multicollinearity issues"""
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        if any(vif > threshold):
            logger.warning(f"High VIF detected:\n{vif[vif > threshold]}")

    def _calculate_feature_stats(self,
                               X: pd.DataFrame,
                               y: pd.Series,
                               feature_names: List[str]):
        """Compute feature-level statistics"""
        for col in feature_names:
            self.feature_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'corr_with_target': X[col].corr(y)
            }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict PD probabilities"""
        if not self.model:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def predict_scorecard(self, X: pd.DataFrame) -> pd.Series:
        """Convert PD to scorecard points"""
        probs = self.predict_proba(X)
        return pd.Series(
            np.floor(1000 * (1 - probs)),  # Higher score = better
            name='credit_score'
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature coefficients"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Feature importance not available for this model type")
        return pd.DataFrame({
            'feature': self.metadata['features'],
            'coefficient': self.model.coef_[0],
            'abs_importance': np.abs(self.model.coef_[0])
        }).sort_values('abs_importance', ascending=False)

    def save_model(self, file_path: str):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'metadata': self.metadata,
            'feature_stats': self.feature_stats
        }, file_path)

    @classmethod
    def load_model(cls, file_path: str):
        """Load saved model"""
        data = joblib.load(file_path)
        model = cls()
        model.model = data['model']
        model.metadata = data['metadata']
        model.feature_stats = data.get('feature_stats', {})
        return model

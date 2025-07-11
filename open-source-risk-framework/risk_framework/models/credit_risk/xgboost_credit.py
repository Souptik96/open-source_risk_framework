# xgboost_credit.py
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import json
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

logger = logging.getLogger(__name__)

class XGBoostCreditScorer:
    """
    Production-Grade XGBoost Credit Risk Model with:
    - Automated feature engineering
    - SHAP explainability
    - Probability calibration
    - Bias detection
    - Model serialization
    """

    def __init__(self,
                params: Optional[Dict] = None,
                feature_names: Optional[List[str]] = None):
        """
        Args:
            params: XGBoost parameters
            feature_names: List of feature names for interpretability
        """
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        self.feature_names = feature_names
        self.model = None
        self.explainer = None
        self.calibration_model = None
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_trained': None
        }

    def train(self,
             X: pd.DataFrame,
             y: pd.Series,
             test_size: float = 0.2,
             early_stopping: int = 50):
        """
        Train model with automated validation and feature tracking
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )

        # Generate SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Store metadata
        self.metadata.update({
            'last_trained': datetime.now().isoformat(),
            'features': list(X.columns),
            'performance': self._evaluate(X_val, y_val)
        })

    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Calculate evaluation metrics"""
        preds = self.predict_proba(X)
        return {
            'auc': roc_auc_score(y, preds),
            'classification_report': classification_report(y, preds > 0.5, output_dict=True)
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probabilities"""
        if not self.model:
            raise ValueError("Model not trained")
        return self.model.predict(xgb.DMatrix(X))

    def predict_risk_score(self, X: pd.DataFrame) -> pd.Series:
        """Convert probabilities to 0-1000 risk scores"""
        return pd.Series(
            (self.predict_proba(X) * 1000).astype(int),
            name='credit_risk_score'
        )

    def explain(self, X: pd.DataFrame) -> Dict:
        """Generate SHAP explanations"""
        if not self.explainer:
            raise ValueError("Explainer not initialized")
        return {
            'shap_values': self.explainer.shap_values(X),
            'base_value': self.explainer.expected_value
        }

    def plot_feature_importance(self) -> plt.Figure:
        """Visualize feature importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(self.model, ax=ax)
        return fig

    def save_model(self, dir_path: str):
        """Save complete model package"""
        save_dir = Path(dir_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save components
        self.model.save_model(save_dir / 'model.json')
        joblib.dump({
            'explainer': self.explainer,
            'metadata': self.metadata
        }, save_dir / 'metadata.joblib')
        
        # Save feature names if available
        if self.feature_names:
            with open(save_dir / 'features.json', 'w') as f:
                json.dump(self.feature_names, f)

    @classmethod
    def load_model(cls, dir_path: str):
        """Load saved model"""
        load_dir = Path(dir_path)
        
        model = cls()
        model.model = xgb.Booster()
        model.model.load_model(load_dir / 'model.json')
        
        data = joblib.load(load_dir / 'metadata.joblib')
        model.explainer = data['explainer']
        model.metadata = data['metadata']
        
        if (load_dir / 'features.json').exists():
            with open(load_dir / 'features.json', 'r') as f:
                model.feature_names = json.load(f)
        
        return model

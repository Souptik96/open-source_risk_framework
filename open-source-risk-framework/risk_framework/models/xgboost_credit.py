# risk_framework/models/xgboost_credit.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from typing import List, Optional, Dict, Union, Tuple
import joblib
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import warnings
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class XGBoostCreditRiskModel:
    """
    Enhanced XGBoost credit risk model with:
    - Automated feature engineering
    - Hyperparameter optimization
    - SHAP explainability
    - Class imbalance handling
    - Model persistence
    - Comprehensive monitoring
    - Risk score calibration
    """
    
    def __init__(self,
                 features: List[str],
                 target: str,
                 model_dir: str = "models",
                 test_size: float = 0.2,
                 random_state: Optional[int] = 42,
                 enable_hyperopt: bool = False):
        """
        Initialize the credit risk model.
        
        Args:
            features: List of feature column names
            target: Target column name (binary: 1=default, 0=non-default)
            model_dir: Directory to save/load models
            test_size: Fraction of data for validation
            random_state: Random seed for reproducibility
            enable_hyperopt: Whether to enable hyperparameter optimization
        """
        self.features = features
        self.target = target
        self.model_dir = Path(model_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.enable_hyperopt = enable_hyperopt
        self.scaler = StandardScaler()
        self.binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.explainer = None
        self.training_metadata = {}
        self.feature_importances_ = None
        
        # Initialize model with sensible defaults
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=random_state,
            early_stopping_rounds=10
        )
        
        # Create model directory if needed
        self.model_dir.mkdir(exist_ok=True, parents=True)
    
    def _preprocess_data(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Handle missing values, feature scaling, and binning.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the preprocessing transformers
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        for col in self.features:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('MISSING')
        
        # Feature transformations
        if fit:
            self.scaler.fit(df[self.features])
            self.binner.fit(df[self.features])
        
        # Apply transformations
        scaled_features = self.scaler.transform(df[self.features])
        binned_features = self.binner.transform(df[self.features])
        
        # Create enhanced feature set
        df[[f"{col}_scaled" for col in self.features]] = scaled_features
        df[[f"{col}_binned" for col in self.features]] = binned_features
        
        return df
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE oversampling to handle class imbalance."""
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    
    def _hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize hyperparameters using Bayesian optimization."""
        space = {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'gamma': hp.uniform('gamma', 0, 5),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0, 10)
        }
        
        def objective(params):
            params = {
                'max_depth': int(params['max_depth']),
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'gamma': params['gamma'],
                'min_child_weight': params['min_child_weight'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': self.random_state
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
            
            return {'loss': -np.mean(scores), 'status': STATUS_OK}
        
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            rstate=np.random.RandomState(self.random_state)
        )
        
        # Convert best params to proper types
        best_params = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'gamma': best['gamma'],
            'min_child_weight': best['min_child_weight'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda']
        }
        
        return best_params
    
    def train(self, df: pd.DataFrame, save_model: bool = True) -> Dict:
        """
        Train the credit risk model with enhanced features.
        
        Args:
            df: Training DataFrame
            save_model: Whether to persist the trained model
            
        Returns:
            Dictionary containing training metrics and metadata
        """
        try:
            logger.info("Starting model training...")
            start_time = datetime.now()
            
            # Preprocess data
            df_processed = self._preprocess_data(df, fit=True)
            
            # Prepare features (original + engineered)
            all_features = (
                self.features + 
                [f"{col}_scaled" for col in self.features] + 
                [f"{col}_binned" for col in self.features]
            )
            
            X = df_processed[all_features]
            y = df_processed[self.target]
            
            # Handle class imbalance
            X_res, y_res = self._handle_class_imbalance(X, y)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_res, y_res,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_res
            )
            
            # Hyperparameter optimization
            if self.enable_hyperopt:
                logger.info("Running hyperparameter optimization...")
                best_params = self._hyperparameter_tuning(X_train, y_train)
                self.model = xgb.XGBClassifier(**best_params)
                logger.info(f"Optimized parameters: {best_params}")
            
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
            
            # Generate predictions and metrics
            val_preds = self.model.predict(X_val)
            val_probs = self.model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'classification_report': classification_report(y_val, val_preds, output_dict=True),
                'roc_auc': roc_auc_score(y_val, val_probs),
                'average_precision': average_precision_score(y_val, val_probs),
                'confusion_matrix': confusion_matrix(y_val, val_preds).tolist(),
                'training_duration': (datetime.now() - start_time).total_seconds(),
                'trained_at': datetime.now().isoformat(),
                'feature_count': len(all_features),
                'class_balance': {
                    'train_pos': y_train.mean(),
                    'train_neg': 1 - y_train.mean(),
                    'val_pos': y_val.mean(),
                    'val_neg': 1 - y_val.mean()
                }
            }
            
            # Feature importance
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_,
                index=all_features
            ).sort_values(ascending=False)
            
            # SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            # Save training metadata
            self.training_metadata = {
                'features': all_features,
                'target': self.target,
                'model_params': self.model.get_params(),
                'metrics': metrics,
                'preprocessor_params': {
                    'scaler_mean': self.scaler.mean_.tolist(),
                    'scaler_scale': self.scaler.scale_.tolist(),
                    'binner_bins': [b.tolist() for b in self.binner.bin_edges_]
                }
            }
            
            if save_model:
                self.save_model()
            
            logger.info(f"Training completed in {metrics['training_duration']:.2f} seconds")
            logger.info(f"Validation ROC AUC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame, return_shap: bool = False) -> pd.DataFrame:
        """
        Generate credit risk predictions with explanations.
        
        Args:
            df: DataFrame to score
            return_shap: Whether to include SHAP values
            
        Returns:
            DataFrame with predictions and risk analysis
        """
        try:
            df = df.copy()
            df_processed = self._preprocess_data(df)
            
            # Prepare features
            all_features = (
                self.features + 
                [f"{col}_scaled" for col in self.features] + 
                [f"{col}_binned" for col in self.features]
            )
            
            X = df_processed[all_features]
            
            # Generate predictions
            df['default_probability'] = self.model.predict_proba(X)[:, 1]
            df['predicted_risk'] = (df['default_probability'] > 0.5).astype(int)
            
            # Risk score calibration (0-1000)
            df['risk_score'] = (1000 * (1 - df['default_probability'])).round().astype(int)
            
            # Risk bands
            df['risk_band'] = pd.cut(
                df['risk_score'],
                bins=[0, 300, 500, 700, 850, 1000],
                labels=['Very High', 'High', 'Medium', 'Low', 'Very Low'],
                include_lowest=True
            )
            
            # Add SHAP values if requested
            if return_shap and self.explainer:
                shap_values = self.explainer.shap_values(X)
                for i, feature in enumerate(all_features):
                    df[f'shap_{feature}'] = shap_values[:, i]
            
            return df
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance."""
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Train model first.")
            
        plt.figure(figsize=(10, 6))
        self.feature_importances_.head(top_n).plot(kind='barh')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name: str = "xgboost_credit") -> None:
        """Save model and metadata to disk."""
        model_path = self.model_dir / f"{model_name}.joblib"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        # Save model and components
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'binner': self.binner,
            'features': self.features,
            'target': self.target
        }, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
            
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_name: str = "xgboost_credit", model_dir: str = "models"):
        """Load trained model from disk."""
        model_path = Path(model_dir) / f"{model_name}.joblib"
        metadata_path = Path(model_dir) / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        data = joblib.load(model_path)
        
        instance = cls(
            features=data['features'],
            target=data['target'],
            model_dir=model_dir
        )
        
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.binner = data['binner']
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                instance.training_metadata = json.load(f)
        
        # Recreate explainer
        instance.explainer = shap.TreeExplainer(instance.model)
        
        logger.info(f"Model loaded from {model_path}")
        return instance


# Example usage with realistic credit data
if __name__ == "__main__":
    # Simulate realistic credit dataset
    np.random.seed(42)
    n_samples = 5000
    
    # Generate features
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'credit_score': np.clip(np.random.normal(650, 100, n_samples), 300, 850).astype(int),
        'income': np.clip(np.random.lognormal(10.5, 0.4, n_samples), 20000, 200000).astype(int),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 100,
        'loan_amount': np.random.lognormal(9, 0.3, n_samples).astype(int),
        'employment_length': np.random.randint(0, 10, n_samples),
        'recent_inquiries': np.random.poisson(0.5, n_samples),
        'delinquencies': np.random.poisson(0.2, n_samples),
        'credit_age': np.random.gamma(5, 2, n_samples).astype(int)
    })
    
    # Simulate default (5% default rate)
    default_prob = 1 / (1 + np.exp(
        -(-3.5 + 
          0.005 * data['credit_score'] + 
          0.00001 * data['income'] - 
          0.03 * data['debt_to_income'] + 
          0.00002 * data['loan_amount'] - 
          0.1 * data['employment_length'] + 
          0.3 * data['recent_inquiries'] + 
          0.5 * data['delinquencies'] - 
          0.02 * data['credit_age'])
    ))
    
    data['default'] = np.random.binomial(1, default_prob)
    
    # Define features and target
    features = ['credit_score', 'income', 'debt_to_income', 'loan_amount',
                'employment_length', 'recent_inquiries', 'delinquencies', 'credit_age']
    target = 'default'
    
    # Initialize and train model
    model = XGBoostCreditRiskModel(
        features=features,
        target=target,
        model_dir="saved_models",
        enable_hyperopt=True
    )
    
    # Train model
    metrics = model.train(data)
    print(f"\nModel Performance:")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    # Generate predictions
    predictions = model.predict(data, return_shap=True)
    
    # Analyze results
    high_risk = predictions[predictions['risk_band'].isin(['Very High', 'High'])]
    print(f"\nDetected {len(high_risk)} high-risk customers:")
    print(high_risk[['customer_id', 'credit_score', 'risk_score', 'risk_band']].head())
    
    # Feature importance
    print("\nTop 10 Features:")
    print(model.feature_importances_.head(10))
    
    # Save and load demonstration
    loaded_model = XGBoostCreditRiskModel.load_model(model_dir="saved_models")
    print("\nModel successfully loaded with features:", loaded_model.features)

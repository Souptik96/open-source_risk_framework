# isolation_forest.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.fe_selection import mutual_info_classif
from typing import Optional, Dict, List, Union
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class FinancialCrimeDetector:
    """
    Enhanced Isolation Forest for Financial Crime Detection with:
    - Dynamic threshold adaptation
    - Feature importance analysis
    - Risk scoring (0-1000)
    - Model explainability
    - Production-ready serialization
    """

    def __init__(self,
                 n_estimators: int = 200,
                 max_samples: Union[str, float] = 'auto',
                 contamination: float = 0.01,
                 max_features: float = 1.0,
                 random_state: int = 42,
                 risk_thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
            n_estimators: Number of base estimators
            max_samples: Samples to draw for each estimator
            contamination: Expected proportion of anomalies
            max_features: Features to draw for each estimator
            random_state: Random seed
            risk_thresholds: Custom thresholds for risk levels
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.scaler = RobustScaler()  # Better for financial data
        self.feature_importances_ = None
        self.training_metadata = {}
        self._is_trained = False

        # Risk thresholds (can be customized)
        self.risk_thresholds = risk_thresholds or {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }

    def fit(self, 
            X: pd.DataFrame,
            feature_names: Optional[List[str]] = None) -> None:
        """
        Train detector on financial transaction data.
        
        Args:
            X: Input features (DataFrame)
            feature_names: Optional list of feature names
        """
        logger.info(f"Training on {X.shape[0]} transactions with {X.shape[1]} features")
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        feature_names = feature_names or list(X.columns)

        # Model training
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1  # Use all cores
        )
        self.model.fit(X_scaled)

        # Calculate feature importance
        self._calculate_feature_importance(X, feature_names)

        # Store training metadata
        self.training_metadata = {
            'training_date': datetime.now().isoformat(),
            'input_shape': X.shape,
            'feature_names': feature_names,
            'model_params': self.get_params(),
            'contamination': self.contamination
        }

        self._is_trained = True
        logger.info("Training completed successfully")

    def _calculate_feature_importance(self,
                                    X: pd.DataFrame,
                                    feature_names: List[str]) -> None:
        """Calculate feature importance using mutual information"""
        scores = self.model.decision_function(self.scaler.transform(X))
        binary_labels = (scores < np.percentile(scores, 100 * self.contamination)).astype(int)
        
        importances = mutual_info_classif(X, binary_labels, random_state=self.random_state)
        self.feature_importances_ = pd.Series(
            importances,
            index=feature_names,
            name='feature_importance'
        ).sort_values(ascending=False)

    def predict(self,
               X: pd.DataFrame,
               return_scores: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """
        Detect suspicious transactions.
        
        Args:
            X: Input data (DataFrame)
            return_scores: If True, returns raw anomaly scores
            
        Returns:
            DataFrame with detection results or array of scores
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
        if return_scores:
            return scores

        # Normalize scores to 0-1 range (higher = more anomalous)
        norm_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        return pd.DataFrame({
            'raw_score': scores,
            'risk_score': self._normalize_to_risk_score(norm_scores),
            'is_alert': norm_scores > self.risk_thresholds['low'],
            'severity': self._assign_severity(norm_scores)
        })

    def _normalize_to_risk_score(self, scores: np.ndarray) -> np.ndarray:
        """Convert to 0-1000 risk score (industry standard)"""
        return np.clip(scores * 1000, 0, 1000).astype(int)

    def _assign_severity(self, scores: np.ndarray) -> List[str]:
        """Classify into risk levels"""
        bins = [0, self.risk_thresholds['low'], 
                self.risk_thresholds['medium'],
                self.risk_thresholds['high'], 1]
        labels = [RiskSeverity.LOW.name, 
                 RiskSeverity.MEDIUM.name,
                 RiskSeverity.HIGH.name,
                 RiskSeverity.CRITICAL.name]
        return pd.cut(scores, bins=bins, labels=labels, include_lowest=True)

    def plot_score_distribution(self, X: pd.DataFrame) -> plt.Figure:
        """Visualize anomaly score distribution"""
        scores = self.predict(X, return_scores=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(scores, bins=50, alpha=0.7)
        
        # Add threshold lines
        colors = ['green', 'orange', 'red']
        for i, (level, threshold) in enumerate(self.risk_thresholds.items()):
            ax.axvline(threshold, color=colors[i], linestyle='--',
                      label=f'{level} risk threshold')
        
        ax.set_title('Anomaly Score Distribution')
        ax.set_xlabel('Normalized Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        return fig

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'random_state': self.random_state
        }

    def save(self, dir_path: str) -> None:
        """Save complete model package"""
        save_dir = Path(dir_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model components
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importances': self.feature_importances_,
            'risk_thresholds': self.risk_thresholds
        }, save_dir / 'model.joblib')
        
        # Save metadata
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(self.training_metadata, f, indent=2)

    @classmethod
    def load(cls, dir_path: str):
        """Load saved model package"""
        load_dir = Path(dir_path)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Directory {dir_path} not found")
            
        # Load components
        data = joblib.load(load_dir / 'model.joblib')
        
        # Initialize detector
        detector = cls(
            n_estimators=data['model'].n_estimators,
            contamination=data['model'].contamination,
            random_state=data['model'].random_state
        )
        
        detector.model = data['model']
        detector.scaler = data['scaler']
        detector.feature_importances_ = data['feature_importances']
        detector.risk_thresholds = data.get('risk_thresholds', detector.risk_thresholds)
        
        # Load metadata
        with open(load_dir / 'metadata.json', 'r') as f:
            detector.training_metadata = json.load(f)
            
        detector._is_trained = True
        return detector


# Example usage
if __name__ == "__main__":
    # Simulate transaction data
    np.random.seed(42)
    n_normal = 10000
    n_anomalies = int(n_normal * 0.01)  # 1% anomalies
    
    normal_data = np.random.normal(0, 1, (n_normal, 10))
    anomalies = np.random.uniform(5, 10, (n_anomalies, 10))
    
    X = pd.DataFrame(np.vstack([normal_data, anomalies]),
                    columns=[f"feature_{i}" for i in range(10)])
    
    # Train detector
    detector = FinancialCrimeDetector(
        n_estimators=500,
        contamination=0.01,
        random_state=42
    )
    detector.fit(X)
    
    # Detect anomalies
    results = detector.predict(X)
    print(f"Detected {results['is_alert'].sum()} suspicious transactions")
    print("\nFeature Importances:")
    print(detector.feature_importances_.head())
    
    # Save model
    detector.save("financial_crime_detector")

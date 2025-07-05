# risk_framework/models/isolation_forest.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict, Union
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsolationForestRiskDetector:
    """
    Enhanced Isolation Forest implementation for financial risk detection with:
    - Automated feature scaling
    - Model persistence
    - Threshold-based anomaly classification
    - Risk scoring
    - Explainability features
    - Performance tracking
    """
    
    def __init__(self, 
                 features: List[str], 
                 contamination: float = 0.05,
                 risk_threshold: float = -0.5,
                 random_state: Optional[int] = 42,
                 model_dir: str = "models"):
        """
        Initialize enhanced Isolation Forest model for risk detection.
        
        Args:
            features: List of feature column names
            contamination: Expected proportion of outliers (0-0.5)
            risk_threshold: Score threshold for risk classification
            random_state: Random seed for reproducibility
            model_dir: Directory to save/load models
        """
        self.features = features
        self.contamination = contamination
        self.risk_threshold = risk_threshold
        self.random_state = random_state
        self.model_dir = Path(model_dir)
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150,  # Increased for better stability
            behaviour='new'
        )
        self.training_stats = {}
        self.feature_importances_ = None
        
        # Ensure model directory exists
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
    def _preprocess_data(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Handle missing values and scale features."""
        df = df.copy()
        
        # Fill missing values with median (robust to outliers)
        for col in self.features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Scale features
        if fit_scaler:
            self.scaler.fit(df[self.features])
        df[self.features] = self.scaler.transform(df[self.features])
        
        return df
    
    def fit(self, df: pd.DataFrame, save_model: bool = False) -> None:
        """
        Fit model and calculate training statistics.
        
        Args:
            df: Training DataFrame
            save_model: Whether to persist the trained model
        """
        try:
            logger.info("Starting model training...")
            start_time = datetime.now()
            
            # Preprocess data
            df_processed = self._preprocess_data(df, fit_scaler=True)
            
            # Train model
            self.model.fit(df_processed[self.features])
            
            # Calculate training statistics
            train_scores = self.model.decision_function(df_processed[self.features])
            self.training_stats = {
                'mean_score': np.mean(train_scores),
                'score_std': np.std(train_scores),
                'contamination': self.contamination,
                'features': self.features,
                'trained_at': datetime.now().isoformat(),
                'training_duration': (datetime.now() - start_time).total_seconds()
            }
            
            # Calculate feature importances (approximation)
            self._calculate_feature_importances(df_processed)
            
            if save_model:
                self.save_model()
                
            logger.info(f"Model training completed in {self.training_stats['training_duration']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect risks and calculate scores with enhanced reporting.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with risk indicators and scores
        """
        try:
            df = df.copy()
            df_processed = self._preprocess_data(df)
            
            # Get anomaly predictions and scores
            df['risk_score'] = self.model.decision_function(df_processed[self.features])
            df['is_risk'] = (df['risk_score'] <= self.risk_threshold).astype(int)
            
            # Add risk severity levels
            df['risk_severity'] = pd.cut(
                df['risk_score'],
                bins=[-np.inf, -1, -0.5, -0.2, np.inf],
                labels=['critical', 'high', 'medium', 'low']
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def _calculate_feature_importances(self, df: pd.DataFrame) -> None:
        """Approximate feature importances using mean decrease in anomaly score."""
        baseline_scores = self.model.decision_function(df[self.features])
        importances = []
        
        for feature in self.features:
            temp_df = df.copy()
            np.random.shuffle(temp_df[feature].values)
            shuffled_scores = self.model.decision_function(temp_df[self.features])
            importances.append(np.mean(baseline_scores - shuffled_scores))
            
        self.feature_importances_ = pd.Series(importances, index=self.features)
    
    def get_feature_importances(self) -> pd.Series:
        """Return feature importance scores."""
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not calculated. Train model first.")
        return self.feature_importances_
    
    def save_model(self, model_name: str = "isolation_forest") -> None:
        """Save model, scaler, and metadata to disk."""
        model_path = self.model_dir / f"{model_name}.joblib"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'risk_threshold': self.risk_threshold
        }, model_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.training_stats, f)
            
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_name: str = "isolation_forest", model_dir: str = "models"):
        """Load trained model from disk."""
        model_path = Path(model_dir) / f"{model_name}.joblib"
        metadata_path = Path(model_dir) / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        data = joblib.load(model_path)
        
        instance = cls(
            features=data['features'],
            contamination=data['model'].contamination,
            risk_threshold=data['risk_threshold'],
            model_dir=model_dir
        )
        
        instance.model = data['model']
        instance.scaler = data['scaler']
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                instance.training_stats = json.load(f)
                
        logger.info(f"Model loaded from {model_path}")
        return instance


# Example usage with enhanced financial risk scenario
if __name__ == "__main__":
    # Simulate transaction data with different risk patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Normal transactions (95%)
    normal_amounts = np.random.lognormal(mean=5, sigma=0.5, size=int(n_samples * 0.95))
    normal_durations = np.random.uniform(1, 30, size=int(n_samples * 0.95))
    
    # Anomalous transactions (5%)
    anomaly_amounts = np.random.lognormal(mean=9, sigma=1, size=int(n_samples * 0.05))
    anomaly_durations = np.random.uniform(0.1, 1, size=int(n_samples * 0.05))
    
    data = pd.DataFrame({
        'transaction_id': range(n_samples),
        'amount': np.concatenate([normal_amounts, anomaly_amounts]),
        'duration': np.concatenate([normal_durations, anomaly_durations]),
        'merchant_category': np.random.choice(['retail', 'travel', 'gambling', 'utilities'], 
                                             size=n_samples, p=[0.6, 0.2, 0.05, 0.15])
    })
    
    # Initialize and train model
    features = ['amount', 'duration']
    detector = IsolationForestRiskDetector(
        features=features,
        contamination=0.05,
        risk_threshold=-0.5,
        model_dir="saved_models"
    )
    
    detector.fit(data, save_model=True)
    
    # Generate predictions
    results = detector.predict(data)
    
    # Analyze results
    risk_transactions = results[results['is_risk'] == 1]
    print(f"\nDetected {len(risk_transactions)} risky transactions:")
    print(risk_transactions[['transaction_id', 'amount', 'duration', 'risk_score', 'risk_severity']].head())
    
    # Feature importance analysis
    print("\nFeature Importances:")
    print(detector.get_feature_importances())
    
    # Save and load demonstration
    loaded_detector = IsolationForestRiskDetector.load_model(model_dir="saved_models")
    print("\nModel successfully loaded with features:", loaded_detector.features)

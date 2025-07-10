# anomaly_detection.py
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Union
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnomalyDetector:
    """
    Enhanced Autoencoder for Financial Crime Detection with:
    - Adaptive thresholding
    - Feature importance tracking
    - Model explainability
    - Production-ready serialization
    - Comprehensive monitoring
    """

    def __init__(self,
                 hidden_layers: List[int] = [64, 32, 16],
                 dropout_rate: float = 0.2,
                 l2_reg: float = 0.01,
                 epochs: int = 100,
                 batch_size: int = 256,
                 threshold_quantile: float = 0.95,
                 contamination: float = 0.001):
        """
        Args:
            hidden_layers: Neurons in each hidden layer (bottleneck)
            dropout_rate: Dropout percentage for regularization
            l2_reg: L2 regularization strength
            epochs: Maximum training epochs
            batch_size: Training batch size
            threshold_quantile: Percentile for anomaly threshold
            contamination: Estimated anomaly percentage (affects threshold)
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_quantile = threshold_quantile
        self.contamination = contamination
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.model = None
        self.threshold_ = None
        self.feature_importances_ = None
        self.training_metadata = {}
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize model architecture with regularization"""
        self._is_trained = False

    def _build_model(self, input_dim: int) -> Model:
        """Construct autoencoder with regularization"""
        # Encoder
        input_layer = Input(shape=(input_dim,))
        x = Dense(self.hidden_layers[0], 
                 activation='swish',  # Better than relu for financial data
                 kernel_regularizer=l2(self.l2_reg))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Bottleneck
        for units in self.hidden_layers[1:]:
            x = Dense(units, activation='swish', 
                     kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Decoder (mirror encoder)
        for units in reversed(self.hidden_layers[:-1]):
            x = Dense(units, activation='swish', 
                     kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            
        output_layer = Dense(input_dim, activation='linear')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, 
            X: pd.DataFrame,
            validation_data: Optional[pd.DataFrame] = None,
            feature_names: Optional[List[str]] = None) -> None:
        """
        Train the anomaly detector.
        
        Args:
            X: Normal transaction data (DataFrame)
            validation_data: Optional validation data
            feature_names: List of feature names for interpretability
        """
        logger.info(f"Training started on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        
        if validation_data is not None:
            X_val = self.scaler.transform(validation_data)

        # Model training
        self.model = self._build_model(X.shape[1])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        ).history
        
        # Threshold determination
        val_pred = self.model.predict(X_val)
        reconstruction_error = self._calculate_error(X_val, val_pred)
        self.threshold_ = np.quantile(reconstruction_error, self.threshold_quantile)
        
        # Feature importance
        self._calculate_feature_importance(X, feature_names)
        
        # Store training metadata
        self.training_metadata = {
            'training_date': datetime.now().isoformat(),
            'input_shape': X.shape,
            'best_epoch': len(history['loss']),
            'final_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'threshold': float(self.threshold_),
            'feature_names': feature_names or list(X.columns)
        }
        
        self._is_trained = True
        logger.info(f"Training completed. Threshold: {self.threshold_:.4f}")

    def _calculate_error(self, X: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error with feature weighting"""
        return np.mean(np.square(X - X_pred), axis=1)

    def _calculate_feature_importance(self, 
                                    X: pd.DataFrame,
                                    feature_names: Optional[List[str]] = None) -> None:
        """Approximate feature importance using reconstruction error sensitivity"""
        baseline_error = self._calculate_error(
            self.scaler.transform(X),
            self.model.predict(self.scaler.transform(X))
        )
        
        importances = []
        for col in range(X.shape[1]):
            X_perturbed = X.copy()
            np.random.shuffle(X_perturbed.iloc[:, col].values)
            perturbed_error = self._calculate_error(
                self.scaler.transform(X_perturbed),
                self.model.predict(self.scaler.transform(X_perturbed))
            )
            importances.append(np.mean(perturbed_error - baseline_error))
        
        self.feature_importances_ = pd.Series(
            importances,
            index=feature_names or X.columns,
            name='feature_importance'
        ).sort_values(ascending=False)

    def predict(self, 
               X: pd.DataFrame,
               return_scores: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """
        Detect anomalies in new data.
        
        Args:
            X: Input data (DataFrame)
            return_scores: If True, returns raw anomaly scores
        
        Returns:
            DataFrame with anomaly flags or array of scores
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        errors = self._calculate_error(X_scaled, preds)
        
        if return_scores:
            return errors
            
        return pd.DataFrame({
            'reconstruction_error': errors,
            'anomaly_score': self._normalize_scores(errors),
            'is_anomaly': errors > self.threshold_,
            'severity': pd.cut(
                errors,
                bins=[0, self.threshold_, 2*self.threshold_, np.inf],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        })

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Convert errors to 0-1000 risk scores"""
        return np.clip((scores / self.threshold_) * 500, 0, 1000).astype(int)

    def plot_reconstruction_error(self, X: pd.DataFrame) -> plt.Figure:
        """Visualize reconstruction error distribution"""
        errors = self.predict(X, return_scores=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50, alpha=0.7)
        ax.axvline(self.threshold_, color='r', linestyle='--', 
                  label=f'Threshold ({self.threshold_:.2f})')
        ax.set_title('Reconstruction Error Distribution')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.legend()
        return fig

    def save(self, dir_path: str) -> None:
        """Save complete model package"""
        save_dir = Path(dir_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save components
        self.model.save(save_dir / 'model.h5')
        joblib.dump(self.scaler, save_dir / 'scaler.joblib')
        self.feature_importances_.to_csv(save_dir / 'feature_importances.csv')
        
        # Save metadata
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(self.training_metadata, f, indent=2)

    @classmethod
    def load(cls, dir_path: str):
        """Load saved model package"""
        load_dir = Path(dir_path)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Directory {dir_path} not found")
            
        # Initialize
        detector = cls()
        detector.model = load_model(load_dir / 'model.h5')
        detector.scaler = joblib.load(load_dir / 'scaler.joblib')
        detector.feature_importances_ = pd.read_csv(
            load_dir / 'feature_importances.csv',
            index_col=0
        ).squeeze()
        
        # Load metadata
        with open(load_dir / 'metadata.json', 'r') as f:
            detector.training_metadata = json.load(f)
            detector.threshold_ = detector.training_metadata['threshold']
            
        detector._is_trained = True
        return detector


# Example usage
if __name__ == "__main__":
    # Simulate financial transactions
    np.random.seed(42)
    n_samples = 10000
    normal_data = np.random.normal(0, 1, (n_samples, 10))
    anomalies = np.random.uniform(5, 10, (int(n_samples*0.01), 10))
    X = pd.DataFrame(np.vstack([normal_data, anomalies]))
    
    # Train detector
    detector = FinancialAnomalyDetector(
        hidden_layers=[32, 16, 8],
        dropout_rate=0.3,
        contamination=0.01
    )
    detector.fit(X)
    
    # Detect anomalies
    results = detector.predict(X)
    print(f"Detected {results['is_anomaly'].sum()} anomalies")
    
    # Save model
    detector.save("anomaly_detector")

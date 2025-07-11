"""
Data preprocessing utilities for the Open-Source Risk Framework.

Includes comprehensive data cleaning, transformation, and feature engineering
functions tailored for risk modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Union, Optional, Dict, List
from enum import Enum


class ScalingMethod(Enum):
    """Supported scaling methods"""
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'


class MissingValueStrategy(Enum):
    """Supported missing value strategies"""
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    DROP = 'drop'
    FILL = 'fill'


class DataPreprocessing:
    """
    Comprehensive data preprocessing for risk models with support for:
    - Missing value handling
    - Feature scaling
    - Outlier detection
    - Feature engineering
    - Data validation
    """

    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.scaling_method = None

    def clean_missing(
        self,
        df: pd.DataFrame,
        strategy: Union[str, MissingValueStrategy] = MissingValueStrategy.MEAN,
        fill_value: Optional[float] = None,
        missing_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values with various strategies and validation.

        Args:
            df: Input DataFrame
            strategy: One of 'mean', 'median', 'mode', 'drop', or 'fill'
            fill_value: Value to use when strategy='fill'
            missing_threshold: Drop columns with missing ratio > threshold

        Returns:
            Cleaned DataFrame
        """
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            strategy = MissingValueStrategy(strategy.lower())

        # Drop columns with too many missing values
        missing_ratio = df.isnull().mean()
        to_drop = missing_ratio[missing_ratio > missing_threshold].index
        if len(to_drop) > 0:
            df = df.drop(columns=to_drop)

        # Apply selected strategy
        if strategy == MissingValueStrategy.DROP:
            return df.dropna()
        
        if strategy == MissingValueStrategy.FILL:
            if fill_value is None:
                raise ValueError("fill_value must be specified when strategy='fill'")
            return df.fillna(fill_value)

        # Use sklearn's SimpleImputer for other strategies
        if strategy == MissingValueStrategy.MEAN:
            self.imputer = SimpleImputer(strategy='mean')
        elif strategy == MissingValueStrategy.MEDIAN:
            self.imputer = SimpleImputer(strategy='median')
        elif strategy == MissingValueStrategy.MODE:
            self.imputer = SimpleImputer(strategy='most_frequent')

        return pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )

    def scale_features(
        self,
        df: pd.DataFrame,
        method: Union[str, ScalingMethod] = ScalingMethod.STANDARD,
        **kwargs
    ) -> pd.DataFrame:
        """
        Scale features using specified method with optional parameters.

        Args:
            df: Input DataFrame
            method: One of 'standard', 'minmax', or 'robust'
            **kwargs: Additional arguments to pass to scaler

        Returns:
            Scaled DataFrame
        """
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Convert string method to enum if needed
        if isinstance(method, str):
            method = ScalingMethod(method.lower())

        # Store feature names for inverse transform
        self.feature_names = df.columns
        self.scaling_method = method

        # Initialize appropriate scaler
        if method == ScalingMethod.STANDARD:
            self.scaler = StandardScaler(**kwargs)
        elif method == ScalingMethod.MINMAX:
            self.scaler = MinMaxScaler(**kwargs)
        elif method == ScalingMethod.ROBUST:
            self.scaler = RobustScaler(**kwargs)

        scaled_data = self.scaler.fit_transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, List[int]]:
        """
        Detect outliers in numerical columns.

        Args:
            df: Input DataFrame
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            Dictionary of {column_name: list_of_outlier_indices}
        """
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif method == 'zscore':
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                mask = np.abs(z_scores) > threshold
            else:
                raise ValueError("Invalid method. Use 'iqr' or 'zscore'")

            outliers[col] = df.index[mask].tolist()

        return outliers

    @staticmethod
    def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Add common time-based features from a datetime column.

        Args:
            df: Input DataFrame
            date_col: Name of datetime column

        Returns:
            DataFrame with added time features
        """
        if date_col not in df.columns:
            raise ValueError(f"Column {date_col} not found in DataFrame")

        df[date_col] = pd.to_datetime(df[date_col])
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['is_weekend'] = df[date_col].dt.dayofweek >= 5
        df['quarter'] = df[date_col].dt.quarter

        return df

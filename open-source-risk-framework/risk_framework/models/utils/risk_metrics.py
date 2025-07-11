"""
Comprehensive risk metrics calculation for the Open-Source Risk Framework.

Includes traditional and advanced risk measures with support for:
- Portfolio risk analysis
- Stress testing metrics
- Risk-adjusted performance measures
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy.stats import norm, t
from enum import Enum


class VaRMethod(Enum):
    """Supported VaR calculation methods"""
    HISTORICAL = 'historical'
    PARAMETRIC = 'parametric'
    MONTE_CARLO = 'monte_carlo'


class RiskMetrics:
    """
    Comprehensive risk measurement toolkit with support for:
    - Value-at-Risk (VaR) and Expected Shortfall (ES)
    - Risk-adjusted performance metrics
    - Drawdown analysis
    - Stress testing metrics
    - Advanced statistical risk measures
    """

    @staticmethod
    def value_at_risk(
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95,
        method: Union[str, VaRMethod] = VaRMethod.HISTORICAL,
        window: Optional[int] = None,
        **kwargs
    ) -> Union[float, pd.Series]:
        """
        Calculate Value-at-Risk using specified method.

        Args:
            returns: Series/array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
            window: Rolling window size for time-varying VaR
            **kwargs: Additional method-specific parameters

        Returns:
            VaR value(s)
        """
        # Convert string method to enum if needed
        if isinstance(method, str):
            method = VaRMethod(method.lower())

        if window is not None:
            return RiskMetrics._rolling_risk_measure(
                returns, 
                RiskMetrics.value_at_risk, 
                window, 
                confidence_level=confidence_level, 
                method=method, 
                **kwargs
            )

        returns = RiskMetrics._validate_returns(returns)

        if method == VaRMethod.HISTORICAL:
            return -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == VaRMethod.PARAMETRIC:
            mu = np.mean(returns)
            sigma = np.std(returns)
            distribution = kwargs.get('distribution', 'normal')
            
            if distribution == 'normal':
                return -(mu + sigma * norm.ppf(1 - confidence_level))
            elif distribution == 't':
                df = kwargs.get('degrees_of_freedom', 5)
                return -(mu + sigma * t.ppf(1 - confidence_level, df))
            else:
                raise ValueError("Invalid distribution. Use 'normal' or 't'")
        elif method == VaRMethod.MONTE_CARLO:
            n_simulations = kwargs.get('n_simulations', 10000)
            mu = np.mean(returns)
            sigma = np.std(returns)
            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            return -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        else:
            raise ValueError("Invalid VaR method")

    @staticmethod
    def expected_shortfall(
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95,
        method: Union[str, VaRMethod] = VaRMethod.HISTORICAL,
        window: Optional[int] = None,
        **kwargs
    ) -> Union[float, pd.Series]:
        """
        Calculate Expected Shortfall (CVaR) using specified method.

        Args:
            returns: Series/array of returns
            confidence_level: Confidence level
            method: 'historical', 'parametric', or 'monte_carlo'
            window: Rolling window size for time-varying ES
            **kwargs: Additional method-specific parameters

        Returns:
            ES value(s)
        """
        if window is not None:
            return RiskMetrics._rolling_risk_measure(
                returns, 
                RiskMetrics.expected_shortfall, 
                window, 
                confidence_level=confidence_level, 
                method=method, 
                **kwargs
            )

        returns = RiskMetrics._validate_returns(returns)

        if method == VaRMethod.HISTORICAL:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return -returns[returns <= var].mean()
        elif method == VaRMethod.PARAMETRIC:
            mu = np.mean(returns)
            sigma = np.std(returns)
            distribution = kwargs.get('distribution', 'normal')
            
            if distribution == 'normal':
                var = mu + sigma * norm.ppf(1 - confidence_level)
                alpha = norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)
                return -(var - sigma * alpha)
            elif distribution == 't':
                df = kwargs.get('degrees_of_freedom', 5)
                var = mu + sigma * t.ppf(1 - confidence_level, df)
                # Approximation for t-distribution ES
                g = t.pdf(t.ppf(1 - confidence_level, df), df)
                F = t.cdf(t.ppf(1 - confidence_level, df), df)
                return -(var + sigma * (g / (1 - F)) * (df + (t.ppf(1 - confidence_level, df))**2) / (df - 1))
            else:
                raise ValueError("Invalid distribution. Use 'normal' or 't'")
        elif method == VaRMethod.MONTE_CARLO:
            n_simulations = kwargs.get('n_simulations', 10000)
            mu = np.mean(returns)
            sigma = np.std(returns)
            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            return -simulated_returns[simulated_returns <= var].mean()
        else:
            raise ValueError("Invalid ES method")

    @staticmethod
    def sharpe_ratio(
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.0,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio with optional annualization.

        Args:
            returns: Series/array of returns
            risk_free_rate: Risk-free rate of return
            annualize: Whether to annualize the ratio
            periods_per_year: Trading periods per year for annualization

        Returns:
            Sharpe ratio
        """
        returns = RiskMetrics._validate_returns(returns)
        excess_returns = returns - risk_free_rate
        ratio = excess_returns.mean() / excess_returns.std()
        
        if annualize:
            return ratio * np.sqrt(periods_per_year)
        return ratio

    @staticmethod
    def sortino_ratio(
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.0,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (focuses on downside risk).

        Args:
            returns: Series/array of returns
            risk_free_rate: Risk-free rate of return
            annualize: Whether to annualize the ratio
            periods_per_year: Trading periods per year for annualization

        Returns:
            Sortino ratio
        """
        returns = RiskMetrics._validate_returns(returns)
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf  # No downside risk
        
        downside_std = downside_returns.std()
        ratio = excess_returns.mean() / downside_std
        
        if annualize:
            return ratio * np.sqrt(periods_per_year)
        return ratio

    @staticmethod
    def max_drawdown(prices: Union[pd.Series, np.ndarray]) -> float:
        """
        Compute maximum drawdown from price series.

        Args:
            prices: Series/array of prices

        Returns:
            Maximum drawdown as negative percentage
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
            
        cumulative_returns = prices / prices[0]
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

    @staticmethod
    def ulcer_index(prices: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Ulcer Index, a measure of downside risk.

        Args:
            prices: Series/array of prices

        Returns:
            Ulcer Index value
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
            
        cumulative_returns = prices / prices[0]
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        squared_drawdown = drawdown ** 2
        return np.sqrt(np.mean(squared_drawdown))

    @staticmethod
    def tail_ratio(
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate the ratio of right tail to left tail performance.

        Args:
            returns: Series/array of returns
            confidence_level: Confidence level for tail definition

        Returns:
            Tail ratio (values >1 indicate positive skew)
        """
        returns = RiskMetrics._validate_returns(returns)
        right_tail = np.percentile(returns, confidence_level * 100)
        left_tail = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(right_tail / left_tail)

    @staticmethod
    def _rolling_risk_measure(
        returns: Union[pd.Series, np.ndarray],
        risk_func: callable,
        window: int,
        **kwargs
    ) -> pd.Series:
        """
        Helper function to calculate rolling risk measures.

        Args:
            returns: Series/array of returns
            risk_func: Risk metric function to apply
            window: Rolling window size
            **kwargs: Arguments to pass to risk_func

        Returns:
            Series of rolling risk measures
        """
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
            
        return returns.rolling(window).apply(
            lambda x: risk_func(x, **kwargs),
            raw=True
        )

    @staticmethod
    def _validate_returns(returns: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Validate and convert returns to numpy array.

        Args:
            returns: Input returns

        Returns:
            Validated numpy array of returns
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        elif not isinstance(returns, np.ndarray):
            raise TypeError("Returns must be pandas Series or numpy array")
            
        if len(returns) < 2:
            raise ValueError("Returns series must have at least 2 observations")
            
        return returns

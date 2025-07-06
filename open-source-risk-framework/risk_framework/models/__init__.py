# risk_framework/models/__init__.py

# Credit Risk Models
from .credit_risk.xgboost_credit import XGBoostCreditModel
from .credit_risk.probability_of_default import ProbabilityOfDefaultModel
from .credit_risk.merton_model import MertonModel

# Operational Risk Models
from .operational_risk.rule_based import RuleBasedModel
from .operational_risk.loss_distribution import LossDistributionModel
from .operational_risk.scenario_analysis import ScenarioAnalysisModel

# Financial Crime / AML / KYC Models
from .fincrime_aml_kyc.isolation_forest import IsolationForestModel
from .fincrime_aml_kyc.anomaly_detection import AnomalyDetectionModel
from .fincrime_aml_kyc.network_analysis import NetworkAnalysisModel

# Market Risk Models
from .market_risk.var_historical import HistoricalVaRModel
from .market_risk.expected_shortfall import ExpectedShortfallModel
from .market_risk.garch_volatility import GARCHVolatilityModel

# Liquidity Risk Models
from .liquidity_risk.liquidity_coverage import LiquidityCoverageModel
from .liquidity_risk.cashflow_forecasting import CashflowForecastingModel

# Regulatory Models
from .regulatory.basel_engine import BaselEngine
from .regulatory.stress_testing import StressTestingModel

__all__ = [
    # Credit
    "XGBoostCreditModel", "ProbabilityOfDefaultModel", "MertonModel",
    # Operational
    "RuleBasedModel", "LossDistributionModel", "ScenarioAnalysisModel",
    # Financial Crime / AML / KYC
    "IsolationForestModel", "AnomalyDetectionModel", "NetworkAnalysisModel",
    # Market
    "HistoricalVaRModel", "ExpectedShortfallModel", "GARCHVolatilityModel",
    # Liquidity
    "LiquidityCoverageModel", "CashflowForecastingModel",
    # Regulatory
    "BaselEngine", "StressTestingModel"
]

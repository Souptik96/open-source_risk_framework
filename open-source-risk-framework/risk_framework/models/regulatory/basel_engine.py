# basel_engine.py
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class BaselStandard(Enum):
    BASEL_III = "III"
    BASEL_IV = "IV"

class RiskWeightCategory(Enum):
    SOVEREIGN = 0.0
    BANK = 0.2
    CORPORATE = 0.3
    RETAIL = 0.75
    HIGH_RISK = 1.5

class BaselComplianceEngine:
    """
    Advanced Basel Regulatory Calculator with:
    - Basel III/IV compliance
    - Risk-weighted asset (RWA) computation
    - Capital buffer requirements
    - Reporting templates
    """

    def __init__(self, 
                standard: BaselStandard = BaselStandard.BASEL_III,
                reporting_currency: str = "USD"):
        """
        Args:
            standard: Basel regulatory standard
            reporting_currency: Currency for calculations
        """
        self.standard = standard
        self.currency = reporting_currency
        self.positions = []
        self.capital = {
            'tier1': 0.0,
            'tier2': 0.0,
            'tier3': 0.0
        }
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'calculation_method': 'standardized'
        }

    def add_position(self,
                   asset_value: float,
                   category: RiskWeightCategory,
                   name: str = None):
        """
        Register a financial position with risk weight
        """
        self.positions.append({
            'name': name or f"position_{len(self.positions)+1}",
            'value': asset_value,
            'category': category,
            'rwa': asset_value * category.value
        })
        self._update_metadata()

    def set_capital(self,
                  tier1: float,
                  tier2: float,
                  tier3: float = 0.0):
        """
        Set capital amounts by tier
        """
        self.capital = {
            'tier1': tier1,
            'tier2': tier2,
            'tier3': tier3
        }
        self._update_metadata()

    def calculate(self) -> Dict:
        """
        Compute all regulatory metrics
        """
        total_rwa = sum(p['rwa'] for p in self.positions)
        total_capital = sum(self.capital.values())
        
        results = {
            'capital_adequacy_ratio': (total_capital / total_rwa) * 100 if total_rwa > 0 else float('inf'),
            'tier1_ratio': (self.capital['tier1'] / total_rwa) * 100 if total_rwa > 0 else float('inf'),
            'total_rwa': total_rwa,
            'capital_buffers': self._calculate_buffers(total_rwa),
            'leverage_ratio': self._calculate_leverage_ratio(),
            'liquidity_coverage': None,  # Would link to LCR module
            'nsfr': None  # Would link to NSFR module
        }
        
        self.metadata.update({
            'last_updated': datetime.now().isoformat(),
            'results': results
        })
        
        return results

    def _calculate_buffers(self, total_rwa: float) -> Dict:
        """Compute capital buffers based on Basel standards"""
        buffers = {
            'capital_conservation': 0.025 * total_rwa,
            'countercyclical': 0.0,  # Country-specific
            'g-sifi': 0.0  # For systemically important banks
        }
        
        if self.standard == BaselStandard.BASEL_IV:
            buffers['output_floor'] = 0.0725 * total_rwa
            
        return buffers

    def _calculate_leverage_ratio(self) -> float:
        """Basel III leverage ratio calculation"""
        exposure = sum(p['value'] for p in self.positions)
        return (self.capital['tier1'] / exposure) * 100 if exposure > 0 else float('inf')

    def generate_report(self, format: str = "json"):
        """Generate regulatory report"""
        report = {
            'metadata': self.metadata,
            'positions': pd.DataFrame(self.positions),
            'capital': self.capital,
            'results': self.calculate()
        }
        
        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "dataframe":
            return {
                'positions': pd.DataFrame(self.positions),
                'results': pd.DataFrame([self.calculate()])
            }
        else:
            raise ValueError("Unsupported format")

    def save_state(self, file_path: str):
        """Save current calculation state"""
        with open(file_path, 'w') as f:
            json.dump({
                'positions': self.positions,
                'capital': self.capital,
                'metadata': self.metadata
            }, f, indent=2)

    def _update_metadata(self):
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['position_count'] = len(self.positions)

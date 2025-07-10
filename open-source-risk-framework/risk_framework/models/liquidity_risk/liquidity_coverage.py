# liquidity_coverage.py
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class LiquidityCategory(Enum):
    LEVEL_1 = 1.0  # Cash, central bank reserves
    LEVEL_2A = 0.85  # Sovereign bonds
    LEVEL_2B = 0.5  # Corporate bonds
    NON_HQLA = 0.0  # Other assets

class LCRCalculator:
    """
    Advanced Liquidity Coverage Ratio (LCR) Engine with:
    - Basel III compliant calculations
    - Asset category breakdown
    - Stress scenario modeling
    - Regulatory reporting
    """

    def __init__(self, 
                reporting_currency: str = "USD",
                scenario: str = "basel_standard"):
        """
        Args:
            reporting_currency: Currency for calculations
            scenario: Stress scenario ('basel_standard', 'severe', 'custom')
        """
        self.reporting_currency = reporting_currency
        self.scenario = scenario
        self.assets = []
        self.outflows = []
        self.inflows = []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_calculated': None,
            'regulatory_standard': 'Basel III'
        }

    def add_asset(self, 
                amount: float, 
                category: LiquidityCategory,
                name: str = None):
        """
        Register a liquid asset with haircut application
        """
        self.assets.append({
            'name': name or f"asset_{len(self.assets)+1}",
            'amount': amount,
            'category': category,
            'haircut_value': amount * category.value
        })

    def add_outflow(self,
                  amount: float,
                  outflow_type: str,
                  stress_factor: float = 1.0):
        """
        Add cash outflow with scenario stress factor
        """
        self.outflows.append({
            'type': outflow_type,
            'base_amount': amount,
            'stressed_amount': amount * stress_factor
        })

    def add_inflow(self,
                 amount: float,
                 inflow_type: str,
                 cap_percentage: float = 0.75):
        """
        Add cash inflow with regulatory cap
        """
        self.inflows.append({
            'type': inflow_type,
            'base_amount': amount,
            'capped_amount': amount * cap_percentage
        })

    def calculate(self) -> Dict:
        """
        Compute LCR with detailed breakdown
        """
        total_hqla = sum(a['haircut_value'] for a in self.assets)
        total_outflows = sum(o['stressed_amount'] for o in self.outflows)
        total_inflows = sum(i['capped_amount'] for i in self.inflows)
        
        net_outflows = max(total_outflows - total_inflows, 0)
        lcr = (total_hqla / net_outflows) if net_outflows > 0 else float('inf')
        
        self.metadata.update({
            'last_calculated': datetime.now().isoformat(),
            'results': {
                'total_hqla': total_hqla,
                'total_outflows': total_outflows,
                'total_inflows': total_inflows,
                'net_outflows': net_outflows,
                'lcr': lcr
            }
        })
        
        return {
            'lcr': lcr,
            'hqla_breakdown': pd.DataFrame(self.assets),
            'cashflow_breakdown': {
                'outflows': pd.DataFrame(self.outflows),
                'inflows': pd.DataFrame(self.inflows)
            },
            'minimum_requirement': 1.0  # Basel III minimum
        }

    def stress_test(self, 
                  scenario: str = "severe") -> Dict:
        """
        Run alternative stress scenarios
        """
        scenario_factors = {
            'basel_standard': 1.0,
            'severe': 1.3,
            'lehman': 1.5
        }
        
        original_scenario = self.scenario
        self.scenario = scenario
        
        # Apply stress factors
        for outflow in self.outflows:
            outflow['stressed_amount'] *= scenario_factors.get(scenario, 1.0)
        
        results = self.calculate()
        self.scenario = original_scenario  # Reset scenario
        
        return results

    def generate_report(self, format: str = "json"):
        """Generate regulatory report"""
        report = {
            'metadata': self.metadata,
            'assets': self.assets,
            'cashflows': {
                'outflows': self.outflows,
                'inflows': self.inflows
            }
        }
        
        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "dataframe":
            return {
                'assets': pd.DataFrame(self.assets),
                'outflows': pd.DataFrame(self.outflows),
                'inflows': pd.DataFrame(self.inflows)
            }
        else:
            raise ValueError("Unsupported format")

    def save_state(self, file_path: str):
        """Save current LCR calculation state"""
        with open(file_path, 'w') as f:
            json.dump({
                'state': {
                    'assets': self.assets,
                    'outflows': self.outflows,
                    'inflows': self.inflows
                },
                'metadata': self.metadata
            }, f, indent=2)

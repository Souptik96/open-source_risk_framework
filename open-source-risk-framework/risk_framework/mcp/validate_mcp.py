#!/usr/bin/env python3
"""
MCP Compliance Validator
Validates model development against the Model Control Policy (MCP)
"""

import yaml
import json
import jsonschema
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-validator")

class MCPValidator:
    """Main validator class implementing policy checks"""
    
    def __init__(self, mcp_path: str = "mcp_template.yaml"):
        self.mcp = self._load_mcp(mcp_path)
        self.schema = self._build_jsonschema()
        self.validation_results = {
            "passed": [],
            "warnings": [],
            "failures": []
        }
        
    def _load_mcp(self, path: str) -> Dict[str, Any]:
        """Load the MCP YAML file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP: {str(e)}")
            raise
            
    def _build_jsonschema(self) -> Dict[str, Any]:
        """Generate JSON Schema from MCP for structural validation"""
        return {
            "type": "object",
            "required": ["metadata", "model_governance", "validation_framework"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["framework_version", "effective_date"]
                },
                "model_governance": {
                    "type": "object",
                    "required": ["oversight_body", "model_tiering"]
                }
            }
        }
    
    def validate_structure(self, model_config: Dict[str, Any]) -> bool:
        """Validate config against MCP schema"""
        try:
            jsonschema.validate(instance=model_config, schema=self.schema)
            self._record_result("passed", "Structural validation", "Config matches MCP schema")
            return True
        except jsonschema.ValidationError as e:
            self._record_result("failures", "Structural validation", f"Schema violation: {str(e)}")
            return False
    
    def validate_documentation(self, model_path: str) -> bool:
        """Check required documentation exists"""
        docs_required = self.mcp["development_standards"]["documentation_requirements"]
        missing = []
        
        for doc in docs_required:
            if not (Path(model_path) / f"docs/{doc.replace(' ', '_').lower()}.md").exists():
                missing.append(doc)
        
        if missing:
            self._record_result(
                "failures", 
                "Documentation check",
                f"Missing required docs: {', '.join(missing)}"
            )
            return False
        
        self._record_result("passed", "Documentation check", "All required docs present")
        return True
    
    def validate_test_coverage(self, coverage_report: Dict[str, float]) -> bool:
        """Verify test coverage meets MCP standards"""
        required = float(self.mcp["development_standards"]["testing"]["unit_test_coverage"].strip("≥%"))
        actual = coverage_report.get("unit_test_coverage", 0)
        
        if actual < required:
            self._record_result(
                "failures",
                "Test coverage",
                f"Unit test coverage {actual}% < required {required}%"
            )
            return False
            
        self._record_result("passed", "Test coverage", f"Coverage {actual}% meets requirement")
        return True
    
    def validate_monitoring(self, model_tier: str, metrics: Dict[str, float]) -> bool:
        """Check monitoring thresholds are properly configured"""
        thresholds = self.mcp["implementation_controls"]["alert_thresholds"]
        violations = []
        
        for metric, threshold_def in thresholds.items():
            if metric in metrics:
                threshold = float(threshold_def.strip(">%"))
                if ">" in threshold_def and metrics[metric] > threshold:
                    violations.append(f"{metric} exceeds {threshold_def}")
                elif "<" in threshold_def and metrics[metric] < threshold:
                    violations.append(f"{metric} below {threshold_def}")
        
        if violations:
            self._record_result(
                "warnings" if model_tier == "tier_2" else "failures",
                "Monitoring thresholds",
                f"Threshold violations: {', '.join(violations)}"
            )
            return len(violations) == 0
            
        self._record_result("passed", "Monitoring thresholds", "All metrics within bounds")
        return True
    
    def _record_result(self, category: str, check: str, message: str) -> None:
        """Store validation results"""
        self.validation_results[category].append({
            "check": check,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_report(self, output_format: str = "json") -> str:
        """Generate validation report"""
        report = {
            "validation_date": datetime.now().isoformat(),
            "results": self.validation_results,
            "summary": {
                "total_checks": sum(len(v) for v in self.validation_results.values()),
                "passed": len(self.validation_results["passed"]),
                "warnings": len(self.validation_results["warnings"]),
                "failures": len(self.validation_results["failures"])
            }
        }
        
        if output_format == "json":
            return json.dumps(report, indent=2)
        elif output_format == "yaml":
            return yaml.dump(report)
        elif output_format == "csv":
            df = pd.DataFrame([
                {**item, "category": cat} 
                for cat, items in self.validation_results.items() 
                for item in items
            ])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

# Example usage
if __name__ == "__main__":
    validator = MCPValidator()
    
    # Simulate validating a model
    model_config = {
        "metadata": {
            "model_name": "credit_risk_v1",
            "tier": "tier_1"
        },
        "documentation": {
            "purpose": "Predicts PD for commercial loans",
            "methodology": "XGBoost with SHAP explainability"
        }
    }
    
    print("\nRunning MCP validation...")
    validator.validate_structure(model_config)
    validator.validate_documentation("models/credit_risk")
    validator.validate_test_coverage({"unit_test_coverage": 87.5})
    validator.validate_monitoring("tier_1", {
        "feature_drift": 6.2,
        "accuracy": 88.0
    })
    
    print("\nValidation Report:")
    print(validator.generate_report("json"))

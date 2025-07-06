"""
Model Control Policy (MCP) Module
Provides governance and validation for risk models
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from .validate_mcp import MCPValidator

__version__ = "1.0.0"
__all__ = ["load_mcp", "validate_model", "generate_report"]

_DEFAULT_MCP_PATH = Path(__file__).parent / "mcp_template.yaml"

def load_mcp(mcp_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the MCP configuration
    
    Args:
        mcp_path: Path to MCP YAML file. Uses default if None.
    
    Returns:
        Dictionary containing MCP configuration
    
    Example:
        >>> mcp = load_mcp()
        >>> print(mcp['metadata']['framework_version'])
    """
    path = Path(mcp_path) if mcp_path else _DEFAULT_MCP_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate_model(
    model_config: Dict[str, Any],
    model_path: str,
    tier: str = "tier_1",
    coverage_report: Optional[Dict[str, float]] = None,
    metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Validate a model against the MCP
    
    Args:
        model_config: Model configuration dictionary
        model_path: Path to model directory
        tier: Model tier ('tier_1' or 'tier_2')
        coverage_report: Test coverage metrics
        metrics: Production monitoring metrics
    
    Returns:
        Dictionary with validation results
    
    Example:
        >>> results = validate_model(
        ...     model_config={'name': 'credit_risk'},
        ...     model_path='models/credit_risk',
        ...     tier='tier_1'
        ... )
    """
    validator = MCPValidator()
    
    # Core validations
    validator.validate_structure(model_config)
    validator.validate_documentation(model_path)
    
    # Conditional validations
    if coverage_report:
        validator.validate_test_coverage(coverage_report)
    if metrics:
        validator.validate_monitoring(tier, metrics)
    
    return validator.validation_results

def generate_report(
    results: Dict[str, Any],
    output_format: str = "json",
    output_file: Optional[str] = None
) -> Optional[str]:
    """
    Generate compliance report
    
    Args:
        results: Validation results from validate_model()
        output_format: 'json', 'yaml', or 'csv'
        output_file: If provided, saves to file instead of returning
    
    Returns:
        Report content if output_file is None, else None
    
    Example:
        >>> report = generate_report(results, output_format='yaml')
    """
    validator = MCPValidator()
    validator.validation_results = results
    report = validator.generate_report(output_format)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        return None
    return report

# CLI Interface
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="MCP Compliance Validator")
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument("--tier", choices=["tier_1", "tier_2"], default="tier_1")
    parser.add_argument("--format", choices=["json", "yaml", "csv"], default="json")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Simulate config - in practice would load from model_dir
    config = {
        "metadata": {
            "model_name": Path(args.model_dir).name,
            "tier": args.tier
        }
    }
    
    results = validate_model(
        model_config=config,
        model_path=args.model_dir,
        tier=args.tier
    )
    
    report = generate_report(
        results=results,
        output_format=args.format,
        output_file=args.output
    )
    
    if not args.output:
        print(report)

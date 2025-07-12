# risk_framework/reporting/report_generator.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Optional, Union, List
import jinja2
import pdfkit
import tempfile
import shutil
import json
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str
    author: str = "Risk Analytics Team"
    format: str = "html"  # html/pdf/both
    theme: str = "light"  # light/dark/custom
    include_toc: bool = True
    custom_css: Optional[str] = None

class RiskReportGenerator:
    """
    Advanced risk report generator with:
    - Multiple output formats (HTML/PDF)
    - Template-based rendering
    - Interactive visualizations
    - Automated distribution
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.template_env = jinja2.Environment(
            loader=jinja2.PackageLoader('risk_framework', 'templates/reports'),
            autoescape=True
        )
        
    def generate_report(
        self,
        config: ReportConfig,
        sections: Dict[str, Dict],
        data: Optional[Dict[str, pd.DataFrame]] = None,
        plots: Optional[Dict[str, plt.Figure]] = None,
        filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive risk report.
        
        Args:
            config: Report configuration
            sections: Dictionary of report sections with content
            data: DataFrames for tabular displays
            plots: Matplotlib figures to include
            filename: Optional base filename (without extension)
            
        Returns:
            Dictionary with paths to generated reports
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = filename or f"risk_report_{timestamp}"
            output_files = {}
            
            # Save plots to temporary directory
            plot_paths = {}
            if plots:
                plot_dir = Path(tempfile.mkdtemp())
                for name, fig in plots.items():
                    path = plot_dir / f"{name}.png"
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    plot_paths[name] = path
                    plt.close(fig)
            
            # Prepare context for template
            context = {
                "config": config,
                "sections": sections,
                "data": self._prepare_data(data) if data else {},
                "plots": plot_paths,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Generate HTML report
            if config.format in ("html", "both"):
                html_path = self.output_dir / f"{base_name}.html"
                template = self.template_env.get_template("base.html")
                html_content = template.render(context)
                
                with open(html_path, "w") as f:
                    f.write(html_content)
                output_files["html"] = str(html_path)
                logger.info(f"HTML report generated: {html_path}")
            
            # Generate PDF report
            if config.format in ("pdf", "both"):
                pdf_path = self.output_dir / f"{base_name}.pdf"
                if "html" not in output_files:
                    # Need to generate temporary HTML if PDF-only
                    temp_html = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                    template = self.template_env.get_template("base.html")
                    temp_html.write(template.render(context).encode("utf-8"))
                    temp_html.close()
                    html_source = temp_html.name
                else:
                    html_source = output_files["html"]
                
                pdfkit.from_file(
                    html_source,
                    pdf_path,
                    options={
                        "enable-local-file-access": "",
                        "quiet": ""
                    }
                )
                output_files["pdf"] = str(pdf_path)
                logger.info(f"PDF report generated: {pdf_path}")
            
            # Clean up temporary files
            if plots:
                shutil.rmtree(plot_dir)
            
            return output_files
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def _prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Convert DataFrames to HTML tables with formatting"""
        formatted = {}
        for name, df in data.items():
            # Truncate large datasets for reporting
            if len(df) > 1000:
                df = df.sample(1000)
                formatted[name] = {
                    "table": df.to_html(
                        index=False,
                        classes="table table-striped",
                        border=0
                    ),
                    "warning": "Displaying 1000 random rows from larger dataset"
                }
            else:
                formatted[name] = {
                    "table": df.to_html(
                        index=False,
                        classes="table table-striped",
                        border=0
                    )
                }
        return formatted
    
    def generate_executive_summary(
        self,
        metrics: Dict[str, Union[float, str]],
        top_risks: pd.DataFrame,
        filename: str = "executive_summary"
    ) -> str:
        """
        Generate a one-page executive summary report.
        
        Args:
            metrics: Key risk metrics
            top_risks: DataFrame of top risks to highlight
            filename: Output filename prefix
            
        Returns:
            Path to generated PDF report
        """
        config = ReportConfig(
            title="Risk Management Executive Summary",
            format="pdf",
            include_toc=False
        )
        
        sections = {
            "overview": {
                "title": "Key Metrics",
                "content": self._format_metrics(metrics)
            },
            "top_risks": {
                "title": "Top Risks",
                "content": "Highest priority risks requiring attention"
            }
        }
        
        plots = self._create_metric_charts(metrics)
        
        result = self.generate_report(
            config=config,
            sections=sections,
            data={"top_risks": top_risks},
            plots=plots,
            filename=filename
        )
        
        return result["pdf"]
    
    def _format_metrics(self, metrics: Dict[str, Union[float, str]]) -> str:
        """Format metrics for display in reports"""
        formatted = []
        for k, v in metrics.items():
            if isinstance(v, float):
                formatted.append(f"{k}: {v:.2f}")
            else:
                formatted.append(f"{k}: {v}")
        return "<br>".join(formatted)
    
    def _create_metric_charts(self, metrics: Dict) -> Dict[str, plt.Figure]:
        """Generate standard metric visualizations"""
        plots = {}
        
        # Metric distribution plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_title("Key Risk Metrics")
        plots["metric_distribution"] = fig
        
        return plots

# Example usage
if __name__ == "__main__":
    # Initialize generator
    reporter = RiskReportGenerator()
    
    # Sample data
    metrics = {
        "Total Risks": 42,
        "High Severity": 5,
        "Avg Risk Score": 72.3,
        "Coverage": 0.95
    }
    
    risk_data = pd.DataFrame({
        "Risk ID": ["R001", "R002", "R003"],
        "Category": ["Market", "Credit", "Operational"],
        "Score": [85, 92, 78]
    })
    
    # Generate full report
    report_config = ReportConfig(
        title="Quarterly Risk Assessment",
        author="Chief Risk Office",
        format="both"
    )
    
    sections = {
        "exec_summary": {
            "title": "Executive Summary",
            "content": "Quarterly overview of key risk indicators"
        },
        "detailed_analysis": {
            "title": "Detailed Analysis",
            "content": "Breakdown by risk category and business unit"
        }
    }
    
    reporter.generate_report(
        config=report_config,
        sections=sections,
        data={"risk_data": risk_data},
        plots=reporter._create_metric_charts(metrics)
    )
    
    # Generate executive summary
    reporter.generate_executive_summary(
        metrics=metrics,
        top_risks=risk_data.nlargest(5, "Score")
    )

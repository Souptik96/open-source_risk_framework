# Open Source Risk Framework

## 🌐 Purpose
The **Open Source Risk Framework** is a modular, extensible, and production-ready system for managing diverse risk domains including **credit risk**, **fraud and AML**, **operational risk**, **market risk**, **liquidity risk**, and **regulatory risk**. Built for both **transparency** and **compliance**, it empowers organizations to proactively manage financial and non-financial risks using machine learning, statistical modeling, and explainability tools.

## 🧱 Core Architecture
- **Modular Design**: Each risk domain has its own folder with domain-specific models.
- **Model Context Protocol (MCP)**: All models are documented with standardized YAML templates for reproducibility, auditability, and clarity.
- **Data Ingestion**: Supports local CSV and extensible connectors for AWS/GCP/Azure (future-ready).
- **Explainability**: Built-in SHAP-based explainability and bias auditing.
- **Dashboard & APIs**: Easily integrate with Streamlit dashboards or FastAPI-based REST endpoints.

## 🧭 Risk Domains
- `credit_risk/`: PD, LGD, XGBoost credit scoring, Merton model.
- `fraud_aml_kyc/`: Anomaly detection, Isolation Forest, Graph-based network analysis.
- `operational_risk/`: Scenario analysis and rule-based risk scoring.
- `market_risk/`: VaR, expected shortfall, volatility models.
- `liquidity_risk/`: Cashflow forecasting, LCR models.
- `regulatory/`: Basel engine, stress testing modules.

## 📁 Project Structure
```
risk_framework/
├── ingestion/              # Data ingestion from CSV/Cloud
├── models/                 # Risk models by domain
├── evaluation/             # SHAP, bias audit, and model evaluation
├── reporting/              # PDF/HTML report generation
├── mcp/                    # MCP template YAMLs and validator
├── dashboard/              # Streamlit/FastAPI dashboards
├── examples/               # Example notebooks
├── tests/                  # Unit and integration tests
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/Souptik96/open-source_risk_framework.git
cd open-source_risk_framework

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run a sample notebook
jupyter notebook examples/financial_crime_example.ipynb
```

## 📑 Model Context Protocol (MCP)
Each model in the framework comes with an MCP YAML file that captures:
- Model purpose and assumptions
- Data sources and features
- Training and evaluation metrics
- Responsible developers
- Versioning and last updated

## 🧪 Testing
```bash
pytest tests/
```

## 📈 Roadmap
- [x] Modular risk folders
- [x] Example notebooks for financial crime
- [x] MCP protocol & validation
- [ ] Streamlit dashboard
- [ ] Cloud connectors (AWS/GCP/Azure)
- [ ] PyPI packaging

## 📜 License
Apache License 2.0. See `LICENSE` file.

## 🤝 Contributing
Pull requests, feature suggestions, and model contributions are welcome! Please raise an issue or submit a PR. Community involvement is central to this project’s long-term success.

## 🌍 Created & Maintained by
**Souptik Chakraborty** – Risk, Compliance, and Data Science specialist.

> Let’s make open-source risk tools better, transparent, and widely accessible. 🎯

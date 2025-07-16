# 🛡️ Open Source Risk Framework

A modular, production-ready, open-source risk management framework for detecting and analyzing Financial Crime, Operational Risk, Market & Credit Risk, Liquidity Stress, and AI governance with built-in Model Context Protocol (MCP) support.

---

## 🎯 Project Purpose

This framework empowers compliance teams, risk analysts, data scientists, and developers to:

- Identify and mitigate **fraud, AML, and KYC risks**
- Quantify and explain **credit, market, and liquidity risk**
- Automate risk modeling with **explainable AI**
- Ensure **model transparency** with Model Context Protocol (MCP)
- Customize & extend modules with minimal overhead

---

## 🏗️ Architecture Overview

```bash
risk_framework/
├── ingestion/         # Loaders for CSV, AWS, GCP, Azure
├── models/            # Domain-specific ML models (modular)
│   ├── credit_risk/
│   ├── market_risk/
│   ├── operational_risk/
│   ├── liquidity_risk/
│   ├── fincrime_aml_kyc/
│   ├── regulatory/
│   └── utils/
├── evaluation/        # SHAP explainability, bias audit
├── reporting/         # Report generation utilities
├── mcp/               # Model Context Protocol (YAML templates + validators)
dashboard/             # Streamlit UI
examples/              # Jupyter notebooks
tests/                 # Unit tests
````

---

## 🔍 Key Features

* ✅ **Modular ML Models** for various risk domains
* 📜 **MCP (Model Context Protocol)** templates and validation
* 🔍 **Anomaly detection**, graph analysis, NLP, and SHAP
* 📊 **Streamlit Dashboard** to visualize flagged transactions
* 🧪 **Notebook Examples** for usage in real-world scenarios
* 📦 Easily extensible with plug-and-play components

---

## 📦 Installation

```bash
git clone https://github.com/Souptik96/open-source_risk_framework.git
cd open-source_risk_framework
pip install -r requirements.txt
```

---

## 🚀 Quick Start (Streamlit)

```bash
streamlit run open-source-risk-framework/dashboard/app.py
```

Upload a transaction `.csv` file and let the Isolation Forest model flag potential anomalies!

---

## 🧠 Example Usage (Notebook)

See: [`examples/financial_crime_example.ipynb`](./examples/financial_crime_example.ipynb)

```python
from risk_framework.models.fincrime_aml_kyc.isolation_forest import IsolationForestModel
model = IsolationForestModel()
model.fit(data)
predictions = model.predict(data)
```

---

## 📁 Model Context Protocol (MCP)

Each model supports machine-readable metadata via YAML to ensure auditability, reproducibility, and trust.

```yaml
model_name: IsolationForestModel
author: Souptik Chakraborty
version: 1.0
data_sources:
  - source: transactions.csv
  - schema: { amount: float, merchant_id: str, country: str }
...
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🤝 Contributing

We welcome community contributions! Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines.

---

## 📜 License

Apache 2.0. See [`LICENSE`](./LICENSE).

---

## 🌐 Stay Connected

* [LinkedIn](https://www.linkedin.com/in/souptik-chakraborty-67385a18a/)
* [Project Repo](https://github.com/Souptik96/open-source_risk_framework)
* Contact: [souptikchakraborty@gmail.com](mailto:souptikc80@gmail.com)

---

> ⭐ Star the repo if you like the project or plan to use it!

```

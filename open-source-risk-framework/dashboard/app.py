{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# risk_framework/dashboard/app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import plotly.express as px\n",
        "from pathlib import Path\n",
        "import tempfile\n",
        "import json\n",
        "from datetime import datetime\n",
        "\n",
        "# Framework imports\n",
        "from risk_framework.models.fincrime_aml_kyc.isolation_forest import IsolationForestRiskDetector\n",
        "from risk_framework.reporting.report_generator import RiskReportGenerator\n",
        "from risk_framework.mcp.validate_mcp import MCPValidator"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Configure page\n",
        "st.set_page_config(\n",
        "    page_title=\"Risk Framework Dashboard\",\n",
        "    page_icon=\"🛡️\",\n",
        "    layout=\"wide\",\n",
        "    initial_sidebar_state=\"expanded\"\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Custom CSS\n",
        "st.markdown(\"\"\"\n",
        "<style>\n",
        "    .main {padding: 2rem;}\n",
        "    .stButton>button {width: 100%;}\n",
        "    .stDownloadButton>button {width: 100%;}\n",
        "    .metric-card {border-radius: 0.5rem; padding: 1rem; background: #f0f2f6;}\n",
        "    .flagged {color: #ff4b4b; font-weight: bold;}\n",
        "</style>\n",
        "\"\"\", unsafe_allow_html=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def main():\n",
        "    st.title(\"Open Source Risk Framework Dashboard\")\n",
        "    st.markdown(\"\"\"\n",
        "    ### Financial Crime Detection Interface\n",
        "    Upload transaction data to identify suspicious activity using machine learning.\n",
        "    \"\"\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "    # Sidebar controls\n",
        "    with st.sidebar:\n",
        "        st.header(\"Configuration\")\n",
        "        contamination = st.slider(\n",
        "            \"Expected fraud rate (%)\",\n",
        "            min_value=0.1, max_value=10.0, value=1.0, step=0.1\n",
        "        ) / 100\n",
        "        risk_threshold = st.slider(\n",
        "            \"Risk threshold\",\n",
        "            min_value=-1.0, max_value=0.0, value=-0.5, step=0.1\n",
        "        )\n",
        "        selected_features = st.multiselect(\n",
        "            \"Features to analyze\",\n",
        "            options=['amount', 'duration', 'customer_age', 'transaction_hour'],\n",
        "            default=['amount', 'duration']\n",
        "        )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "    # File upload section\n",
        "    uploaded_file = st.file_uploader(\n",
        "        \"Upload transaction data (CSV)\",\n",
        "        type=[\"csv\"],\n",
        "        accept_multiple_files=False\n",
        "    )\n",
        "\n",
        "    if not uploaded_file:\n",
        "        st.info(\"Please upload a CSV file to begin analysis\")\n",
        "        st.stop()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "    # Data processing\n",
        "    @st.cache_data\n",
        "    def load_data(file):\n",
        "        return pd.read_csv(file, parse_dates=['transaction_time'])\n",
        "\n",
        "    try:\n",
        "        df = load_data(uploaded_file)\n",
        "        \n",
        "        with st.expander(\"Raw Data Preview\", expanded=False):\n",
        "            st.dataframe(df.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        # Feature engineering\n",
        "        df_processed = df.copy()\n",
        "        if 'amount' in selected_features:\n",
        "            df_processed['amount_log'] = np.log1p(df['amount'])\n",
        "        if 'transaction_hour' in selected_features:\n",
        "            df_processed['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour']/24)\n",
        "            df_processed['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour']/24)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        # Model training and prediction\n",
        "        with st.spinner(\"Training anomaly detection model...\"):\n",
        "            model = IsolationForestRiskDetector(\n",
        "                features=selected_features,\n",
        "                contamination=contamination,\n",
        "                risk_threshold=risk_threshold\n",
        "            )\n",
        "            model.fit(df_processed)\n",
        "            results = model.predict(df_processed)\n",
        "            risky_txns = results[results['is_risk'] == 1]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        # Metrics\n",
        "        col1, col2, col3 = st.columns(3)\n",
        "        with col1:\n",
        "            st.metric(\"Total Transactions\", len(results))\n",
        "        with col2:\n",
        "            st.metric(\"Flagged Transactions\", len(risky_txns))\n",
        "        with col3:\n",
        "            st.metric(\"Detection Rate\", f\"{len(risky_txns)/len(results):.2%}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        # Visualization\n",
        "        tab1, tab2, tab3 = st.tabs([\"Risk Distribution\", \"Feature Analysis\", \"Flagged Transactions\"])\n",
        "\n",
        "        with tab1:\n",
        "            fig = px.histogram(\n",
        "                results,\n",
        "                x='risk_score',\n",
        "                nbins=50,\n",
        "                title='Transaction Risk Scores'\n",
        "            )\n",
        "            fig.add_vline(x=500, line_dash=\"dash\", line_color=\"red\")\n",
        "            st.plotly_chart(fig, use_container_width=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        with tab2:\n",
        "            st.pyplot(model.plot_feature_importance(top_n=5))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        with tab3:\n",
        "            st.dataframe(\n",
        "                risky_txns.style.applymap(\n",
        "                    lambda x: \"background-color: #ffcccc\" if x == \"HIGH\" else \"\",\n",
        "                    subset=['risk_severity']\n",
        "                ),\n",
        "                height=400\n",
        "            )\n",
        "\n",
        "            # Download options\n",
        "            csv = risky_txns.to_csv(index=False).encode('utf-8')\n",
        "            st.download_button(\n",
        "                label=\"Download Flagged Transactions\",\n",
        "                data=csv,\n",
        "                file_name=f\"flagged_transactions_{datetime.now().date()}.csv\",\n",
        "                mime=\"text/csv\"\n",
        "            )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        # Report generation\n",
        "        if st.button(\"Generate Full Risk Report\"):\n",
        "            with st.spinner(\"Generating report...\"):\n",
        "                reporter = RiskReportGenerator()\n",
        "                report = reporter.generate_html_report(\n",
        "                    title=\"Financial Crime Analysis\",\n",
        "                    summary={\n",
        "                        \"Total Transactions\": len(results),\n",
        "                        \"Flagged Transactions\": len(risky_txns),\n",
        "                        \"Max Risk Score\": risky_txns['risk_score'].max()\n",
        "                    },\n",
        "                    data=risky_txns\n",
        "                )\n",
        "                st.success(f\"Report generated: {report}\")\n",
        "                with open(report, \"rb\") as f:\n",
        "                    st.download_button(\n",
        "                        label=\"Download Full Report\",\n",
        "                        data=f,\n",
        "                        file_name=\"risk_analysis_report.html\"\n",
        "                    )\n",
        "\n",
        "    except Exception as e:\n",
        "        st.error(f\"An error occurred: {str(e)}\")\n",
        "        st.stop()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "if __name__ == \"__main__\":\n",
        "    main()"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}

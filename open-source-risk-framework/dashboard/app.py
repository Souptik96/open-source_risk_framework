# risk_framework/dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Framework imports
from risk_framework.models.fincrime_aml_kyc.isolation_forest import IsolationForestRiskDetector
from risk_framework.reporting.report_generator import RiskReportGenerator
from risk_framework.mcp.validate_mcp import MCPValidator

# Configure page
st.set_page_config(
    page_title="Risk Framework Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%;}
    .stDownloadButton>button {width: 100%;}
    .metric-card {border-radius: 0.5rem; padding: 1rem; background: #f0f2f6;}
    .flagged {color: #ff4b4b; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Open Source Risk Framework Dashboard")
    st.markdown("""
    ### Financial Crime Detection Interface
    Upload transaction data to identify suspicious activity using machine learning.
    """)

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        contamination = st.slider(
            "Expected fraud rate (%)",
            min_value=0.1, max_value=10.0, value=1.0, step=0.1
        ) / 100
        risk_threshold = st.slider(
            "Risk threshold",
            min_value=-1.0, max_value=0.0, value=-0.5, step=0.1
        )
        selected_features = st.multiselect(
            "Features to analyze",
            options=['amount', 'duration', 'customer_age', 'transaction_hour'],
            default=['amount', 'duration']
        )

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload transaction data (CSV)",
        type=["csv"],
        accept_multiple_files=False
    )

    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()

    # Data processing
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file, parse_dates=['transaction_time'])

    try:
        df = load_data(uploaded_file)
        
        with st.expander("Raw Data Preview", expanded=False):
            st.dataframe(df.head())

        # Feature engineering
        df_processed = df.copy()
        if 'amount' in selected_features:
            df_processed['amount_log'] = np.log1p(df['amount'])
        if 'transaction_hour' in selected_features:
            df_processed['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour']/24)
            df_processed['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour']/24)

        # Model training and prediction
        with st.spinner("Training anomaly detection model..."):
            model = IsolationForestRiskDetector(
                features=selected_features,
                contamination=contamination,
                risk_threshold=risk_threshold
            )
            model.fit(df_processed)
            results = model.predict(df_processed)
            risky_txns = results[results['is_risk'] == 1]

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(results))
        with col2:
            st.metric("Flagged Transactions", len(risky_txns))
        with col3:
            st.metric("Detection Rate", f"{len(risky_txns)/len(results):.2%}")

        # Visualization
        tab1, tab2, tab3 = st.tabs(["Risk Distribution", "Feature Analysis", "Flagged Transactions"])

        with tab1:
            fig = px.histogram(
                results,
                x='risk_score',
                nbins=50,
                title='Transaction Risk Scores'
            )
            fig.add_vline(x=500, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.pyplot(model.plot_feature_importance(top_n=5))

        with tab3:
            st.dataframe(
                risky_txns.style.applymap(
                    lambda x: "background-color: #ffcccc" if x == "HIGH" else "",
                    subset=['risk_severity']
                ),
                height=400
            )

            # Download options
            csv = risky_txns.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Flagged Transactions",
                data=csv,
                file_name=f"flagged_transactions_{datetime.now().date()}.csv",
                mime="text/csv"
            )

        # Report generation
        if st.button("Generate Full Risk Report"):
            with st.spinner("Generating report..."):
                reporter = RiskReportGenerator()
                report = reporter.generate_html_report(
                    title="Financial Crime Analysis",
                    summary={
                        "Total Transactions": len(results),
                        "Flagged Transactions": len(risky_txns),
                        "Max Risk Score": risky_txns['risk_score'].max()
                    },
                    data=risky_txns
                )
                st.success(f"Report generated: {report}")
                with open(report, "rb") as f:
                    st.download_button(
                        label="Download Full Report",
                        data=f,
                        file_name="risk_analysis_report.html"
                    )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()

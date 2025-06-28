### Risk Matrix (Likelihood × Impact)

import pandas as pd

# Load operational risk data
data = pd.read_csv('operational_risk_data.csv')

# Basic risk scoring: Likelihood (1-5) × Impact (1-5)
def risk_matrix_score(row):
    return row['likelihood'] * row['impact']

data['risk_score'] = data.apply(risk_matrix_score, axis=1)

# Output top risks
top_risks = data.sort_values('risk_score', ascending=False).head(5)
print("Top 5 Operational Risks (Risk Matrix):")
print(top_risks[['risk_id', 'description', 'likelihood', 'impact', 'risk_score']])

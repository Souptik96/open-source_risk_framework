import pandas as pd

# Load the scored operational risk data
data = pd.read_csv('operational_risk_data.csv')

# If using the basic or advanced models, ensure 'risk_score' column exists
if 'risk_score' not in data.columns:
    print("No 'risk_score' column found. Please run the model snippet first.")
else:
    # Generate summary statistics
    print("Operational Risk Assessment Report")
    print("="*40)
    print(f"Total risks assessed: {len(data)}")
    print(f"Mean risk score: {data['risk_score'].mean():.2f}")
    print(f"Max risk score: {data['risk_score'].max():.2f}")
    print(f"Min risk score: {data['risk_score'].min():.2f}")

    # Top 5 highest risk events
    top_risks = data.sort_values('risk_score', ascending=False).head(5)
    print("\nTop 5 Operational Risks:")
    print(top_risks[['risk_id', 'description', 'likelihood', 'impact', 'risk_score']])

    # Export report
    top_risks.to_csv('operational_risk_report.csv', index=False)
    print("\nReport saved to operational_risk_report.csv")

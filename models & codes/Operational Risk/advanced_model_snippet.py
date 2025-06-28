### Decision Tree Classifier

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and prepare data
data = pd.read_csv('operational_risk_data.csv')
features = ['likelihood', 'impact', 'duration_hours', 'incident_type_code']
X = data[features]
y = data['critical_flag']  # 1 = critical risk event, 0 = non-critical

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred))

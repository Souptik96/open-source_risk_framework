# risk_framework/models/rule_based.py

import pandas as pd
from typing import List, Dict, Any

class RuleBasedModel:
    def __init__(self, rules: List[Dict[str, Any]]):
        """
        Initialize the rule-based model.

        :param rules: A list of rules. Each rule is a dict with keys:
                      - "column": column name to apply the rule
                      - "operator": comparison operator as string ('>', '<', '==', etc.)
                      - "value": value to compare against
                      - "label": risk label to assign if rule matches
        """
        self.rules = rules

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rules to a DataFrame and return with a 'risk_flag' column.

        :param df: Input DataFrame
        :return: DataFrame with 'risk_flag' column
        """
        df = df.copy()
        df["risk_flag"] = "LOW_RISK"  # Default value

        for rule in self.rules:
            mask = self._evaluate_rule(df, rule)
            df.loc[mask, "risk_flag"] = rule.get("label", "HIGH_RISK")

        return df

    def _evaluate_rule(self, df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
        """
        Evaluate a single rule on the DataFrame.

        :param df: Input DataFrame
        :param rule: Rule dictionary
        :return: Boolean Series where rule matches
        """
        col = rule["column"]
        op = rule["operator"]
        val = rule["value"]

        if op == ">":
            return df[col] > val
        elif op == "<":
            return df[col] < val
        elif op == "==":
            return df[col] == val
        elif op == "!=":
            return df[col] != val
        elif op == ">=":
            return df[col] >= val
        elif op == "<=":
            return df[col] <= val
        else:
            raise ValueError(f"Unsupported operator: {op}")


# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({
        "id": [1, 2, 3],
        "amount": [5000, 15000, 2000],
        "country": ["US", "PK", "UK"]
    })

    rules = [
        {"column": "amount", "operator": ">", "value": 10000, "label": "HIGH_AMOUNT"},
        {"column": "country", "operator": "==", "value": "PK", "label": "HIGH_RISK_COUNTRY"}
    ]

    model = RuleBasedModel(rules)
    flagged = model.apply(data)
    print(flagged)

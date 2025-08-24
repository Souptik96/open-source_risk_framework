import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest
import openai

@dataclass
class Transaction:
    amount: float
    currency: str
    source_country: str
    destination_country: str
    customer_id: str
    timestamp: str
    transaction_type: str
    
@dataclass
class RiskAssessment:
    risk_score: float
    confidence: float
    risk_level: str
    explanation: str
    suggested_action: str

class AIAMLPolicyEngine:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.policy_llm = openai.OpenAI()
        self.risk_thresholds = {
            "LOW": 0.3,
            "MEDIUM": 0.6, 
            "HIGH": 0.85
        }
    
    def analyze_transaction(self, transaction: Transaction, 
                          customer_history: List[Dict]) -> RiskAssessment:
        """AI-enhanced transaction analysis"""
        
        # Feature engineering
        features = self._extract_features(transaction, customer_history)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector.decision_function([features])[0]
        
        # Contextual AI analysis
        context_analysis = self._analyze_context(transaction, customer_history)
        
        # Combine scores
        final_score = self._combine_scores(anomaly_score, context_analysis)
        
        # Generate explanation
        explanation = self._generate_explanation(transaction, final_score, context_analysis)
        
        risk_level = self._determine_risk_level(final_score)
        
        return RiskAssessment(
            risk_score=final_score,
            confidence=context_analysis.get('confidence', 0.8),
            risk_level=risk_level,
            explanation=explanation,
            suggested_action=self._suggest_action(risk_level)
        )
    
    def _extract_features(self, transaction: Transaction, 
                         history: List[Dict]) -> np.ndarray:
        """Extract ML features from transaction and history"""
        features = [
            transaction.amount,
            len(history),  # Transaction frequency
            self._country_risk_score(transaction.destination_country),
            self._time_based_features(transaction.timestamp),
            self._amount_deviation_from_history(transaction.amount, history)
        ]
        return np.array(features)
    
    def _analyze_context(self, transaction: Transaction, 
                        history: List[Dict]) -> Dict:
        """Use LLM for contextual analysis"""
        
        prompt = f"""
        Analyze this transaction for AML risk:
        
        Transaction: {transaction.amount} {transaction.currency}
        From: {transaction.source_country} To: {transaction.destination_country}
        Type: {transaction.transaction_type}
        
        Customer History: {len(history)} previous transactions
        Average amount: {np.mean([t.get('amount', 0) for t in history])}
        
        Consider: typologies, jurisdictional risks, behavioral patterns.
        Return JSON with risk_factors, confidence, and reasoning.
        """
        
        response = self.policy_llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"confidence": 0.5, "risk_factors": []}
    
    def update_policy_from_feedback(self, transaction_id: str, 
                                  was_false_positive: bool):
        """Continuous learning from feedback"""
        # Update model weights based on feedback
        # This would integrate with your ML pipeline
        pass
    
    def _determine_risk_level(self, score: float) -> str:
        if score >= self.risk_thresholds["HIGH"]:
            return "HIGH"
        elif score >= self.risk_thresholds["MEDIUM"]:
            return "MEDIUM"
        return "LOW"
    
    def _suggest_action(self, risk_level: str) -> str:
        actions = {
            "HIGH": "IMMEDIATE_REVIEW_REQUIRED",
            "MEDIUM": "ENHANCED_DUE_DILIGENCE", 
            "LOW": "ROUTINE_MONITORING"
        }
        return actions.get(risk_level, "ROUTINE_MONITORING")

# Policy Engine Integration
class PolicyAsCodeEngine:
    def __init__(self):
        self.ai_engine = AIAMLPolicyEngine()
        self.policies = {}
    
    def evaluate_transaction(self, transaction: Transaction) -> Dict:
        """Main policy evaluation entry point"""
        
        # Get customer history
        history = self._get_customer_history(transaction.customer_id)
        
        # AI-enhanced analysis
        ai_assessment = self.ai_engine.analyze_transaction(transaction, history)
        
        # Apply traditional rule-based policies
        rule_results = self._apply_traditional_rules(transaction)
        
        # Combine results
        final_decision = self._combine_assessments(ai_assessment, rule_results)
        
        return {
            "transaction_id": f"tx_{transaction.timestamp}_{transaction.customer_id}",
            "risk_assessment": ai_assessment.__dict__,
            "rule_results": rule_results,
            "final_decision": final_decision,
            "compliance_status": "FLAGGED" if final_decision["requires_review"] else "CLEARED"
        }

# Usage Example
if __name__ == "__main__":
    # Initialize policy engine
    policy_engine = PolicyAsCodeEngine()
    
    # Sample transaction
    transaction = Transaction(
        amount=15000.0,
        currency="USD",
        source_country="US",
        destination_country="PK",
        customer_id="CUST_12345",
        timestamp="2024-01-15T10:30:00Z",
        transaction_type="WIRE_TRANSFER"
    )
    
    # Evaluate transaction
    result = policy_engine.evaluate_transaction(transaction)
    print(json.dumps(result, indent=2))

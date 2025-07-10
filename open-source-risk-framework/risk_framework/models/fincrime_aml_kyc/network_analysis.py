# network_analysis.py
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
from enum import Enum
import community as community_louvain  # python-louvain package
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class FinancialNetworkAnalyzer:
    """
    Advanced Transaction Network Analysis for Financial Crime Detection with:
    - Dynamic risk scoring of entities and transactions
    - Community detection for fraud ring identification
    - Temporal pattern analysis
    - Production-ready serialization
    - Comprehensive visualization
    """

    def __init__(self, 
                 min_amount_threshold: float = 1000,
                 time_window: str = '7D',
                 risk_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            min_amount_threshold: Minimum amount to consider for high-risk edges
            time_window: Time window for temporal analysis (pandas offset)
            risk_weights: Custom weights for risk calculations
        """
        self.graph = nx.DiGraph()
        self.min_amount = min_amount_threshold
        self.time_window = time_window
        self.risk_weights = risk_weights or {
            'amount': 0.4,
            'frequency': 0.3,
            'centrality': 0.2,
            'community': 0.1
        }
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initialize analysis metadata"""
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'graph_stats': {},
            'risk_parameters': self.risk_weights
        }

    def build_graph(self, 
                   transactions: pd.DataFrame,
                   entity_cols: Optional[List[str]] = None,
                   timestamp_col: str = 'timestamp') -> None:
        """
        Builds enriched transaction graph with temporal and entity attributes.
        
        Args:
            transactions: DataFrame containing transaction records
            entity_cols: Additional entity attributes to include (e.g., ['country', 'occupation'])
            timestamp_col: Column containing transaction timestamps
        """
        logger.info(f"Building graph from {len(transactions)} transactions")
        
        # Convert timestamps if needed
        if timestamp_col in transactions.columns:
            transactions[timestamp_col] = pd.to_datetime(transactions[timestamp_col])
        
        # Create weighted edges with attributes
        edges = []
        for _, row in transactions.iterrows():
            edge_attrs = {
                'amount': row.get('amount', 0),
                'timestamp': row.get(timestamp_col),
                'currency': row.get('currency', 'USD')
            }
            if entity_cols:
                for col in entity_cols:
                    edge_attrs[col] = row.get(col)
            edges.append((row['sender'], row['receiver'], edge_attrs))
        
        self.graph.add_edges_from(edges)
        self._update_metadata()
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def _update_metadata(self) -> None:
        """Update graph metadata statistics"""
        self.metadata.update({
            'last_updated': datetime.now().isoformat(),
            'graph_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'is_directed': self.graph.is_directed(),
                'density': nx.density(self.graph),
                'components': nx.number_weakly_connected_components(self.graph)
            }
        })

    def detect_risk_patterns(self) -> Dict[str, List]:
        """
        Comprehensive financial crime pattern detection.
        
        Returns:
            Dictionary containing detected patterns
        """
        patterns = {
            'cycles': self._detect_cycles(),
            'high_risk_communities': self._detect_suspicious_communities(),
            'central_hubs': self._identify_central_entities(),
            'rapid_transfers': self._detect_rapid_transfers()
        }
        return patterns

    def _detect_cycles(self, min_length: int = 3) -> List[List[str]]:
        """Detect circular transaction flows"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return [cycle for cycle in cycles if len(cycle) >= min_length]
        except nx.NetworkXNoCycle:
            return []

    def _detect_suspicious_communities(self, 
                                      min_risk_score: float = 0.7) -> List[Dict]:
        """Detect high-risk communities using Louvain method"""
        undirected_graph = self.graph.to_undirected()
        partition = community_louvain.best_partition(undirected_graph)
        
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        
        suspicious = []
        for comm_id, members in communities.items():
            if len(members) > 2:  # Ignore trivial communities
                risk_score = self._calculate_community_risk(members)
                if risk_score >= min_risk_score:
                    suspicious.append({
                        'community_id': comm_id,
                        'members': members,
                        'risk_score': risk_score,
                        'risk_factors': self._analyze_community_risk(members)
                    })
        
        return sorted(suspicious, key=lambda x: x['risk_score'], reverse=True)

    def _calculate_community_risk(self, members: List[str]) -> float:
        """Calculate composite risk score for a community"""
        total_amount = sum(
            data['amount'] 
            for u, v, data in self.graph.edges(members, data=True)
            if 'amount' in data
        )
        return min(total_amount / (self.min_amount * 100), 1.0)

    def _analyze_community_risk(self, members: List[str]) -> Dict[str, float]:
        """Analyze risk factors for a community"""
        factors = {
            'total_amount': 0,
            'transaction_count': 0,
            'countries': set(),
            'currencies': set()
        }
        
        for u, v, data in self.graph.edges(members, data=True):
            factors['total_amount'] += data.get('amount', 0)
            factors['transaction_count'] += 1
            if 'country' in data:
                factors['countries'].add(data['country'])
            if 'currency' in data:
                factors['currencies'].add(data['currency'])
        
        return factors

    def _identify_central_entities(self, 
                                 top_n: int = 10,
                                 min_degree: int = 5) -> List[Tuple[str, Dict]]:
        """Identify high-risk central entities using multiple centrality measures"""
        candidates = [
            node for node, degree in self.graph.degree()
            if degree >= min_degree
        ]
        
        if not candidates:
            return []
        
        centrality_measures = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph)
        }
        
        ranked_entities = []
        for node in candidates:
            scores = {
                measure: values[node] * weight
                for measure, values in centrality_measures.items()
                for weight in [self.risk_weights.get(measure, 1.0)]
            }
            composite_score = sum(scores.values())
            ranked_entities.append((node, {
                'composite_score': composite_score,
                **scores,
                'degree': self.graph.degree(node)
            }))
        
        return sorted(ranked_entities, key=lambda x: x[1]['composite_score'], reverse=True)[:top_n]

    def _detect_rapid_transfers(self, 
                               time_threshold: str = '1H') -> List[Dict]:
        """Detect rapid movement of funds between accounts"""
        rapid_transfers = []
        
        for node in self.graph.nodes():
            outgoing = []
            for _, _, data in self.graph.out_edges(node, data=True):
                if 'timestamp' in data:
                    outgoing.append(data['timestamp'])
            
            if len(outgoing) > 1:
                time_diff = max(outgoing) - min(outgoing)
                if time_diff <= pd.Timedelta(time_threshold):
                    amount = sum(
                        data['amount']
                        for _, _, data in self.graph.out_edges(node, data=True)
                        if 'amount' in data
                    )
                    rapid_transfers.append({
                        'node': node,
                        'time_window': str(time_diff),
                        'transaction_count': len(outgoing),
                        'total_amount': amount
                    })
        
        return sorted(rapid_transfers, key=lambda x: x['total_amount'], reverse=True)

    def visualize_network(self, 
                         highlight_nodes: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """Generate network visualization with risk highlights"""
        fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.spring_layout(self.graph)
        node_colors = []
        
        for node in self.graph.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, ax=ax)
        
        if highlight_nodes:
            nx.draw_networkx_labels(
                self.graph, 
                pos,
                labels={n: n for n in highlight_nodes},
                font_color='darkred',
                ax=ax
            )
        
        ax.set_title("Transaction Network with Risk Highlights")
        plt.tight_layout()
        return fig

    def save_graph(self, file_path: str) -> None:
        """Save graph and metadata to disk"""
        save_path = Path(file_path)
        save_path.mkdir(exist_ok=True)
        
        # Save graph
        nx.write_graphml(self.graph, save_path / "transaction_graph.graphml")
        
        # Save metadata
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Saved network to {file_path}")

    @classmethod
    def load_graph(cls, file_path: str):
        """Load saved network analysis"""
        load_path = Path(file_path)
        
        analyzer = cls()
        analyzer.graph = nx.read_graphml(load_path / "transaction_graph.graphml")
        
        with open(load_path / "metadata.json", 'r') as f:
            analyzer.metadata = json.load(f)
        
        logger.info(f"Loaded network with {analyzer.graph.number_of_nodes()} nodes")
        return analyzer


# Example usage
if __name__ == "__main__":
    # Sample transaction data
    transactions = pd.DataFrame({
        'sender': ['A', 'B', 'C', 'D', 'A', 'B', 'E', 'F', 'A'],
        'receiver': ['B', 'C', 'A', 'A', 'D', 'E', 'F', 'A', 'F'],
        'amount': [5000, 15000, 5000, 20000, 3000, 8000, 12000, 5000, 10000],
        'timestamp': pd.date_range(start='2023-01-01', periods=9, freq='H'),
        'country': ['US', 'PK', 'US', 'CA', 'US', 'PK', 'RU', 'US', 'US']
    })
    
    # Initialize analyzer
    analyzer = FinancialNetworkAnalyzer(min_amount_threshold=10000)
    analyzer.build_graph(transactions, entity_cols=['country'])
    
    # Detect patterns
    patterns = analyzer.detect_risk_patterns()
    print(f"Detected {len(patterns['cycles'])} circular transaction flows")
    print(f"Found {len(patterns['high_risk_communities'])} suspicious communities")
    
    # Visualize
    high_risk_nodes = list(set(
        node 
        for comm in patterns['high_risk_communities'] 
        for node in comm['members']
    ))
    fig = analyzer.visualize_network(highlight_nodes=high_risk_nodes)
    plt.savefig('network.png')
    
    # Save analysis
    analyzer.save_graph("transaction_network_analysis")

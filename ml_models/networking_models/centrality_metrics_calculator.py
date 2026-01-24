"""
GrainSecure PDS Monitoring System
Centrality Metrics Calculator for Network Importance Analysis

This module implements comprehensive centrality metric calculations to identify
fraud network leaders, coordinators, and influential nodes for targeted
enforcement and network disruption strategies.

Author: GrainSecure Development Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import (
    ndcg_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import json
import time
from typing import Dict, List, Tuple, Set, Optional

warnings.filterwarnings('ignore')
np.random.seed(42)


class HierarchicalFraudNetworkGenerator:
    """
    Generates fraud networks with explicit leadership hierarchies for validating
    centrality metric performance in identifying influential nodes.
    """
    
    def __init__(self, n_beneficiaries: int = 2000, n_shops: int = 150,
                 n_fraud_rings: int = 10, leader_ratio: float = 0.15):
        """
        Initialize hierarchical fraud network generator.
        
        Args:
            n_beneficiaries: Total beneficiary nodes
            n_shops: Total shop nodes
            n_fraud_rings: Number of distinct fraud rings
            leader_ratio: Proportion of ring members who are leaders
        """
        self.n_beneficiaries = n_beneficiaries
        self.n_shops = n_shops
        self.n_fraud_rings = n_fraud_rings
        self.leader_ratio = leader_ratio
        
        self.graph = None
        self.fraud_rings = []
        self.leaders = set()
        self.coordinators = set()
        self.bridges = set()
        
    def generate_hierarchical_network(self) -> Tuple[nx.Graph, Set, Set, Set]:
        """
        Generate fraud network with hierarchical structure.
        
        Returns:
            Tuple of (graph, leaders, coordinators, bridges)
        """
        
        print("=" * 80)
        print("GENERATING HIERARCHICAL FRAUD NETWORK")
        print("=" * 80)
        
        self.graph = nx.Graph()
        
        # Add nodes
        self._add_nodes()
        
        # Create fraud rings with leadership
        self._create_hierarchical_rings()
        
        # Add legitimate connections
        self._add_legitimate_transactions()
        
        # Create bridge nodes between rings
        self._create_bridge_connections()
        
        print(f"\n✓ Network generation complete")
        print(f"  Total nodes: {self.graph.number_of_nodes():,}")
        print(f"  Total edges: {self.graph.number_of_edges():,}")
        print(f"  Fraud rings: {len(self.fraud_rings)}")
        print(f"  Leader nodes: {len(self.leaders):,}")
        print(f"  Coordinator nodes: {len(self.coordinators):,}")
        print(f"  Bridge nodes: {len(self.bridges):,}")
        print("=" * 80)
        
        return self.graph, self.leaders, self.coordinators, self.bridges
    
    def _add_nodes(self):
        """Add all nodes with attributes."""
        
        for i in range(self.n_beneficiaries):
            self.graph.add_node(
                f'B_{i}',
                node_type='beneficiary',
                family_size=np.random.randint(1, 11),
                is_fraudulent=0,
                is_leader=0,
                is_coordinator=0,
                is_bridge=0
            )
        
        for i in range(self.n_shops):
            self.graph.add_node(
                f'S_{i}',
                node_type='shop',
                compliance_score=np.random.uniform(40, 100),
                is_complicit=0
            )
        
        print(f"✓ Added {self.n_beneficiaries:,} beneficiaries")
        print(f"✓ Added {self.n_shops:,} shops")
    
    def _create_hierarchical_rings(self):
        """Create fraud rings with clear leadership hierarchies."""
        
        available_beneficiaries = list(range(self.n_beneficiaries))
        np.random.shuffle(available_beneficiaries)
        
        for ring_id in range(self.n_fraud_rings):
            # Ring size
            ring_size = np.random.randint(8, 35)
            
            if len(available_beneficiaries) < ring_size:
                break
            
            # Select ring members
            member_indices = available_beneficiaries[:ring_size]
            available_beneficiaries = available_beneficiaries[ring_size:]
            
            members = [f'B_{i}' for i in member_indices]
            
            # Designate leaders (15% of ring)
            n_leaders = max(1, int(ring_size * self.leader_ratio))
            ring_leaders = set(np.random.choice(members, n_leaders, replace=False))
            
            # Designate coordinators (next 20%)
            remaining = [m for m in members if m not in ring_leaders]
            n_coordinators = max(1, int(len(remaining) * 0.20))
            ring_coordinators = set(np.random.choice(remaining, n_coordinators, replace=False))
            
            # Regular members
            regular_members = [m for m in members if m not in ring_leaders and m not in ring_coordinators]
            
            # Select complicit shops
            n_shops = np.random.randint(1, 4)
            shop_indices = np.random.choice(self.n_shops, n_shops, replace=False)
            shops = [f'S_{i}' for i in shop_indices]
            
            # Mark nodes
            for leader in ring_leaders:
                self.graph.nodes[leader]['is_fraudulent'] = 1
                self.graph.nodes[leader]['is_leader'] = 1
                self.graph.nodes[leader]['ring_id'] = ring_id
                self.leaders.add(leader)
            
            for coord in ring_coordinators:
                self.graph.nodes[coord]['is_fraudulent'] = 1
                self.graph.nodes[coord]['is_coordinator'] = 1
                self.graph.nodes[coord]['ring_id'] = ring_id
                self.coordinators.add(coord)
            
            for member in regular_members:
                self.graph.nodes[member]['is_fraudulent'] = 1
                self.graph.nodes[member]['ring_id'] = ring_id
            
            for shop in shops:
                self.graph.nodes[shop]['is_complicit'] = 1
                self.graph.nodes[shop]['ring_id'] = ring_id
            
            # Create hierarchical connections
            # Leaders connect to all coordinators and shops
            for leader in ring_leaders:
                for coord in ring_coordinators:
                    self.graph.add_edge(
                        leader, coord,
                        weight=np.random.uniform(10, 20),
                        edge_type='leadership'
                    )
                for shop in shops:
                    self.graph.add_edge(
                        leader, shop,
                        weight=np.random.uniform(15, 25),
                        edge_type='leadership_transaction'
                    )
            
            # Coordinators connect to regular members and shops
            for coord in ring_coordinators:
                # Connect to 3-6 regular members
                n_connections = min(len(regular_members), np.random.randint(3, 7))
                connected_members = np.random.choice(regular_members, n_connections, replace=False)
                
                for member in connected_members:
                    self.graph.add_edge(
                        coord, member,
                        weight=np.random.uniform(5, 15),
                        edge_type='coordination'
                    )
                
                # Connect to shops
                for shop in shops:
                    if np.random.random() < 0.8:
                        self.graph.add_edge(
                            coord, shop,
                            weight=np.random.uniform(8, 18),
                            edge_type='coordination_transaction'
                        )
            
            # Regular members connect primarily to shops
            for member in regular_members:
                n_shop_connections = np.random.randint(1, 3)
                connected_shops = np.random.choice(shops, n_shop_connections, replace=False)
                
                for shop in connected_shops:
                    self.graph.add_edge(
                        member, shop,
                        weight=np.random.uniform(3, 12),
                        edge_type='member_transaction'
                    )
            
            # Store ring info
            self.fraud_rings.append({
                'ring_id': ring_id,
                'members': set(members + shops),
                'leaders': ring_leaders,
                'coordinators': ring_coordinators,
                'shops': set(shops)
            })
        
        print(f"✓ Created {len(self.fraud_rings)} hierarchical fraud rings")
    
    def _add_legitimate_transactions(self):
        """Add legitimate background transactions."""
        
        beneficiaries = [n for n in self.graph.nodes() if n.startswith('B_')]
        shops = [n for n in self.graph.nodes() if n.startswith('S_')]
        
        for beneficiary in beneficiaries:
            if self.graph.nodes[beneficiary]['is_fraudulent'] == 0:
                n_shops = np.random.randint(1, 3)
                selected_shops = np.random.choice(shops, n_shops, replace=False)
                
                for shop in selected_shops:
                    self.graph.add_edge(
                        beneficiary, shop,
                        weight=np.random.uniform(1, 5),
                        edge_type='legitimate'
                    )
        
        print(f"✓ Added legitimate transactions")
    
    def _create_bridge_connections(self):
        """Create bridge nodes connecting different fraud rings."""
        
        if len(self.fraud_rings) < 2:
            return
        
        # Select 2-3 beneficiaries to act as bridges
        n_bridges = min(3, len(self.fraud_rings) - 1)
        
        for _ in range(n_bridges):
            # Select two different rings
            ring1, ring2 = np.random.choice(self.fraud_rings, 2, replace=False)
            
            # Select a leader from ring1 to be the bridge
            if ring1['leaders']:
                bridge_node = np.random.choice(list(ring1['leaders']))
                self.bridges.add(bridge_node)
                self.graph.nodes[bridge_node]['is_bridge'] = 1
                
                # Connect to a leader in ring2
                if ring2['leaders']:
                    target_leader = np.random.choice(list(ring2['leaders']))
                    self.graph.add_edge(
                        bridge_node, target_leader,
                        weight=np.random.uniform(3, 8),
                        edge_type='inter_ring'
                    )
        
        print(f"✓ Created {len(self.bridges)} bridge connections")


class CentralityMetricsCalculator:
    """
    Comprehensive centrality metrics calculator implementing six complementary
    algorithms for network importance analysis with fraud leader identification.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize centrality calculator.
        
        Args:
            graph: NetworkX graph to analyze
        """
        self.graph = graph
        self.centrality_scores = {}
        self.rankings = {}
        self.evaluation_metrics = {}
        self.computation_times = {}
        
    def calculate_degree_centrality(self) -> Dict[str, float]:
        """
        Calculate degree centrality (basic connectivity measure).
        
        Returns:
            Dictionary mapping nodes to degree centrality scores
        """
        
        print("\nCalculating Degree Centrality...")
        start_time = time.time()
        
        degree_cent = nx.degree_centrality(self.graph)
        
        elapsed = time.time() - start_time
        self.centrality_scores['degree'] = degree_cent
        self.computation_times['degree'] = elapsed
        
        print(f"✓ Degree centrality computed in {elapsed:.4f}s")
        print(f"  Max score: {max(degree_cent.values()):.4f}")
        print(f"  Mean score: {np.mean(list(degree_cent.values())):.4f}")
        
        return degree_cent
    
    def calculate_betweenness_centrality(self, k: int = None) -> Dict[str, float]:
        """
        Calculate betweenness centrality (bridge/broker nodes).
        
        Args:
            k: Sample size for approximation (None for exact)
            
        Returns:
            Dictionary mapping nodes to betweenness centrality scores
        """
        
        print("\nCalculating Betweenness Centrality...")
        start_time = time.time()
        
        if k is not None:
            betweenness = nx.betweenness_centrality(
                self.graph,
                k=k,
                weight='weight',
                normalized=True
            )
        else:
            betweenness = nx.betweenness_centrality(
                self.graph,
                weight='weight',
                normalized=True
            )
        
        elapsed = time.time() - start_time
        self.centrality_scores['betweenness'] = betweenness
        self.computation_times['betweenness'] = elapsed
        
        print(f"✓ Betweenness centrality computed in {elapsed:.4f}s")
        print(f"  Max score: {max(betweenness.values()):.4f}")
        print(f"  Mean score: {np.mean(list(betweenness.values())):.4f}")
        
        return betweenness
    
    def calculate_closeness_centrality(self) -> Dict[str, float]:
        """
        Calculate closeness centrality (central positioning).
        
        Returns:
            Dictionary mapping nodes to closeness centrality scores
        """
        
        print("\nCalculating Closeness Centrality...")
        start_time = time.time()
        
        closeness = nx.closeness_centrality(
            self.graph,
            distance='weight'
        )
        
        elapsed = time.time() - start_time
        self.centrality_scores['closeness'] = closeness
        self.computation_times['closeness'] = elapsed
        
        print(f"✓ Closeness centrality computed in {elapsed:.4f}s")
        print(f"  Max score: {max(closeness.values()):.4f}")
        print(f"  Mean score: {np.mean(list(closeness.values())):.4f}")
        
        return closeness
    
    def calculate_eigenvector_centrality(self, max_iter: int = 100) -> Dict[str, float]:
        """
        Calculate eigenvector centrality (connected to important nodes).
        
        Args:
            max_iter: Maximum iterations for convergence
            
        Returns:
            Dictionary mapping nodes to eigenvector centrality scores
        """
        
        print("\nCalculating Eigenvector Centrality...")
        start_time = time.time()
        
        try:
            eigenvector = nx.eigenvector_centrality(
                self.graph,
                max_iter=max_iter,
                weight='weight'
            )
        except nx.PowerIterationFailedConvergence:
            print("  Warning: Using approximation due to convergence issues")
            eigenvector = nx.eigenvector_centrality_numpy(
                self.graph,
                weight='weight'
            )
        
        elapsed = time.time() - start_time
        self.centrality_scores['eigenvector'] = eigenvector
        self.computation_times['eigenvector'] = elapsed
        
        print(f"✓ Eigenvector centrality computed in {elapsed:.4f}s")
        print(f"  Max score: {max(eigenvector.values()):.4f}")
        print(f"  Mean score: {np.mean(list(eigenvector.values())):.4f}")
        
        return eigenvector
    
    def calculate_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """
        Calculate PageRank centrality (web-style importance).
        
        Args:
            alpha: Damping parameter
            
        Returns:
            Dictionary mapping nodes to PageRank scores
        """
        
        print("\nCalculating PageRank Centrality...")
        start_time = time.time()
        
        pagerank = nx.pagerank(
            self.graph,
            alpha=alpha,
            weight='weight'
        )
        
        elapsed = time.time() - start_time
        self.centrality_scores['pagerank'] = pagerank
        self.computation_times['pagerank'] = elapsed
        
        print(f"✓ PageRank computed in {elapsed:.4f}s")
        print(f"  Max score: {max(pagerank.values()):.4f}")
        print(f"  Mean score: {np.mean(list(pagerank.values())):.4f}")
        
        return pagerank
    
    def calculate_katz_centrality(self, alpha: float = 0.005) -> Dict[str, float]:
        """
        Calculate Katz centrality (walk-based importance).
        
        Args:
            alpha: Attenuation factor
            
        Returns:
            Dictionary mapping nodes to Katz centrality scores
        """
        
        print("\nCalculating Katz Centrality...")
        start_time = time.time()
        
        try:
            katz = nx.katz_centrality(
                self.graph,
                alpha=alpha,
                weight='weight',
                max_iter=1000
            )
        except nx.PowerIterationFailedConvergence:
            print("  Warning: Using approximation with lower alpha")
            katz = nx.katz_centrality_numpy(
                self.graph,
                alpha=alpha * 0.5,
                weight='weight'
            )
        
        elapsed = time.time() - start_time
        self.centrality_scores['katz'] = katz
        self.computation_times['katz'] = elapsed
        
        print(f"✓ Katz centrality computed in {elapsed:.4f}s")
        print(f"  Max score: {max(katz.values()):.4f}")
        print(f"  Mean score: {np.mean(list(katz.values())):.4f}")
        
        return katz
    
    def calculate_all_metrics(self):
        """Calculate all centrality metrics."""
        
        print("\n" + "=" * 80)
        print("CALCULATING ALL CENTRALITY METRICS")
        print("=" * 80)
        
        self.calculate_degree_centrality()
        self.calculate_betweenness_centrality(k=100)  # Approximation for speed
        self.calculate_closeness_centrality()
        self.calculate_eigenvector_centrality()
        self.calculate_pagerank()
        self.calculate_katz_centrality()
        
        print(f"\n✓ All centrality metrics calculated")
        print(f"  Total computation time: {sum(self.computation_times.values()):.2f}s")
        print("=" * 80)
    
    def create_composite_score(self, weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Create composite centrality score from multiple metrics.
        
        Args:
            weights: Dictionary of metric weights (default: equal weights)
            
        Returns:
            Dictionary mapping nodes to composite scores
        """
        
        if weights is None:
            # Equal weights
            weights = {metric: 1.0/len(self.centrality_scores) 
                      for metric in self.centrality_scores.keys()}
        
        # Normalize each metric to 0-1
        normalized_scores = {}
        for metric, scores in self.centrality_scores.items():
            values = np.array(list(scores.values()))
            min_val, max_val = values.min(), values.max()
            
            if max_val > min_val:
                normalized = {node: (score - min_val) / (max_val - min_val)
                            for node, score in scores.items()}
            else:
                normalized = {node: 0.5 for node in scores.keys()}
            
            normalized_scores[metric] = normalized
        
        # Compute weighted sum
        composite = {}
        all_nodes = list(self.graph.nodes())
        
        for node in all_nodes:
            score = sum(
                normalized_scores[metric].get(node, 0) * weights.get(metric, 0)
                for metric in self.centrality_scores.keys()
            )
            composite[node] = score
        
        self.centrality_scores['composite'] = composite
        
        print(f"\n✓ Composite score created with weights: {weights}")
        
        return composite
    
    def identify_top_influencers(self, metric: str = 'composite',
                                top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Identify top influencers by centrality metric.
        
        Args:
            metric: Which centrality metric to use
            top_k: Number of top nodes to return
            
        Returns:
            List of (node, score) tuples sorted by score
        """
        
        scores = self.centrality_scores.get(metric)
        if scores is None:
            print(f"Error: Metric '{metric}' not calculated")
            return []
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_influencers = ranked[:top_k]
        
        self.rankings[metric] = ranked
        
        return top_influencers
    
    def evaluate_leader_detection(self, ground_truth_leaders: Set,
                                  metric: str = 'composite',
                                  top_k: int = 20) -> Dict:
        """
        Evaluate how well centrality metrics identify ground truth leaders.
        
        Args:
            ground_truth_leaders: Set of actual leader nodes
            metric: Which centrality metric to evaluate
            top_k: Number of top nodes to consider
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        print(f"\n{'='*80}")
        print(f"EVALUATING LEADER DETECTION: {metric.upper()}")
        print(f"{'='*80}")
        
        top_nodes = self.identify_top_influencers(metric, top_k)
        predicted_leaders = set([node for node, score in top_nodes])
        
        # Calculate metrics
        all_nodes = set(self.graph.nodes())
        
        # Binary classification: leader vs non-leader
        y_true = np.array([1 if node in ground_truth_leaders else 0 
                          for node in all_nodes])
        y_pred = np.array([1 if node in predicted_leaders else 0 
                          for node in all_nodes])
        
        # Get scores for ROC AUC
        scores = self.centrality_scores[metric]
        y_scores = np.array([scores.get(node, 0) for node in all_nodes])
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Ranking metrics
        # NDCG - Normalized Discounted Cumulative Gain
        y_true_relevance = np.array([1 if node in ground_truth_leaders else 0 
                                     for node in all_nodes]).reshape(1, -1)
        y_scores_ranking = y_scores.reshape(1, -1)
        
        try:
            ndcg = ndcg_score(y_true_relevance, y_scores_ranking)
        except:
            ndcg = 0.0
        
        # Precision@K
        true_positives_at_k = len(predicted_leaders & ground_truth_leaders)
        precision_at_k = true_positives_at_k / top_k if top_k > 0 else 0
        
        # Recall@K
        recall_at_k = true_positives_at_k / len(ground_truth_leaders) if len(ground_truth_leaders) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'ndcg': ndcg,
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'top_k': top_k,
            'true_positives': true_positives_at_k,
            'ground_truth_size': len(ground_truth_leaders)
        }
        
        self.evaluation_metrics[metric] = metrics
        
        # Print results
        print(f"\nClassification Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        print(f"\nRanking Metrics:")
        print(f"  NDCG: {ndcg:.4f}")
        print(f"  Precision@{top_k}: {precision_at_k:.4f}")
        print(f"  Recall@{top_k}: {recall_at_k:.4f}")
        
        print(f"\nDetection Results:")
        print(f"  Ground truth leaders: {len(ground_truth_leaders)}")
        print(f"  Predicted leaders (top-{top_k}): {len(predicted_leaders)}")
        print(f"  Correctly identified: {true_positives_at_k}")
        
        print(f"{'='*80}")
        
        return metrics
    
    def visualize_network_centrality(self, metric: str = 'composite',
                                    top_k: int = 20,
                                    ground_truth_leaders: Set = None,
                                    figsize: Tuple[int, int] = (20, 16)):
        """
        Visualize network with centrality-based node sizing and coloring.
        
        Args:
            metric: Which centrality metric to visualize
            top_k: Number of top nodes to highlight
            ground_truth_leaders: Ground truth leaders for comparison
            figsize: Figure size
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Get centrality scores
        scores = self.centrality_scores.get(metric, {})
        if not scores:
            print(f"No scores for metric '{metric}'")
            return
        
        # Identify top influencers
        top_nodes = self.identify_top_influencers(metric, top_k)
        top_node_set = set([node for node, score in top_nodes])
        
        # 1. Network visualization
        ax1 = axes[0, 0]
        
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)
        
        # Node sizes based on centrality
        node_sizes = [scores.get(node, 0) * 1000 + 30 for node in self.graph.nodes()]
        
        # Node colors based on type and importance
        node_colors = []
        for node in self.graph.nodes():
            if node in top_node_set:
                if ground_truth_leaders and node in ground_truth_leaders:
                    node_colors.append('darkgreen')  # Correctly identified leader
                else:
                    node_colors.append('orange')  # Predicted but not actual leader
            elif ground_truth_leaders and node in ground_truth_leaders:
                node_colors.append('red')  # Missed leader
            else:
                node_colors.append('lightblue')  # Regular node
        
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax1
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.2,
            width=0.5,
            ax=ax1
        )
        
        ax1.set_title(f'Network with {metric.capitalize()} Centrality', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='Correctly Identified Leader'),
            Patch(facecolor='orange', label='Predicted Leader (FP)'),
            Patch(facecolor='red', label='Missed Leader (FN)'),
            Patch(facecolor='lightblue', label='Regular Node')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        # 2. Centrality score distribution
        ax2 = axes[0, 1]
        
        score_values = list(scores.values())
        ax2.hist(score_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(np.percentile(score_values, 95), color='red', linestyle='--', 
                   linewidth=2, label='95th Percentile')
        ax2.set_title(f'{metric.capitalize()} Score Distribution', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Centrality Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Top influencers ranking
        ax3 = axes[1, 0]
        
        top_15 = top_nodes[:15]
        nodes_list = [node for node, score in top_15]
        scores_list = [score for node, score in top_15]
        
        # Color by whether they're actual leaders
        colors_bar = []
        for node in nodes_list:
            if ground_truth_leaders and node in ground_truth_leaders:
                colors_bar.append('darkgreen')
            else:
                colors_bar.append('coral')
        
        bars = ax3.barh(range(len(nodes_list)), scores_list, color=colors_bar, 
                       alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(nodes_list)))
        ax3.set_yticklabels(nodes_list, fontsize=9)
        ax3.set_xlabel('Centrality Score')
        ax3.set_title(f'Top 15 Nodes by {metric.capitalize()}', 
                     fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Metrics comparison
        ax4 = axes[1, 1]
        
        if self.evaluation_metrics:
            metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc', 'ndcg', 'precision_at_k']
            metric_labels = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'NDCG', f'Precision@{top_k}']
            
            # Gather data
            all_metrics_data = []
            all_metric_names = []
            
            for metric_name in self.evaluation_metrics.keys():
                metric_data = self.evaluation_metrics[metric_name]
                values = [metric_data.get(m, 0) for m in metrics_to_plot]
                all_metrics_data.append(values)
                all_metric_names.append(metric_name)
            
            if all_metrics_data:
                # Create heatmap
                sns.heatmap(
                    np.array(all_metrics_data).T,
                    xticklabels=all_metric_names,
                    yticklabels=metric_labels,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=1,
                    ax=ax4,
                    cbar_kws={'label': 'Score'}
                )
                ax4.set_title('Centrality Metrics Performance Comparison', 
                            fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Centrality Analysis - {metric.capitalize()} Metric', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(f'centrality_analysis_{metric}.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as 'centrality_analysis_{metric}.png'")
        plt.show()
    
    def generate_comprehensive_report(self, ground_truth_leaders: Set,
                                     top_k: int = 20) -> pd.DataFrame:
        """
        Generate comprehensive comparison report of all metrics.
        
        Args:
            ground_truth_leaders: Ground truth leader nodes
            top_k: Number of top nodes to evaluate
            
        Returns:
            DataFrame with comparative analysis
        """
        
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE CENTRALITY ANALYSIS REPORT")
        print(f"{'='*80}")
        
        # Evaluate all metrics
        for metric in self.centrality_scores.keys():
            if metric not in self.evaluation_metrics:
                self.evaluate_leader_detection(ground_truth_leaders, metric, top_k)
        
        # Create comparison dataframe
        comparison_data = []
        
        for metric, metrics_dict in self.evaluation_metrics.items():
            row = {
                'Metric': metric.capitalize(),
                'Precision': metrics_dict['precision'],
                'Recall': metrics_dict['recall'],
                'F1-Score': metrics_dict['f1_score'],
                'ROC-AUC': metrics_dict['roc_auc'],
                'NDCG': metrics_dict['ndcg'],
                f'Precision@{top_k}': metrics_dict['precision_at_k'],
                f'Recall@{top_k}': metrics_dict['recall_at_k'],
                'Computation Time (s)': self.computation_times.get(metric, 0)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        print("\n" + df.to_string(index=False))
        print(f"\n{'='*80}")
        
        return df
    
    def save_results(self, filepath: str = 'centrality_results.pkl'):
        """Save all centrality analysis results."""
        
        results = {
            'centrality_scores': self.centrality_scores,
            'rankings': self.rankings,
            'evaluation_metrics': self.evaluation_metrics,
            'computation_times': self.computation_times
        }
        
        joblib.dump(results, filepath)
        print(f"\n✓ Results saved to {filepath}")
    
    def load_results(self, filepath: str = 'centrality_results.pkl'):
        """Load previously saved results."""
        
        results = joblib.load(filepath)
        
        self.centrality_scores = results['centrality_scores']
        self.rankings = results['rankings']
        self.evaluation_metrics = results['evaluation_metrics']
        self.computation_times = results['computation_times']
        
        print(f"✓ Results loaded from {filepath}")


def main():
    """
    Execute complete centrality analysis pipeline with evaluation.
    """
    
    print("\n" + "╔" + "═"*88 + "╗")
    print("║" + " "*20 + "GRAINSECURE PDS MONITORING SYSTEM" + " "*35 + "║")
    print("║" + " "*12 + "Centrality Metrics Calculator for Network Importance Analysis" + " "*13 + "║")
    print("║" + " "*10 + "Advanced Node Ranking and Fraud Leader Identification System" + " "*16 + "║")
    print("╚" + "═"*88 + "╝")
    print()
    
    # Step 1: Generate Hierarchical Network
    print("STEP 1: Generating Hierarchical Fraud Network with Leaders")
    print("="*80)
    
    generator = HierarchicalFraudNetworkGenerator(
        n_beneficiaries=2000,
        n_shops=150,
        n_fraud_rings=10,
        leader_ratio=0.15
    )
    
    graph, leaders, coordinators, bridges = generator.generate_hierarchical_network()
    
    # Combine all important nodes for evaluation
    ground_truth_important = leaders | coordinators | bridges
    
    print(f"\n  Ground truth important nodes: {len(ground_truth_important)}")
    print()
    
    # Step 2: Initialize Calculator
    print("\nSTEP 2: Initializing Centrality Metrics Calculator")
    print("="*80)
    
    calculator = CentralityMetricsCalculator(graph)
    print(f"✓ Calculator initialized with {graph.number_of_nodes():,} nodes")
    print()
    
    # Step 3: Calculate All Metrics
    print("\nSTEP 3: Computing All Centrality Metrics")
    
    calculator.calculate_all_metrics()
    print()
    
    # Step 4: Create Composite Score
    print("\nSTEP 4: Creating Composite Centrality Score")
    print("="*80)
    
    # Weight metrics based on their relevance for fraud detection
    weights = {
        'degree': 0.15,
        'betweenness': 0.25,  # Bridge nodes important
        'closeness': 0.15,
        'eigenvector': 0.20,  # Connected to important nodes
        'pagerank': 0.15,
        'katz': 0.10
    }
    
    calculator.create_composite_score(weights)
    print()
    
    # Step 5: Evaluate All Metrics
    print("\nSTEP 5: Evaluating Leader Detection Performance")
    
    top_k = 30  # Evaluate top 30 nodes
    
    print(f"\nEvaluating with top-{top_k} nodes...")
    print(f"Ground truth leaders: {len(leaders)}")
    print(f"Ground truth important nodes (leaders + coordinators + bridges): {len(ground_truth_important)}")
    
    # Evaluate against leaders specifically
    for metric in calculator.centrality_scores.keys():
        calculator.evaluate_leader_detection(leaders, metric, top_k)
    
    print()
    
    # Step 6: Generate Comprehensive Report
    print("\nSTEP 6: Generating Comprehensive Performance Report")
    
    report_df = calculator.generate_comprehensive_report(leaders, top_k)
    
    # Find best metric
    best_metric = report_df.iloc[0]['Metric'].lower()
    best_f1 = report_df.iloc[0]['F1-Score']
    best_precision = report_df.iloc[0]['Precision']
    best_recall = report_df.iloc[0]['Recall']
    
    print()
    
    # Step 7: Check Performance Targets
    print("\nSTEP 7: Performance Target Validation")
    print("="*80)
    
    targets_met = (
        best_f1 >= 0.95 and
        best_precision >= 0.88 and
        best_recall >= 0.85
    )
    
    print(f"\nBest Performing Metric: {best_metric.upper()}")
    print(f"  F1-Score:  {best_f1:.4f} ({'✓ EXCEEDS' if best_f1 >= 0.95 else '✗ BELOW'} 0.95 target)")
    print(f"  Precision: {best_precision:.4f} ({'✓ EXCEEDS' if best_precision >= 0.88 else '✗ BELOW'} 0.88 target)")
    print(f"  Recall:    {best_recall:.4f} ({'✓ EXCEEDS' if best_recall >= 0.85 else '✗ BELOW'} 0.85 target)")
    
    if targets_met:
        print("\n╔" + "═"*88 + "╗")
        print("║" + " "*15 + "🎯 ALL PERFORMANCE TARGETS ACHIEVED! 🎯" + " "*32 + "║")
        print("║" + " "*18 + "Ready for Production Deployment" + " "*37 + "║")
        print("╚" + "═"*88 + "╝")
    
    print()
    
    # Step 8: Visualizations
    print("\nSTEP 8: Generating Comprehensive Visualizations")
    print("="*80)
    
    calculator.visualize_network_centrality(
        metric='composite',
        top_k=top_k,
        ground_truth_leaders=leaders
    )
    print()
    
    # Step 9: Save Results
    print("\nSTEP 9: Saving Analysis Results")
    print("="*80)
    
    calculator.save_results('grainsecure_centrality_analysis.pkl')
    
    # Save detailed report
    report_df.to_csv('centrality_metrics_comparison.csv', index=False)
    print("✓ Comparison report saved to 'centrality_metrics_comparison.csv'")
    
    # Save top influencers
    top_influencers = calculator.identify_top_influencers('composite', 50)
    influencers_df = pd.DataFrame(top_influencers, columns=['Node', 'Centrality_Score'])
    influencers_df['Is_Actual_Leader'] = influencers_df['Node'].apply(lambda x: x in leaders)
    influencers_df['Is_Coordinator'] = influencers_df['Node'].apply(lambda x: x in coordinators)
    influencers_df['Is_Bridge'] = influencers_df['Node'].apply(lambda x: x in bridges)
    influencers_df.to_csv('top_influencers.csv', index=False)
    print("✓ Top influencers saved to 'top_influencers.csv'")
    print()
    
    print("="*80)
    print("CENTRALITY METRICS ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print()
    print("Deliverables:")
    print("  1. grainsecure_centrality_analysis.pkl - Complete analysis results")
    print("  2. centrality_metrics_comparison.csv - Performance comparison table")
    print("  3. top_influencers.csv - Ranked list of influential nodes")
    print("  4. centrality_analysis_composite.png - Network visualization")
    print()
    print("Key Findings:")
    print(f"  • Best metric: {best_metric} (F1={best_f1:.4f})")
    print(f"  • Leaders correctly identified: {calculator.evaluation_metrics['composite']['true_positives']}/{len(leaders)}")
    print(f"  • Precision@{top_k}: {calculator.evaluation_metrics['composite']['precision_at_k']:.4f}")
    print(f"  • Total computation time: {sum(calculator.computation_times.values()):.2f}s")
    print()
    print("Next Steps:")
    print("  • Deploy centrality ranking to production systems")
    print("  • Integrate with GNN and community detection for comprehensive analysis")
    print("  • Implement temporal centrality tracking for leadership changes")
    print("  • Begin temporal pattern analyzer development")
    print()


if __name__ == "__main__":
    main()
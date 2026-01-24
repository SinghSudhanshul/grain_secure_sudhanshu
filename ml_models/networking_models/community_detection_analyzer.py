"""
GrainSecure PDS Monitoring System
Community Detection Analyzer for Fraud Ring Identification

This module implements advanced graph clustering algorithms to identify
fraud ring communities, analyze network structure, and detect coordinated
collusion patterns in PDS transaction networks.

Author: GrainSecure Development Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime, timedelta
import warnings
import joblib
import json
from typing import List, Set, Dict, Tuple, Optional
import time

warnings.filterwarnings('ignore')
np.random.seed(42)


class FraudNetworkGenerator:
    """
    Generates realistic fraud network structures with ground truth communities
    for community detection algorithm validation and evaluation.
    """
    
    def __init__(self, n_beneficiaries: int = 3000, n_shops: int = 200,
                 n_fraud_rings: int = 12, ring_size_range: Tuple[int, int] = (5, 30)):
        """
        Initialize fraud network generator.
        
        Args:
            n_beneficiaries: Total number of beneficiary nodes
            n_shops: Total number of shop nodes
            n_fraud_rings: Number of distinct fraud rings to create
            ring_size_range: Min and max size for each fraud ring
        """
        self.n_beneficiaries = n_beneficiaries
        self.n_shops = n_shops
        self.n_fraud_rings = n_fraud_rings
        self.ring_size_range = ring_size_range
        
        self.graph = None
        self.ground_truth_communities = []
        self.fraud_ring_metadata = []
        
    def generate_fraud_network(self) -> Tuple[nx.Graph, List[Set], List[Dict]]:
        """
        Generate complete fraud network with ground truth communities.
        
        Returns:
            Tuple of (NetworkX graph, ground truth communities, metadata)
        """
        
        print("=" * 80)
        print("GENERATING FRAUD NETWORK WITH GROUND TRUTH COMMUNITIES")
        print("=" * 80)
        
        self.graph = nx.Graph()
        
        # Add all nodes
        self._add_nodes()
        
        # Generate fraud ring communities
        self._generate_fraud_rings()
        
        # Add legitimate transactions (background noise)
        self._add_legitimate_connections()
        
        # Add inter-community connections (some fraud rings cooperate)
        self._add_intercommunity_connections()
        
        print(f"\n✓ Network generation complete")
        print(f"  Total nodes: {self.graph.number_of_nodes():,}")
        print(f"  Total edges: {self.graph.number_of_edges():,}")
        print(f"  Fraud rings: {len(self.ground_truth_communities)}")
        print(f"  Average ring size: {np.mean([len(ring) for ring in self.ground_truth_communities]):.1f}")
        print(f"  Nodes in fraud rings: {sum(len(ring) for ring in self.ground_truth_communities):,}")
        print("=" * 80)
        
        return self.graph, self.ground_truth_communities, self.fraud_ring_metadata
    
    def _add_nodes(self):
        """Add beneficiary and shop nodes with attributes."""
        
        # Beneficiaries
        for i in range(self.n_beneficiaries):
            self.graph.add_node(
                f'B_{i}',
                node_type='beneficiary',
                family_size=np.random.randint(1, 11),
                risk_score=np.random.random(),
                is_fraudulent=0
            )
        
        # Shops
        for i in range(self.n_shops):
            self.graph.add_node(
                f'S_{i}',
                node_type='shop',
                compliance_score=np.random.uniform(40, 100),
                is_complicit=0
            )
        
        print(f"✓ Added {self.n_beneficiaries:,} beneficiary nodes")
        print(f"✓ Added {self.n_shops:,} shop nodes")
    
    def _generate_fraud_rings(self):
        """Generate distinct fraud ring communities."""
        
        available_beneficiaries = set([f'B_{i}' for i in range(self.n_beneficiaries)])
        
        for ring_id in range(self.n_fraud_rings):
            # Determine ring size
            ring_size = np.random.randint(self.ring_size_range[0], self.ring_size_range[1])
            
            if len(available_beneficiaries) < ring_size:
                break
            
            # Select beneficiaries for this ring
            ring_beneficiaries = set(np.random.choice(
                list(available_beneficiaries),
                size=ring_size,
                replace=False
            ))
            
            # Select 1-3 complicit shops
            n_shops_in_ring = np.random.randint(1, 4)
            ring_shops = set([f'S_{i}' for i in np.random.choice(
                self.n_shops,
                size=n_shops_in_ring,
                replace=False
            )])
            
            # Complete community includes beneficiaries and shops
            complete_ring = ring_beneficiaries | ring_shops
            self.ground_truth_communities.append(complete_ring)
            
            # Mark nodes as fraudulent/complicit
            for node in ring_beneficiaries:
                self.graph.nodes[node]['is_fraudulent'] = 1
                self.graph.nodes[node]['fraud_ring_id'] = ring_id
            
            for node in ring_shops:
                self.graph.nodes[node]['is_complicit'] = 1
                self.graph.nodes[node]['fraud_ring_id'] = ring_id
            
            # Create dense internal connections
            # Beneficiary-to-shop connections (transaction edges)
            for beneficiary in ring_beneficiaries:
                for shop in ring_shops:
                    # High probability of connection within ring
                    if np.random.random() < 0.85:
                        self.graph.add_edge(
                            beneficiary,
                            shop,
                            weight=np.random.uniform(5, 20),  # High transaction frequency
                            edge_type='transaction',
                            is_fraudulent=1
                        )
            
            # Beneficiary-to-beneficiary connections (coordination edges)
            beneficiary_list = list(ring_beneficiaries)
            for i, b1 in enumerate(beneficiary_list):
                for b2 in beneficiary_list[i+1:]:
                    # Some coordination connections
                    if np.random.random() < 0.3:
                        self.graph.add_edge(
                            b1,
                            b2,
                            weight=np.random.uniform(1, 5),
                            edge_type='coordination',
                            is_fraudulent=1
                        )
            
            # Store metadata
            self.fraud_ring_metadata.append({
                'ring_id': ring_id,
                'size': len(complete_ring),
                'beneficiaries': len(ring_beneficiaries),
                'shops': len(ring_shops),
                'internal_edges': self.graph.subgraph(complete_ring).number_of_edges(),
                'cohesion': self._calculate_cohesion(complete_ring)
            })
            
            available_beneficiaries -= ring_beneficiaries
        
        print(f"✓ Generated {len(self.ground_truth_communities)} fraud rings")
    
    def _calculate_cohesion(self, community: Set) -> float:
        """Calculate internal cohesion of a community."""
        subgraph = self.graph.subgraph(community)
        n_nodes = len(community)
        n_edges = subgraph.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) / 2
        return n_edges / max_edges if max_edges > 0 else 0
    
    def _add_legitimate_connections(self):
        """Add legitimate transaction connections as background noise."""
        
        beneficiaries = [n for n in self.graph.nodes() if n.startswith('B_')]
        shops = [n for n in self.graph.nodes() if n.startswith('S_')]
        
        # Each non-fraudulent beneficiary transacts with 1-3 shops
        for beneficiary in beneficiaries:
            if self.graph.nodes[beneficiary]['is_fraudulent'] == 0:
                n_shops = np.random.randint(1, 4)
                selected_shops = np.random.choice(shops, size=n_shops, replace=False)
                
                for shop in selected_shops:
                    self.graph.add_edge(
                        beneficiary,
                        shop,
                        weight=np.random.uniform(1, 5),
                        edge_type='transaction',
                        is_fraudulent=0
                    )
        
        print(f"✓ Added legitimate transaction connections")
    
    def _add_intercommunity_connections(self):
        """Add sparse connections between some fraud rings (cooperation)."""
        
        # 20% chance of connection between any two rings
        for i, ring1 in enumerate(self.ground_truth_communities):
            for ring2 in self.ground_truth_communities[i+1:]:
                if np.random.random() < 0.2:
                    # Connect 1-2 members from each ring
                    members1 = list(ring1)
                    members2 = list(ring2)
                    
                    n_connections = np.random.randint(1, 3)
                    for _ in range(n_connections):
                        m1 = np.random.choice(members1)
                        m2 = np.random.choice(members2)
                        
                        if not self.graph.has_edge(m1, m2):
                            self.graph.add_edge(
                                m1,
                                m2,
                                weight=np.random.uniform(0.5, 2),
                                edge_type='inter_ring',
                                is_fraudulent=1
                            )
        
        print(f"✓ Added inter-community cooperation connections")


class CommunityDetectionAnalyzer:
    """
    Advanced community detection analyzer implementing multiple algorithms
    for fraud ring identification with comprehensive evaluation metrics.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize community detection analyzer.
        
        Args:
            graph: NetworkX graph to analyze
        """
        self.graph = graph
        self.detected_communities = {}
        self.evaluation_metrics = {}
        self.algorithm_performance = {}
        
    def detect_communities_louvain(self) -> List[Set]:
        """
        Detect communities using Louvain modularity optimization.
        
        Returns:
            List of detected communities (node sets)
        """
        
        print("\nRunning Louvain Community Detection...")
        start_time = time.time()
        
        # Louvain algorithm
        communities_dict = community.greedy_modularity_communities(
            self.graph,
            weight='weight',
            resolution=1.0
        )
        
        communities = [set(comm) for comm in communities_dict]
        
        elapsed_time = time.time() - start_time
        modularity = community.modularity(self.graph, communities_dict, weight='weight')
        
        print(f"✓ Louvain detection complete in {elapsed_time:.2f}s")
        print(f"  Communities detected: {len(communities)}")
        print(f"  Modularity: {modularity:.4f}")
        print(f"  Average community size: {np.mean([len(c) for c in communities]):.1f}")
        
        self.detected_communities['louvain'] = communities
        self.algorithm_performance['louvain'] = {
            'n_communities': len(communities),
            'modularity': modularity,
            'time': elapsed_time
        }
        
        return communities
    
    def detect_communities_label_propagation(self) -> List[Set]:
        """
        Detect communities using label propagation algorithm.
        
        Returns:
            List of detected communities
        """
        
        print("\nRunning Label Propagation Community Detection...")
        start_time = time.time()
        
        communities_gen = community.label_propagation_communities(self.graph)
        communities = [set(comm) for comm in communities_gen]
        
        elapsed_time = time.time() - start_time
        modularity = community.modularity(self.graph, communities, weight='weight')
        
        print(f"✓ Label Propagation complete in {elapsed_time:.2f}s")
        print(f"  Communities detected: {len(communities)}")
        print(f"  Modularity: {modularity:.4f}")
        
        self.detected_communities['label_propagation'] = communities
        self.algorithm_performance['label_propagation'] = {
            'n_communities': len(communities),
            'modularity': modularity,
            'time': elapsed_time
        }
        
        return communities
    
    def detect_communities_girvan_newman(self, k: int = None) -> List[Set]:
        """
        Detect communities using Girvan-Newman edge betweenness.
        
        Args:
            k: Number of communities to detect (auto if None)
            
        Returns:
            List of detected communities
        """
        
        print("\nRunning Girvan-Newman Community Detection...")
        start_time = time.time()
        
        # This can be slow for large graphs, so limit iterations
        comp = community.girvan_newman(self.graph)
        
        if k is None:
            # Auto-detect optimal number of communities based on modularity
            best_communities = None
            best_modularity = -1
            
            for communities in comp:
                comm_list = [set(c) for c in communities]
                mod = community.modularity(self.graph, comm_list, weight='weight')
                
                if mod > best_modularity:
                    best_modularity = mod
                    best_communities = comm_list
                
                # Stop if we have reasonable number of communities
                if len(comm_list) > 20:
                    break
            
            communities = best_communities
        else:
            # Get specific k communities
            for _ in range(k - 1):
                communities = next(comp)
            communities = [set(c) for c in communities]
        
        elapsed_time = time.time() - start_time
        modularity = community.modularity(self.graph, communities, weight='weight')
        
        print(f"✓ Girvan-Newman complete in {elapsed_time:.2f}s")
        print(f"  Communities detected: {len(communities)}")
        print(f"  Modularity: {modularity:.4f}")
        
        self.detected_communities['girvan_newman'] = communities
        self.algorithm_performance['girvan_newman'] = {
            'n_communities': len(communities),
            'modularity': modularity,
            'time': elapsed_time
        }
        
        return communities
    
    def detect_communities_spectral(self, n_communities: int = None) -> List[Set]:
        """
        Detect communities using spectral clustering on graph Laplacian.
        
        Args:
            n_communities: Number of communities (estimated if None)
            
        Returns:
            List of detected communities
        """
        
        print("\nRunning Spectral Clustering Community Detection...")
        start_time = time.time()
        
        # Get adjacency matrix
        nodes = list(self.graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=nodes, weight='weight')
        
        # Estimate number of communities if not provided
        if n_communities is None:
            # Use eigengap heuristic
            laplacian = nx.laplacian_matrix(self.graph).todense()
            eigenvalues = np.linalg.eigvalsh(laplacian)
            eigengaps = np.diff(eigenvalues)
            n_communities = np.argmax(eigengaps[:20]) + 2  # +2 for index adjustment
        
        # Spectral clustering
        from sklearn.cluster import SpectralClustering
        
        sc = SpectralClustering(
            n_clusters=n_communities,
            affinity='precomputed',
            random_state=42
        )
        
        labels = sc.fit_predict(adj_matrix)
        
        # Convert to community sets
        communities = [set() for _ in range(n_communities)]
        for node, label in zip(nodes, labels):
            communities[label].add(node)
        
        # Remove empty communities
        communities = [c for c in communities if len(c) > 0]
        
        elapsed_time = time.time() - start_time
        modularity = community.modularity(self.graph, communities, weight='weight')
        
        print(f"✓ Spectral Clustering complete in {elapsed_time:.2f}s")
        print(f"  Communities detected: {len(communities)}")
        print(f"  Modularity: {modularity:.4f}")
        
        self.detected_communities['spectral'] = communities
        self.algorithm_performance['spectral'] = {
            'n_communities': len(communities),
            'modularity': modularity,
            'time': elapsed_time
        }
        
        return communities
    
    def evaluate_against_ground_truth(self, ground_truth: List[Set],
                                      algorithm: str = 'louvain') -> Dict:
        """
        Evaluate detected communities against ground truth.
        
        Args:
            ground_truth: Ground truth communities
            algorithm: Which detection algorithm to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        print(f"\n{'='*80}")
        print(f"EVALUATING {algorithm.upper()} AGAINST GROUND TRUTH")
        print(f"{'='*80}")
        
        detected = self.detected_communities.get(algorithm)
        if detected is None:
            print(f"Error: No detected communities for algorithm '{algorithm}'")
            return {}
        
        # Convert to label format for sklearn metrics
        all_nodes = set()
        for comm in ground_truth:
            all_nodes.update(comm)
        for comm in detected:
            all_nodes.update(comm)
        
        all_nodes = sorted(list(all_nodes))
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create label arrays
        true_labels = np.zeros(len(all_nodes), dtype=int)
        pred_labels = np.zeros(len(all_nodes), dtype=int)
        
        for comm_id, comm in enumerate(ground_truth):
            for node in comm:
                if node in node_to_idx:
                    true_labels[node_to_idx[node]] = comm_id
        
        for comm_id, comm in enumerate(detected):
            for node in comm:
                if node in node_to_idx:
                    pred_labels[node_to_idx[node]] = comm_id
        
        # Calculate metrics
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        fmi = fowlkes_mallows_score(true_labels, pred_labels)
        
        # Calculate precision, recall, F1 for fraud ring detection
        fraud_detection_metrics = self._calculate_fraud_detection_metrics(
            ground_truth, detected
        )
        
        # Community quality metrics
        quality_metrics = self._calculate_community_quality(detected)
        
        metrics = {
            'normalized_mutual_info': nmi,
            'adjusted_rand_index': ari,
            'fowlkes_mallows_index': fmi,
            **fraud_detection_metrics,
            **quality_metrics,
            'n_ground_truth_communities': len(ground_truth),
            'n_detected_communities': len(detected)
        }
        
        self.evaluation_metrics[algorithm] = metrics
        
        # Print results
        print(f"\nClustering Agreement Metrics:")
        print(f"  Normalized Mutual Information: {nmi:.4f}")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Fowlkes-Mallows Index: {fmi:.4f}")
        
        print(f"\nFraud Ring Detection Performance:")
        print(f"  Precision: {fraud_detection_metrics['precision']:.4f}")
        print(f"  Recall: {fraud_detection_metrics['recall']:.4f}")
        print(f"  F1-Score: {fraud_detection_metrics['f1_score']:.4f}")
        print(f"  Accuracy: {fraud_detection_metrics['accuracy']:.4f}")
        
        print(f"\nCommunity Quality Metrics:")
        print(f"  Average Modularity: {quality_metrics['avg_modularity']:.4f}")
        print(f"  Average Conductance: {quality_metrics['avg_conductance']:.4f}")
        print(f"  Coverage: {quality_metrics['coverage']:.4f}")
        
        print(f"{'='*80}")
        
        return metrics
    
    def _calculate_fraud_detection_metrics(self, ground_truth: List[Set],
                                           detected: List[Set]) -> Dict:
        """Calculate precision, recall, F1 for fraud ring detection."""
        
        # For each detected community, find best matching ground truth
        matches = []
        
        for det_comm in detected:
            best_overlap = 0
            best_gt = None
            
            for gt_comm in ground_truth:
                overlap = len(det_comm & gt_comm)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gt = gt_comm
            
            if best_gt is not None:
                precision = best_overlap / len(det_comm) if len(det_comm) > 0 else 0
                recall = best_overlap / len(best_gt) if len(best_gt) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                matches.append({
                    'overlap': best_overlap,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        if len(matches) == 0:
            return {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0}
        
        # Aggregate metrics
        avg_precision = np.mean([m['precision'] for m in matches])
        avg_recall = np.mean([m['recall'] for m in matches])
        avg_f1 = np.mean([m['f1'] for m in matches])
        
        # Overall accuracy: proportion of correctly clustered nodes
        all_nodes = set()
        for comm in ground_truth:
            all_nodes.update(comm)
        
        correctly_clustered = sum(m['overlap'] for m in matches)
        accuracy = correctly_clustered / len(all_nodes) if len(all_nodes) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'accuracy': accuracy
        }
    
    def _calculate_community_quality(self, communities: List[Set]) -> Dict:
        """Calculate intrinsic quality metrics for detected communities."""
        
        modularities = []
        conductances = []
        
        for comm in communities:
            if len(comm) > 1:
                # Modularity for this community
                subgraph = self.graph.subgraph(comm)
                mod = community.modularity(
                    self.graph,
                    [comm],
                    weight='weight'
                )
                modularities.append(mod)
                
                # Conductance
                internal_edges = subgraph.number_of_edges()
                boundary_edges = 0
                for node in comm:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in comm:
                            boundary_edges += 1
                
                total_edges = internal_edges + boundary_edges
                conductance = boundary_edges / total_edges if total_edges > 0 else 0
                conductances.append(conductance)
        
        # Coverage: fraction of edges within communities
        total_edges = self.graph.number_of_edges()
        edges_in_communities = 0
        for comm in communities:
            edges_in_communities += self.graph.subgraph(comm).number_of_edges()
        
        coverage = edges_in_communities / total_edges if total_edges > 0 else 0
        
        return {
            'avg_modularity': np.mean(modularities) if modularities else 0,
            'avg_conductance': np.mean(conductances) if conductances else 0,
            'coverage': coverage
        }
    
    def identify_fraud_rings(self, algorithm: str = 'louvain',
                            min_size: int = 3) -> List[Dict]:
        """
        Identify likely fraud rings from detected communities.
        
        Args:
            algorithm: Community detection algorithm to use
            min_size: Minimum community size to consider
            
        Returns:
            List of fraud ring information dictionaries
        """
        
        print(f"\n{'='*80}")
        print(f"IDENTIFYING FRAUD RINGS FROM {algorithm.upper()} COMMUNITIES")
        print(f"{'='*80}")
        
        communities = self.detected_communities.get(algorithm)
        if communities is None:
            print(f"Error: No communities detected with algorithm '{algorithm}'")
            return []
        
        fraud_rings = []
        
        for comm_id, comm in enumerate(communities):
            if len(comm) < min_size:
                continue
            
            # Analyze community characteristics
            subgraph = self.graph.subgraph(comm)
            
            # Count fraudulent nodes
            fraudulent_nodes = sum(
                1 for node in comm
                if self.graph.nodes[node].get('is_fraudulent', 0) == 1 or
                   self.graph.nodes[node].get('is_complicit', 0) == 1
            )
            
            fraud_percentage = fraudulent_nodes / len(comm) if len(comm) > 0 else 0
            
            # Calculate suspicion score
            internal_density = nx.density(subgraph)
            avg_degree = np.mean([subgraph.degree(node) for node in comm])
            
            # Weighted suspicion score
            suspicion_score = (
                fraud_percentage * 0.5 +
                internal_density * 0.3 +
                min(avg_degree / 10, 1.0) * 0.2
            )
            
            # Identify member types
            beneficiaries = [n for n in comm if n.startswith('B_')]
            shops = [n for n in comm if n.startswith('S_')]
            
            fraud_ring_info = {
                'community_id': comm_id,
                'size': len(comm),
                'beneficiaries': len(beneficiaries),
                'shops': len(shops),
                'fraudulent_nodes': fraudulent_nodes,
                'fraud_percentage': fraud_percentage,
                'internal_density': internal_density,
                'avg_degree': avg_degree,
                'suspicion_score': suspicion_score,
                'members': list(comm),
                'is_likely_fraud': suspicion_score > 0.6
            }
            
            fraud_rings.append(fraud_ring_info)
        
        # Sort by suspicion score
        fraud_rings.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        # Print top suspicious communities
        print(f"\nDetected {len(fraud_rings)} communities (size >= {min_size})")
        print(f"High-suspicion fraud rings (score > 0.6): {sum(1 for r in fraud_rings if r['is_likely_fraud'])}")
        
        print(f"\nTop 5 Most Suspicious Communities:")
        for i, ring in enumerate(fraud_rings[:5], 1):
            print(f"\n  {i}. Community #{ring['community_id']}")
            print(f"     Size: {ring['size']} ({ring['beneficiaries']} beneficiaries, {ring['shops']} shops)")
            print(f"     Suspicion Score: {ring['suspicion_score']:.3f}")
            print(f"     Fraud Percentage: {ring['fraud_percentage']*100:.1f}%")
            print(f"     Internal Density: {ring['internal_density']:.3f}")
        
        print(f"{'='*80}")
        
        return fraud_rings
    
    def visualize_communities(self, algorithm: str = 'louvain',
                             max_communities: int = 15,
                             figsize: Tuple[int, int] = (20, 16)):
        """
        Visualize detected communities and network structure.
        
        Args:
            algorithm: Which algorithm's communities to visualize
            max_communities: Maximum communities to display
            figsize: Figure size
        """
        
        communities = self.detected_communities.get(algorithm)
        if communities is None:
            print(f"No communities to visualize for algorithm '{algorithm}'")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Network with community colors
        ax1 = axes[0, 0]
        
        # Assign colors to communities
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(communities), max_communities)))
        node_colors = []
        node_to_comm = {}
        
        for comm_id, comm in enumerate(communities[:max_communities]):
            for node in comm:
                node_to_comm[node] = comm_id
        
        for node in self.graph.nodes():
            if node in node_to_comm:
                node_colors.append(colors[node_to_comm[node]])
            else:
                node_colors.append('lightgray')
        
        # Layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)
        
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=30,
            alpha=0.7,
            ax=ax1
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.2,
            width=0.5,
            ax=ax1
        )
        
        ax1.set_title(f'Network Communities ({algorithm})', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Community size distribution
        ax2 = axes[0, 1]
        
        sizes = [len(comm) for comm in communities]
        ax2.hist(sizes, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_title('Community Size Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Community Size')
        ax2.set_ylabel('Frequency')
        ax2.grid(alpha=0.3)
        
        # 3. Modularity comparison
        ax3 = axes[1, 0]
        
        algorithms = list(self.algorithm_performance.keys())
        modularities = [self.algorithm_performance[alg]['modularity'] for alg in algorithms]
        
        bars = ax3.bar(algorithms, modularities, color='coral', alpha=0.7, edgecolor='black')
        ax3.set_title('Algorithm Modularity Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Modularity')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, mod in zip(bars, modularities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mod:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance metrics heatmap
        ax4 = axes[1, 1]
        
        if self.evaluation_metrics:
            metric_names = ['NMI', 'ARI', 'FMI', 'Precision', 'Recall', 'F1']
            data = []
            
            for alg in algorithms:
                if alg in self.evaluation_metrics:
                    metrics = self.evaluation_metrics[alg]
                    row = [
                        metrics.get('normalized_mutual_info', 0),
                        metrics.get('adjusted_rand_index', 0),
                        metrics.get('fowlkes_mallows_index', 0),
                        metrics.get('precision', 0),
                        metrics.get('recall', 0),
                        metrics.get('f1_score', 0)
                    ]
                    data.append(row)
            
            if data:
                sns.heatmap(
                    np.array(data).T,
                    xticklabels=algorithms,
                    yticklabels=metric_names,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=1,
                    ax=ax4,
                    cbar_kws={'label': 'Score'}
                )
                ax4.set_title('Evaluation Metrics Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'community_detection_{algorithm}.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as 'community_detection_{algorithm}.png'")
        plt.show()
    
    def save_results(self, filepath: str = 'community_detection_results.pkl'):
        """Save all detection results and metrics."""
        
        results = {
            'detected_communities': self.detected_communities,
            'evaluation_metrics': self.evaluation_metrics,
            'algorithm_performance': self.algorithm_performance
        }
        
        joblib.dump(results, filepath)
        print(f"\n✓ Results saved to {filepath}")
    
    def load_results(self, filepath: str = 'community_detection_results.pkl'):
        """Load previously saved results."""
        
        results = joblib.load(filepath)
        
        self.detected_communities = results['detected_communities']
        self.evaluation_metrics = results['evaluation_metrics']
        self.algorithm_performance = results['algorithm_performance']
        
        print(f"✓ Results loaded from {filepath}")


def main():
    """
    Execute complete community detection pipeline with evaluation.
    """
    
    print("\n" + "╔" + "═"*88 + "╗")
    print("║" + " "*20 + "GRAINSECURE PDS MONITORING SYSTEM" + " "*35 + "║")
    print("║" + " "*15 + "Community Detection Analyzer for Fraud Rings" + " "*28 + "║")
    print("║" + " "*10 + "Advanced Graph Clustering and Network Structure Analysis" + " "*20 + "║")
    print("╚" + "═"*88 + "╝")
    print()
    
    # Step 1: Generate Fraud Network
    print("STEP 1: Generating Fraud Network with Ground Truth Communities")
    print("="*80)
    
    generator = FraudNetworkGenerator(
        n_beneficiaries=3000,
        n_shops=200,
        n_fraud_rings=12,
        ring_size_range=(5, 30)
    )
    
    graph, ground_truth, metadata = generator.generate_fraud_network()
    print()
    
    # Step 2: Initialize Analyzer
    print("\nSTEP 2: Initializing Community Detection Analyzer")
    print("="*80)
    
    analyzer = CommunityDetectionAnalyzer(graph)
    print(f"✓ Analyzer initialized with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
    print()
    
    # Step 3: Run Multiple Detection Algorithms
    print("\nSTEP 3: Running Multiple Community Detection Algorithms")
    print("="*80)
    
    # Louvain
    analyzer.detect_communities_louvain()
    
    # Label Propagation
    analyzer.detect_communities_label_propagation()
    
    # Spectral Clustering
    analyzer.detect_communities_spectral(n_communities=12)
    
    print()
    
    # Step 4: Evaluate Against Ground Truth
    print("\nSTEP 4: Evaluating Detection Performance")
    
    algorithms = ['louvain', 'label_propagation', 'spectral']
    
    for algorithm in algorithms:
        analyzer.evaluate_against_ground_truth(ground_truth, algorithm)
    
    print()
    
    # Step 5: Identify Fraud Rings
    print("\nSTEP 5: Identifying High-Suspicion Fraud Rings")
    
    fraud_rings = analyzer.identify_fraud_rings('louvain', min_size=3)
    print()
    
    # Step 6: Generate Comprehensive Report
    print("\nSTEP 6: Generating Comprehensive Performance Report")
    print("="*80)
    
    print("\nALGORITHM PERFORMANCE SUMMARY:")
    print(f"{'Algorithm':<20} {'Communities':<15} {'Modularity':<12} {'Time (s)':<10}")
    print("-"*80)
    
    for alg, perf in analyzer.algorithm_performance.items():
        print(f"{alg:<20} {perf['n_communities']:<15} {perf['modularity']:<12.4f} {perf['time']:<10.2f}")
    
    print("\n" + "="*80)
    print("DETECTION ACCURACY METRICS:")
    print(f"{'Algorithm':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
    print("-"*80)
    
    for alg in algorithms:
        if alg in analyzer.evaluation_metrics:
            metrics = analyzer.evaluation_metrics[alg]
            print(f"{alg:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} {metrics['accuracy']:<12.4f}")
    
    # Check if targets met
    print("\n" + "="*80)
    print("PERFORMANCE TARGET VALIDATION:")
    
    best_algorithm = max(
        algorithms,
        key=lambda a: analyzer.evaluation_metrics[a]['f1_score']
    )
    
    best_metrics = analyzer.evaluation_metrics[best_algorithm]
    
    targets_met = (
        best_metrics['f1_score'] >= 0.95 and
        best_metrics['precision'] >= 0.88 and
        best_metrics['recall'] >= 0.85 and
        best_metrics['accuracy'] >= 0.90
    )
    
    print(f"\nBest Algorithm: {best_algorithm.upper()}")
    print(f"  F1-Score: {best_metrics['f1_score']:.4f} ({'✓ EXCEEDS' if best_metrics['f1_score'] >= 0.95 else '✗ BELOW'} 0.95 target)")
    print(f"  Precision: {best_metrics['precision']:.4f} ({'✓ EXCEEDS' if best_metrics['precision'] >= 0.88 else '✗ BELOW'} 0.88 target)")
    print(f"  Recall: {best_metrics['recall']:.4f} ({'✓ EXCEEDS' if best_metrics['recall'] >= 0.85 else '✗ BELOW'} 0.85 target)")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f} ({'✓ EXCEEDS' if best_metrics['accuracy'] >= 0.90 else '✗ BELOW'} 0.90 target)")
    
    if targets_met:
        print("\n╔" + "═"*88 + "╗")
        print("║" + " "*15 + "🎯 ALL PERFORMANCE TARGETS ACHIEVED! 🎯" + " "*32 + "║")
        print("║" + " "*18 + "Ready for Production Deployment" + " "*37 + "║")
        print("╚" + "═"*88 + "╝")
    
    print()
    
    # Step 7: Visualizations
    print("\nSTEP 7: Generating Comprehensive Visualizations")
    print("="*80)
    
    analyzer.visualize_communities('louvain')
    print()
    
    # Step 8: Save Results
    print("\nSTEP 8: Saving Analysis Results")
    print("="*80)
    
    analyzer.save_results('grainsecure_community_detection.pkl')
    
    # Save fraud ring report
    fraud_ring_df = pd.DataFrame(fraud_rings)
    fraud_ring_df.to_csv('detected_fraud_rings.csv', index=False)
    print("✓ Fraud ring report saved to 'detected_fraud_rings.csv'")
    print()
    
    print("="*80)
    print("COMMUNITY DETECTION ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print()
    print("Deliverables:")
    print("  1. grainsecure_community_detection.pkl - Complete analysis results")
    print("  2. detected_fraud_rings.csv - Identified fraud rings with suspicion scores")
    print("  3. community_detection_louvain.png - Network visualization")
    print()
    print("Key Findings:")
    print(f"  • Detected {len(fraud_rings)} communities")
    print(f"  • Identified {sum(1 for r in fraud_rings if r['is_likely_fraud'])} high-suspicion fraud rings")
    print(f"  • Best algorithm: {best_algorithm} (F1={best_metrics['f1_score']:.4f})")
    print(f"  • Average fraud detection accuracy: {best_metrics['accuracy']*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
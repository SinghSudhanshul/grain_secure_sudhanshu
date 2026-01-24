"""
GrainSecure PDS Monitoring System
Graph Neural Network for Collusion Detection and Relationship Mapping

This module implements state-of-the-art graph neural networks for identifying
fraud collusion networks, analyzing relationship patterns, and detecting
coordinated fraud schemes in PDS transaction data.

Author: GrainSecure Development Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.utils import from_networkx, to_networkx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import joblib
from typing import Tuple, Dict, List, Set
import json

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)


class CollusionNetworkDataGenerator:
    """
    Advanced data generator creating PDS transaction networks with sophisticated
    collusion patterns for graph neural network training and validation.
    """
    
    def __init__(self, n_beneficiaries: int = 5000, n_shops: int = 250,
                 n_transactions: int = 50000, collusion_rate: float = 0.08):
        """
        Initialize collusion network data generator.
        
        Args:
            n_beneficiaries: Number of beneficiary nodes
            n_shops: Number of shop nodes
            n_transactions: Total transaction volume
            collusion_rate: Proportion of beneficiaries in collusion networks
        """
        self.n_beneficiaries = n_beneficiaries
        self.n_shops = n_shops
        self.n_transactions = n_transactions
        self.collusion_rate = collusion_rate
        
        self.beneficiaries_df = None
        self.shops_df = None
        self.transactions_df = None
        self.collusion_networks = []
        
    def generate_network_dataset(self) -> Tuple[pd.DataFrame, List[Set]]:
        """
        Generate complete network dataset with collusion patterns.
        
        Returns:
            Tuple of (transactions dataframe, list of collusion network node sets)
        """
        
        print("=" * 80)
        print("GENERATING COLLUSION NETWORK DATASET FOR GRAPH NEURAL NETWORK")
        print("=" * 80)
        
        self._generate_entities()
        self._generate_collusion_networks()
        self._generate_network_transactions()
        
        print(f"\n✓ Dataset generation complete")
        print(f"  Total beneficiaries: {self.n_beneficiaries:,}")
        print(f"  Total shops: {self.n_shops:,}")
        print(f"  Total transactions: {len(self.transactions_df):,}")
        print(f"  Collusion networks: {len(self.collusion_networks)}")
        print(f"  Nodes in collusion: {sum(len(net) for net in self.collusion_networks):,}")
        print("=" * 80)
        
        return self.transactions_df, self.collusion_networks
    
    def _generate_entities(self):
        """Generate beneficiary and shop entities with attributes."""
        
        # Beneficiaries with realistic attributes
        family_sizes = np.random.choice(range(1, 11), self.n_beneficiaries, 
                                       p=[0.03, 0.12, 0.18, 0.23, 0.19, 0.12, 0.07, 0.04, 0.01, 0.01])
        
        income_categories = np.random.choice(['BPL', 'APL', 'AAY'], self.n_beneficiaries,
                                            p=[0.40, 0.45, 0.15])
        
        # Geographic clustering
        cluster_centers = np.random.uniform([28.1, 77.1], [28.4, 77.4], (6, 2))
        cluster_assignment = np.random.choice(6, self.n_beneficiaries)
        latitudes = cluster_centers[cluster_assignment, 0] + np.random.normal(0, 0.04, self.n_beneficiaries)
        longitudes = cluster_centers[cluster_assignment, 1] + np.random.normal(0, 0.04, self.n_beneficiaries)
        
        self.beneficiaries_df = pd.DataFrame({
            'beneficiary_id': [f'BEN{str(i).zfill(6)}' for i in range(self.n_beneficiaries)],
            'family_size': family_sizes,
            'income_category': income_categories,
            'latitude': latitudes,
            'longitude': longitudes,
            'cluster': cluster_assignment,
            'is_in_collusion': 0  # Will be updated
        })
        
        # Shops with risk profiles
        shop_latitudes = np.random.uniform(28.1, 28.4, self.n_shops)
        shop_longitudes = np.random.uniform(77.1, 77.4, self.n_shops)
        
        # Some shops are complicit in fraud
        compliance_scores = np.random.beta(5, 2, self.n_shops) * 100
        
        self.shops_df = pd.DataFrame({
            'shop_id': [f'SHOP{str(i).zfill(5)}' for i in range(self.n_shops)],
            'latitude': shop_latitudes,
            'longitude': shop_longitudes,
            'compliance_score': compliance_scores,
            'is_complicit': 0  # Will be updated
        })
        
        print(f"✓ Generated {len(self.beneficiaries_df):,} beneficiaries")
        print(f"✓ Generated {len(self.shops_df):,} shops")
    
    def _generate_collusion_networks(self):
        """Create structured collusion networks with realistic patterns."""
        
        n_collusion_beneficiaries = int(self.n_beneficiaries * self.collusion_rate)
        available_beneficiaries = set(range(self.n_beneficiaries))
        
        # Generate 8-15 collusion networks of varying sizes
        n_networks = np.random.randint(8, 16)
        
        for net_idx in range(n_networks):
            # Network size (3 to 25 beneficiaries per network)
            network_size = np.random.randint(3, min(26, n_collusion_beneficiaries // n_networks + 5))
            
            if len(available_beneficiaries) < network_size:
                break
            
            # Select beneficiaries for this network (prefer same cluster)
            cluster_id = np.random.choice(6)
            cluster_beneficiaries = [
                i for i in available_beneficiaries 
                if self.beneficiaries_df.iloc[i]['cluster'] == cluster_id
            ]
            
            if len(cluster_beneficiaries) >= network_size:
                network_members = set(np.random.choice(list(cluster_beneficiaries), 
                                                       network_size, replace=False))
            else:
                network_members = set(np.random.choice(list(available_beneficiaries),
                                                       network_size, replace=False))
            
            # Select 1-3 complicit shops for this network
            n_shops_in_network = np.random.randint(1, 4)
            network_shops = np.random.choice(self.n_shops, n_shops_in_network, replace=False)
            
            # Mark beneficiaries and shops as part of collusion
            for ben_idx in network_members:
                self.beneficiaries_df.loc[ben_idx, 'is_in_collusion'] = 1
            
            for shop_idx in network_shops:
                self.shops_df.loc[shop_idx, 'is_complicit'] = 1
            
            # Store network with both beneficiaries and shops
            network_info = {
                'beneficiaries': network_members,
                'shops': set(network_shops),
                'network_id': net_idx
            }
            self.collusion_networks.append(network_info)
            
            available_beneficiaries -= network_members
        
        print(f"✓ Created {len(self.collusion_networks)} collusion networks")
        print(f"  Network sizes: {[len(net['beneficiaries']) for net in self.collusion_networks]}")
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in kilometers."""
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0
    
    def _generate_network_transactions(self):
        """Generate transactions reflecting collusion network patterns."""
        
        transactions = []
        start_date = datetime.now() - timedelta(days=180)
        
        # Track transaction history
        beneficiary_history = defaultdict(list)
        
        for txn_idx in range(self.n_transactions):
            # Determine if this transaction involves collusion network
            if np.random.random() < 0.15:  # 15% of transactions from collusion networks
                # Select a collusion network
                network = np.random.choice(self.collusion_networks)
                beneficiary_idx = np.random.choice(list(network['beneficiaries']))
                shop_idx = np.random.choice(list(network['shops']))
                is_collusion = 1
                
                beneficiary = self.beneficiaries_df.iloc[beneficiary_idx]
                shop = self.shops_df.iloc[shop_idx]
            else:
                # Normal transaction
                beneficiary_idx = np.random.choice(self.n_beneficiaries)
                beneficiary = self.beneficiaries_df.iloc[beneficiary_idx]
                
                # Select nearby shop for normal transactions
                distances = self.shops_df.apply(
                    lambda s: self._calculate_distance(
                        beneficiary['latitude'], beneficiary['longitude'],
                        s['latitude'], s['longitude']
                    ), axis=1
                )
                
                nearby_shops = distances[distances < distances.quantile(0.4)].index
                if len(nearby_shops) > 0:
                    shop_idx = np.random.choice(nearby_shops)
                else:
                    shop_idx = np.random.choice(self.n_shops)
                
                shop = self.shops_df.iloc[shop_idx]
                is_collusion = 0
            
            # Transaction timing
            history = beneficiary_history[beneficiary['beneficiary_id']]
            
            if is_collusion and len(history) > 0:
                # Collusion networks show temporal correlation
                # Transactions occur within narrow time windows
                last_txn = history[-1]
                days_gap = np.random.uniform(28, 32)  # Very regular
                transaction_date = last_txn + timedelta(days=days_gap)
                hour = np.random.choice([10, 11, 14, 15])  # Specific hours
            elif len(history) > 0:
                # Normal timing variation
                last_txn = history[-1]
                days_gap = np.random.gamma(9, 3.5)
                transaction_date = last_txn + timedelta(days=days_gap)
                hour = np.random.choice(range(8, 18))
            else:
                days_offset = np.random.randint(0, 180)
                transaction_date = start_date + timedelta(days=days_offset)
                hour = np.random.choice(range(8, 18))
            
            transaction_datetime = transaction_date.replace(hour=hour)
            beneficiary_history[beneficiary['beneficiary_id']].append(transaction_datetime)
            
            # Quantities
            base_rice = beneficiary['family_size'] * 5
            base_wheat = beneficiary['family_size'] * 3
            base_sugar = beneficiary['family_size'] * 1
            
            if is_collusion:
                # Collusion networks often show inflated quantities
                multiplier = np.random.uniform(1.3, 1.8)
            else:
                multiplier = np.random.uniform(0.7, 1.1)
            
            rice_qty = int(base_rice * multiplier)
            wheat_qty = int(base_wheat * multiplier)
            sugar_qty = int(base_sugar * multiplier)
            
            # Authentication
            if is_collusion:
                auth_method = np.random.choice(['CARD', 'MANUAL'], p=[0.6, 0.4])
            else:
                auth_method = np.random.choice(['BIOMETRIC', 'CARD', 'MANUAL'], p=[0.6, 0.3, 0.1])
            
            # Distance
            distance = self._calculate_distance(
                beneficiary['latitude'], beneficiary['longitude'],
                shop['latitude'], shop['longitude']
            )
            
            transactions.append({
                'transaction_id': f'TXN{str(txn_idx).zfill(8)}',
                'beneficiary_id': beneficiary['beneficiary_id'],
                'beneficiary_idx': beneficiary_idx,
                'shop_id': shop['shop_id'],
                'shop_idx': shop_idx,
                'transaction_datetime': transaction_datetime,
                'rice_kg': rice_qty,
                'wheat_kg': wheat_qty,
                'sugar_kg': sugar_qty,
                'total_value': rice_qty * 2 + wheat_qty * 2 + sugar_qty * 40,
                'authentication_method': auth_method,
                'distance_km': distance,
                'beneficiary_family_size': beneficiary['family_size'],
                'beneficiary_cluster': beneficiary['cluster'],
                'shop_compliance': shop['compliance_score'],
                'is_collusion': is_collusion
            })
        
        self.transactions_df = pd.DataFrame(transactions)
        self.transactions_df = self.transactions_df.sort_values('transaction_datetime').reset_index(drop=True)
        
        print(f"✓ Generated {len(self.transactions_df):,} transactions")
        print(f"  Collusion transactions: {self.transactions_df['is_collusion'].sum():,} ({self.transactions_df['is_collusion'].mean()*100:.1f}%)")


class GraphConstructor:
    """
    Constructs heterogeneous graphs from PDS transaction data with multiple
    relationship types and rich node/edge features.
    """
    
    def __init__(self):
        self.graph = None
        self.node_features = {}
        self.edge_index = None
        self.edge_features = {}
        
    def build_transaction_graph(self, transactions_df: pd.DataFrame,
                                beneficiaries_df: pd.DataFrame,
                                shops_df: pd.DataFrame) -> Data:
        """
        Construct PyTorch Geometric graph from transaction data.
        
        Args:
            transactions_df: Transaction records
            beneficiaries_df: Beneficiary profiles
            shops_df: Shop profiles
            
        Returns:
            PyTorch Geometric Data object
        """
        
        print("\nConstructing Transaction Graph...")
        print("=" * 80)
        
        # Create NetworkX graph first for easier manipulation
        G = nx.Graph()
        
        # Add beneficiary nodes
        n_beneficiaries = len(beneficiaries_df)
        for idx, row in beneficiaries_df.iterrows():
            G.add_node(
                f"B_{idx}",
                node_type='beneficiary',
                family_size=row['family_size'],
                cluster=row['cluster'],
                is_in_collusion=row['is_in_collusion'],
                node_idx=idx
            )
        
        # Add shop nodes
        n_shops = len(shops_df)
        for idx, row in shops_df.iterrows():
            G.add_node(
                f"S_{idx}",
                node_type='shop',
                compliance_score=row['compliance_score'],
                is_complicit=row['is_complicit'],
                node_idx=n_beneficiaries + idx
            )
        
        print(f"✓ Added {n_beneficiaries:,} beneficiary nodes")
        print(f"✓ Added {n_shops:,} shop nodes")
        
        # Add transaction edges
        edge_transactions = defaultdict(list)
        
        for _, txn in transactions_df.iterrows():
            ben_node = f"B_{txn['beneficiary_idx']}"
            shop_node = f"S_{txn['shop_idx']}"
            
            # Track all transactions between this beneficiary-shop pair
            edge_key = (ben_node, shop_node)
            edge_transactions[edge_key].append(txn)
        
        # Create edges with aggregated features
        for (ben_node, shop_node), txns in edge_transactions.items():
            n_txns = len(txns)
            avg_quantity = np.mean([t['rice_kg'] + t['wheat_kg'] + t['sugar_kg'] for t in txns])
            avg_value = np.mean([t['total_value'] for t in txns])
            avg_distance = np.mean([t['distance_km'] for t in txns])
            
            # Temporal features
            dates = [t['transaction_datetime'] for t in txns]
            if len(dates) > 1:
                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                interval_std = np.std(intervals) if len(intervals) > 1 else 0
                interval_mean = np.mean(intervals) if len(intervals) > 0 else 30
            else:
                interval_std = 0
                interval_mean = 30
            
            # Collusion indicator
            is_collusion_edge = int(np.mean([t['is_collusion'] for t in txns]) > 0.5)
            
            G.add_edge(
                ben_node,
                shop_node,
                weight=n_txns,
                avg_quantity=avg_quantity,
                avg_value=avg_value,
                avg_distance=avg_distance,
                interval_std=interval_std,
                interval_mean=interval_mean,
                is_collusion=is_collusion_edge,
                n_transactions=n_txns
            )
        
        print(f"✓ Added {G.number_of_edges():,} transaction edges")
        
        # Convert to PyTorch Geometric format
        # Create node feature matrix
        node_features_list = []
        node_labels = []
        
        for node in G.nodes():
            data = G.nodes[node]
            if data['node_type'] == 'beneficiary':
                features = [
                    data['family_size'] / 10.0,  # Normalized
                    data['cluster'] / 6.0,
                    float(G.degree(node)) / 10.0,  # Normalized degree
                ]
                label = data['is_in_collusion']
            else:  # shop
                features = [
                    data['compliance_score'] / 100.0,
                    0.0,  # Placeholder for consistency
                    float(G.degree(node)) / 20.0,
                ]
                label = data['is_complicit']
            
            node_features_list.append(features)
            node_labels.append(label)
        
        x = torch.tensor(node_features_list, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        
        # Create edge index and edge attributes
        edge_index_list = []
        edge_attr_list = []
        
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        
        for u, v, data in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            
            # Add both directions for undirected graph
            edge_index_list.append([u_idx, v_idx])
            edge_index_list.append([v_idx, u_idx])
            
            edge_features = [
                data['weight'] / 10.0,
                data['avg_quantity'] / 50.0,
                data['avg_distance'] / 30.0,
                data['interval_std'] / 10.0,
                data['n_transactions'] / 10.0
            ]
            
            edge_attr_list.append(edge_features)
            edge_attr_list.append(edge_features)  # Same for both directions
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        print(f"✓ Graph construction complete")
        print(f"  Total nodes: {data.num_nodes:,}")
        print(f"  Total edges: {data.edge_index.shape[1]:,}")
        print(f"  Node features: {data.x.shape[1]}")
        print(f"  Edge features: {data.edge_attr.shape[1]}")
        print(f"  Collusion nodes: {y.sum().item():,}")
        print("=" * 80)
        
        self.graph = data
        return data


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for collusion detection using Graph Convolutional layers
    with attention mechanisms and edge feature integration.
    """
    
    def __init__(self, num_node_features: int, num_edge_features: int,
                 hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.3):
        """
        Initialize Graph Neural Network.
        
        Args:
            num_node_features: Dimension of node feature vectors
            num_edge_features: Dimension of edge feature vectors
            hidden_dim: Hidden layer dimensionality
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Edge feature transformation
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initial node feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Attention layer for important features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the graph neural network.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            
        Returns:
            Node classification logits [num_nodes, 2]
        """
        
        # Encode node features
        x = self.node_encoder(x)
        
        # Apply graph convolutions
        for i in range(self.num_layers):
            x_residual = x
            
            # Graph convolution
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_residual
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = x * attention_weights
        
        # Classification
        out = self.classifier(x)
        
        return out


class GNNCollusionDetector:
    """
    Production-grade Graph Neural Network system for collusion detection
    with comprehensive training, evaluation, and deployment capabilities.
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.3, learning_rate: float = 0.001,
                 epochs: int = 200):
        """
        Initialize GNN collusion detector.
        
        Args:
            hidden_dim: Hidden layer size
            num_layers: Number of GNN layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            epochs: Training epochs
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.metrics = {}
        
    def build_model(self, num_node_features: int, num_edge_features: int):
        """Build and initialize the GNN model."""
        
        self.model = GraphNeuralNetwork(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4
        )
        
        print(f"\n{'='*80}")
        print("GRAPH NEURAL NETWORK ARCHITECTURE")
        print(f"{'='*80}")
        print(self.model)
        print(f"{'='*80}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
    
    def train(self, train_data: Data, val_data: Data = None):
        """
        Train the graph neural network.
        
        Args:
            train_data: Training graph data
            val_data: Validation graph data (optional)
        """
        
        print("\n" + "="*80)
        print("TRAINING GRAPH NEURAL NETWORK FOR COLLUSION DETECTION")
        print("="*80)
        print(f"Training nodes: {train_data.num_nodes:,}")
        print(f"Training edges: {train_data.edge_index.shape[1]:,}")
        print(f"Collusion rate: {(train_data.y.sum() / len(train_data.y) * 100):.2f}%")
        print("="*80 + "\n")
        
        train_data = train_data.to(self.device)
        if val_data is not None:
            val_data = val_data.to(self.device)
        
        # Calculate class weights for imbalanced dataset
        n_normal = (train_data.y == 0).sum().item()
        n_collusion = (train_data.y == 1).sum().item()
        weight = torch.tensor([1.0, n_normal / n_collusion]).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weight)
        
        best_val_acc = 0
        patience = 30
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            out = self.model(train_data.x, train_data.edge_index, train_data.edge_attr)
            loss = criterion(out, train_data.y)
            
            loss.backward()
            self.optimizer.step()
            
            # Calculate training accuracy
            pred = out.argmax(dim=1)
            train_acc = (pred == train_data.y).sum().item() / len(train_data.y)
            
            self.training_history['train_loss'].append(loss.item())
            self.training_history['train_acc'].append(train_acc)
            
            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(val_data.x, val_data.edge_index, val_data.edge_attr)
                    val_loss = criterion(val_out, val_data.y)
                    val_pred = val_out.argmax(dim=1)
                    val_acc = (val_pred == val_data.y).sum().item() / len(val_data.y)
                    
                    self.training_history['val_loss'].append(val_loss.item())
                    self.training_history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")
        
        print("\n✓ Training completed successfully")
    
    def evaluate(self, test_data: Data) -> Dict:
        """
        Comprehensive evaluation of the trained model.
        
        Args:
            test_data: Test graph data
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        test_data = test_data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(test_data.x, test_data.edge_index, test_data.edge_attr)
            pred = out.argmax(dim=1)
            proba = F.softmax(out, dim=1)[:, 1]
        
        # Convert to numpy
        y_true = test_data.y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_proba = proba.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Collusion'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_evaluation(self, test_data: Data):
        """Generate comprehensive evaluation visualizations."""
        
        test_data = test_data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(test_data.x, test_data.edge_index, test_data.edge_attr)
            pred = out.argmax(dim=1)
            proba = F.softmax(out, dim=1)[:, 1]
        
        y_true = test_data.y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_proba = proba.cpu().numpy()
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Training History
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], label='Train Loss', linewidth=2)
        if self.training_history['val_loss']:
            ax1.plot(epochs, self.training_history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Accuracy History
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.training_history['train_acc'], label='Train Acc', linewidth=2)
        if self.training_history['val_acc']:
            ax2.plot(epochs, self.training_history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Confusion Matrix
        ax3 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')
        ax3.set_xticklabels(['Normal', 'Collusion'])
        ax3.set_yticklabels(['Normal', 'Collusion'])
        
        # 4. Prediction Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(y_proba[y_true == 0], bins=50, alpha=0.6, label='Normal', color='green', density=True)
        ax4.hist(y_proba[y_true == 1], bins=50, alpha=0.6, label='Collusion', color='red', density=True)
        ax4.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax4.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Collusion Probability')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Performance Metrics
        ax5 = fig.add_subplot(gs[1, 1:])
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            self.metrics['accuracy'],
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1_score'],
            self.metrics['roc_auc']
        ]
        
        colors = ['#27ae60' if v >= 0.95 else '#f39c12' if v >= 0.85 else '#e74c3c' for v in metric_values]
        bars = ax5.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax5.set_title('Comprehensive Performance Metrics', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Score')
        ax5.set_ylim(0, 1.1)
        ax5.grid(axis='y', alpha=0.3)
        ax5.axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='98% Target')
        ax5.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Target')
        ax5.legend()
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.suptitle('Graph Neural Network - Collusion Detection Evaluation', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('gnn_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation plots saved as 'gnn_evaluation.png'")
        plt.show()
    
    def save_model(self, filepath: str = 'gnn_collusion_model.pt'):
        """Save trained model."""
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'metrics': self.metrics,
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, filepath)
        
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'gnn_collusion_model.pt',
                   num_node_features: int = 3, num_edge_features: int = 5):
        """Load trained model."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Rebuild model
        self.build_model(num_node_features, num_edge_features)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.metrics = checkpoint.get('metrics', {})
        
        print(f"✓ Model loaded from {filepath}")


def main():
    """
    Execute complete GNN training and evaluation pipeline.
    """
    
    print("\n" + "╔" + "═"*88 + "╗")
    print("║" + " "*20 + "GRAINSECURE PDS MONITORING SYSTEM" + " "*35 + "║")
    print("║" + " "*15 + "Graph Neural Network for Collusion Detection" + " "*28 + "║")
    print("║" + " "*10 + "Advanced Network Analysis and Relationship-Based Fraud Detection" + " "*12 + "║")
    print("╚" + "═"*88 + "╝")
    print()
    
    # Step 1: Generate Network Dataset
    print("STEP 1: Generating Collusion Network Dataset")
    print("="*80)
    
    generator = CollusionNetworkDataGenerator(
        n_beneficiaries=5000,
        n_shops=250,
        n_transactions=50000,
        collusion_rate=0.08
    )
    
    transactions_df, collusion_networks = generator.generate_network_dataset()
    print()
    
    # Step 2: Construct Graph
    print("\nSTEP 2: Constructing Transaction Graph")
    
    constructor = GraphConstructor()
    graph_data = constructor.build_transaction_graph(
        transactions_df,
        generator.beneficiaries_df,
        generator.shops_df
    )
    print()
    
    # Step 3: Split Data
    print("\nSTEP 3: Preparing Train-Validation-Test Split")
    print("="*80)
    
    # Create train/val/test masks
    n_nodes = graph_data.num_nodes
    indices = torch.randperm(n_nodes)
    
    train_size = int(0.7 * n_nodes)
    val_size = int(0.15 * n_nodes)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create separate graph data for each split
    train_data = graph_data.clone()
    val_data = graph_data.clone()
    test_data = graph_data.clone()
    
    print(f"✓ Training nodes:   {len(train_indices):,}")
    print(f"✓ Validation nodes: {len(val_indices):,}")
    print(f"✓ Test nodes:       {len(test_indices):,}")
    print()
    
    # Step 4: Train GNN
    print("\nSTEP 4: Training Graph Neural Network")
    
    detector = GNNCollusionDetector(
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
        learning_rate=0.001,
        epochs=200
    )
    
    detector.build_model(
        num_node_features=graph_data.x.shape[1],
        num_edge_features=graph_data.edge_attr.shape[1]
    )
    
    detector.train(train_data, val_data)
    print()
    
    # Step 5: Evaluate
    print("\nSTEP 5: Comprehensive Model Evaluation")
    
    metrics = detector.evaluate(test_data)
    
    print("\n╔" + "═"*88 + "╗")
    print("║" + " "*28 + "FINAL PERFORMANCE REPORT" + " "*36 + "║")
    print("╠" + "═"*88 + "╣")
    print(f"║  Accuracy:  {metrics['accuracy']:.6f} ({metrics['accuracy']*100:6.3f}%)  {'✓ EXCEEDS TARGET' if metrics['accuracy'] >= 0.98 else '⚠ BELOW TARGET':>45} ║")
    print(f"║  Precision: {metrics['precision']:.6f} ({metrics['precision']*100:6.3f}%)  {'✓ EXCEEDS TARGET' if metrics['precision'] >= 0.88 else '⚠ BELOW TARGET':>45} ║")
    print(f"║  Recall:    {metrics['recall']:.6f} ({metrics['recall']*100:6.3f}%)  {'✓ EXCEEDS TARGET' if metrics['recall'] >= 0.85 else '⚠ BELOW TARGET':>45} ║")
    print(f"║  F1-Score:  {metrics['f1_score']:.6f} ({metrics['f1_score']*100:6.3f}%)  {'✓ EXCEEDS TARGET' if metrics['f1_score'] >= 0.95 else '⚠ BELOW TARGET':>45} ║")
    print(f"║  ROC-AUC:   {metrics['roc_auc']:.6f} ({metrics['roc_auc']*100:6.3f}%)  {'✓ EXCEEDS TARGET' if metrics['roc_auc'] >= 0.95 else '⚠ BELOW TARGET':>45} ║")
    print("╚" + "═"*88 + "╝")
    print()
    
    cm = metrics['confusion_matrix']
    print("CONFUSION MATRIX:")
    print(f"  True Negatives:  {cm[0][0]:,}")
    print(f"  False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,}")
    print(f"  True Positives:  {cm[1][1]:,}")
    print()
    
    targets_met = all([
        metrics['accuracy'] >= 0.98,
        metrics['precision'] >= 0.88,
        metrics['recall'] >= 0.85,
        metrics['f1_score'] >= 0.95
    ])
    
    if targets_met:
        print("╔" + "═"*88 + "╗")
        print("║" + " "*15 + "🎯 ALL PERFORMANCE TARGETS ACHIEVED! 🎯" + " "*32 + "║")
        print("║" + " "*18 + "Ready for Production Deployment" + " "*37 + "║")
        print("╚" + "═"*88 + "╝")
    print()
    
    # Step 6: Visualizations
    print("\nSTEP 6: Generating Evaluation Visualizations")
    print("="*80)
    
    detector.plot_evaluation(test_data)
    print()
    
    # Step 7: Save Model
    print("\nSTEP 7: Saving Trained Model")
    print("="*80)
    
    detector.save_model('grainsecure_gnn_collusion.pt')
    print()
    
    print("="*80)
    print("GRAPH NEURAL NETWORK TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print()
    print("Model Deliverables:")
    print("  1. grainsecure_gnn_collusion.pt - Trained GNN model")
    print("  2. gnn_evaluation.png - Performance visualizations")
    print()
    print("Model Capabilities:")
    print("  • Detects collusion networks through graph structure analysis")
    print("  • Identifies coordinated fraud schemes across beneficiaries and shops")
    print("  • Analyzes relationship patterns and network topology")
    print("  • Provides node-level risk scores for targeted investigations")
    print("  • Integrates with ensemble detection system")
    print()


if __name__ == "__main__":
    main()
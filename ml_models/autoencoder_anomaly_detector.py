"""
Author: GrainSecure Development Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib
from typing import Tuple, Dict, List
import json

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# Configure TensorFlow for optimal performance
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


class AdvancedPDSDataGenerator:
    """
    Enhanced synthetic PDS transaction data generator with sophisticated
    fraud pattern injection for autoencoder training and validation.
    """
    
    def __init__(self, n_beneficiaries: int = 15000, n_shops: int = 600,
                 n_transactions: int = 150000, fraud_rate: float = 0.05):
        """
        Initialize advanced data generator.
        
        Args:
            n_beneficiaries: Number of beneficiary households
            n_shops: Number of fair price shops
            n_transactions: Total transactions to generate
            fraud_rate: Proportion of fraudulent transactions
        """
        self.n_beneficiaries = n_beneficiaries
        self.n_shops = n_shops
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        
        self.beneficiaries_df = None
        self.shops_df = None
        self.transactions_df = None
        
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete PDS transaction dataset with all entities."""
        
        print("Generating comprehensive PDS dataset...")
        
        # Generate beneficiary profiles
        self._generate_beneficiaries()
        print(f"  ✓ Generated {len(self.beneficiaries_df):,} beneficiaries")
        
        # Generate shop profiles
        self._generate_shops()
        print(f"  ✓ Generated {len(self.shops_df):,} shops")
        
        # Generate transactions with fraud patterns
        self._generate_transactions()
        print(f"  ✓ Generated {len(self.transactions_df):,} transactions")
        print(f"  ✓ Fraud rate: {self.transactions_df['is_fraud'].mean()*100:.2f}%")
        
        return self.transactions_df
    
    def _generate_beneficiaries(self):
        """Generate beneficiary profiles with demographic realism."""
        
        income_categories = np.random.choice(
            ['BPL', 'APL', 'AAY'],
            size=self.n_beneficiaries,
            p=[0.42, 0.43, 0.15]
        )
        
        family_sizes = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            size=self.n_beneficiaries,
            p=[0.04, 0.14, 0.19, 0.24, 0.17, 0.11, 0.06, 0.03, 0.01, 0.01]
        )
        
        # Geographic clustering with hotspots
        cluster_centers = np.random.uniform([28.1, 77.1], [28.4, 77.4], (5, 2))
        cluster_assignment = np.random.choice(5, self.n_beneficiaries)
        latitudes = cluster_centers[cluster_assignment, 0] + np.random.normal(0, 0.05, self.n_beneficiaries)
        longitudes = cluster_centers[cluster_assignment, 1] + np.random.normal(0, 0.05, self.n_beneficiaries)
        
        entitlements = []
        for size, category in zip(family_sizes, income_categories):
            multiplier = 1.3 if category in ['BPL', 'AAY'] else 0.9
            entitlements.append({
                'rice_kg': int(size * 5 * multiplier),
                'wheat_kg': int(size * 3 * multiplier),
                'sugar_kg': int(size * 1 * multiplier),
                'kerosene_l': int(size * 2 * multiplier) if category == 'BPL' else 0
            })
        
        self.beneficiaries_df = pd.DataFrame({
            'beneficiary_id': [f'BEN{str(i).zfill(7)}' for i in range(self.n_beneficiaries)],
            'family_size': family_sizes,
            'income_category': income_categories,
            'latitude': latitudes,
            'longitude': longitudes,
            'monthly_entitlement': entitlements,
            'cluster': cluster_assignment
        })
    
    def _generate_shops(self):
        """Generate fair price shop profiles."""
        
        # Shops located near population clusters
        shop_latitudes = np.random.uniform(28.1, 28.4, self.n_shops)
        shop_longitudes = np.random.uniform(77.1, 77.4, self.n_shops)
        
        operational_status = np.random.choice(
            ['ACTIVE', 'SUSPENDED', 'CLOSED'],
            size=self.n_shops,
            p=[0.91, 0.06, 0.03]
        )
        
        # Risk scores for shops (some inherently riskier)
        base_risk = np.random.beta(2, 8, self.n_shops) * 100
        
        self.shops_df = pd.DataFrame({
            'shop_id': [f'SHOP{str(i).zfill(5)}' for i in range(self.n_shops)],
            'latitude': shop_latitudes,
            'longitude': shop_longitudes,
            'operational_status': operational_status,
            'base_risk_score': base_risk
        })
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance in kilometers."""
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0
    
    def _generate_transactions(self):
        """Generate transactions with sophisticated fraud patterns."""
        
        transactions = []
        start_date = datetime.now() - pd.Timedelta(days=180)
        
        n_fraud = int(self.n_transactions * self.fraud_rate)
        fraud_indices = set(np.random.choice(self.n_transactions, n_fraud, replace=False))
        
        # Track beneficiary transaction history for temporal features
        beneficiary_history = {bid: [] for bid in self.beneficiaries_df['beneficiary_id']}
        
        for i in range(self.n_transactions):
            is_fraud = i in fraud_indices
            
            # Select beneficiary
            if is_fraud and np.random.random() < 0.3:
                # Some fraud uses same beneficiaries repeatedly
                if len(beneficiary_history) > 0:
                    frequent_users = [k for k, v in beneficiary_history.items() if len(v) > 5]
                    if frequent_users:
                        beneficiary = self.beneficiaries_df[
                            self.beneficiaries_df['beneficiary_id'] == np.random.choice(frequent_users)
                        ].iloc[0]
                    else:
                        beneficiary = self.beneficiaries_df.sample(1).iloc[0]
                else:
                    beneficiary = self.beneficiaries_df.sample(1).iloc[0]
            else:
                beneficiary = self.beneficiaries_df.sample(1).iloc[0]
            
            # Fraud type determination
            if is_fraud:
                fraud_type = np.random.choice([
                    'ghost_beneficiary',
                    'quantity_manipulation', 
                    'collusion_network',
                    'identity_fraud',
                    'timing_anomaly',
                    'geographic_impossibility'
                ], p=[0.20, 0.25, 0.15, 0.15, 0.15, 0.10])
            else:
                fraud_type = 'legitimate'
            
            # Shop selection
            distances = self.shops_df.apply(
                lambda x: self._calculate_distance(
                    beneficiary['latitude'], beneficiary['longitude'],
                    x['latitude'], x['longitude']
                ), axis=1
            )
            
            if fraud_type == 'geographic_impossibility':
                # Deliberately choose distant shop
                far_shops = self.shops_df[distances > distances.quantile(0.85)]
                shop = far_shops.sample(1).iloc[0] if len(far_shops) > 0 else self.shops_df.sample(1).iloc[0]
            elif fraud_type == 'collusion_network':
                # Use high-risk shops
                high_risk_shops = self.shops_df[self.shops_df['base_risk_score'] > 70]
                shop = high_risk_shops.sample(1).iloc[0] if len(high_risk_shops) > 0 else self.shops_df.sample(1).iloc[0]
            else:
                # Normal proximity-based selection
                nearby_shops = self.shops_df[distances < distances.quantile(0.4)]
                shop = nearby_shops.sample(1).iloc[0] if len(nearby_shops) > 0 else self.shops_df.sample(1).iloc[0]
            
            # Transaction timing
            history = beneficiary_history[beneficiary['beneficiary_id']]
            
            if fraud_type == 'ghost_beneficiary' and len(history) > 0:
                # Mechanically precise intervals
                last_txn = history[-1]
                days_gap = 30.0  # Exactly monthly
                transaction_date = last_txn + pd.Timedelta(days=days_gap)
                hour = 10  # Always same hour
            elif fraud_type == 'timing_anomaly':
                # Suspicious hours
                days_offset = np.random.randint(0, 180)
                transaction_date = start_date + pd.Timedelta(days=days_offset)
                hour = np.random.choice([1, 2, 3, 22, 23])
            elif len(history) > 0:
                # Realistic intervals with variation
                last_txn = history[-1]
                days_gap = np.random.gamma(30, 0.3)  # Mean ~30 days, realistic variation
                transaction_date = last_txn + pd.Timedelta(days=days_gap)
                hour = np.random.choice(range(7, 19), p=[0.02, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.05, 0.03, 0.01, 0.01])
            else:
                days_offset = np.random.randint(0, 180)
                transaction_date = start_date + pd.Timedelta(days=days_offset)
                hour = np.random.choice(range(7, 19))
            
            transaction_datetime = transaction_date.replace(hour=hour)
            beneficiary_history[beneficiary['beneficiary_id']].append(transaction_datetime)
            
            # Quantities
            entitlement = beneficiary['monthly_entitlement']
            
            if fraud_type == 'quantity_manipulation':
                rice_qty = int(entitlement['rice_kg'] * np.random.uniform(1.8, 3.0))
                wheat_qty = int(entitlement['wheat_kg'] * np.random.uniform(1.8, 3.0))
                sugar_qty = int(entitlement['sugar_kg'] * np.random.uniform(1.8, 3.0))
                kerosene_qty = int(entitlement.get('kerosene_l', 0) * np.random.uniform(2.0, 4.0))
            elif fraud_type == 'ghost_beneficiary':
                # Exactly entitled amount, no variation
                rice_qty = entitlement['rice_kg']
                wheat_qty = entitlement['wheat_kg']
                sugar_qty = entitlement['sugar_kg']
                kerosene_qty = entitlement.get('kerosene_l', 0)
            elif fraud_type == 'identity_fraud':
                # Slightly inflated but not extreme
                rice_qty = int(entitlement['rice_kg'] * np.random.uniform(1.2, 1.5))
                wheat_qty = int(entitlement['wheat_kg'] * np.random.uniform(1.2, 1.5))
                sugar_qty = int(entitlement['sugar_kg'] * np.random.uniform(1.2, 1.5))
                kerosene_qty = int(entitlement.get('kerosene_l', 0) * np.random.uniform(1.3, 1.6))
            else:
                # Normal variation
                rice_qty = int(entitlement['rice_kg'] * np.random.uniform(0.6, 1.15))
                wheat_qty = int(entitlement['wheat_kg'] * np.random.uniform(0.6, 1.15))
                sugar_qty = int(entitlement['sugar_kg'] * np.random.uniform(0.6, 1.15))
                kerosene_qty = int(entitlement.get('kerosene_l', 0) * np.random.uniform(0.7, 1.1))
            
            # Authentication method
            if is_fraud:
                auth_method = np.random.choice(
                    ['BIOMETRIC', 'CARD', 'MANUAL'],
                    p=[0.15, 0.35, 0.50]
                )
            else:
                auth_method = np.random.choice(
                    ['BIOMETRIC', 'CARD', 'MANUAL'],
                    p=[0.65, 0.28, 0.07]
                )
            
            distance_km = self._calculate_distance(
                beneficiary['latitude'], beneficiary['longitude'],
                shop['latitude'], shop['longitude']
            )
            
            transactions.append({
                'transaction_id': f'TXN{str(i).zfill(9)}',
                'beneficiary_id': beneficiary['beneficiary_id'],
                'shop_id': shop['shop_id'],
                'transaction_datetime': transaction_datetime,
                'rice_kg': rice_qty,
                'wheat_kg': wheat_qty,
                'sugar_kg': sugar_qty,
                'kerosene_l': kerosene_qty,
                'total_value': rice_qty * 2 + wheat_qty * 2 + sugar_qty * 40 + kerosene_qty * 15,
                'authentication_method': auth_method,
                'distance_km': distance_km,
                'shop_risk_score': shop['base_risk_score'],
                'is_fraud': 1 if is_fraud else 0,
                'fraud_type': fraud_type
            })
        
        self.transactions_df = pd.DataFrame(transactions)
        self.transactions_df = self.transactions_df.sort_values('transaction_datetime').reset_index(drop=True)


class EnhancedFeatureEngineer:
    """
    Advanced feature engineering specifically optimized for autoencoder
    neural network input, including normalization and encoding strategies.
    """
    
    def __init__(self):
        self.beneficiary_stats = {}
        self.shop_stats = {}
        
    def engineer_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive feature set optimized for neural network processing.
        
        Args:
            transactions_df: Raw transaction data
            
        Returns:
            Feature-engineered dataframe
        """
        
        df = transactions_df.copy()
        
        # Temporal features with cyclical encoding
        df['hour'] = pd.to_datetime(df['transaction_datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['transaction_datetime']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['transaction_datetime']).dt.day
        df['month'] = pd.to_datetime(df['transaction_datetime']).dt.month
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Transaction interval features
        df = df.sort_values(['beneficiary_id', 'transaction_datetime'])
        df['days_since_last_txn'] = df.groupby('beneficiary_id')['transaction_datetime'].diff().dt.total_seconds() / 86400
        df['days_since_last_txn'].fillna(30, inplace=True)
        
        # Rolling statistics for timing regularity
        df['timing_mean'] = df.groupby('beneficiary_id')['days_since_last_txn'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['timing_std'] = df.groupby('beneficiary_id')['days_since_last_txn'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        df['timing_regularity'] = df['timing_std'] / (df['timing_mean'] + 1)
        df['timing_regularity'].fillna(0.5, inplace=True)
        
        # Transaction frequency features
        df['beneficiary_txn_count'] = df.groupby('beneficiary_id').cumcount() + 1
        df['shop_txn_count'] = df.groupby('shop_id').cumcount() + 1
        
        
        # Shop diversity metrics
        df['unique_shops_used'] = df.groupby('beneficiary_id')['shop_id'].transform('nunique')
        
        # Quantity features
        df['total_quantity_kg'] = df['rice_kg'] + df['wheat_kg'] + df['sugar_kg']
        df['total_commodities'] = df['total_quantity_kg'] + df['kerosene_l']
        
        # Commodity ratios
        df['rice_ratio'] = df['rice_kg'] / (df['total_quantity_kg'] + 1)
        df['wheat_ratio'] = df['wheat_kg'] / (df['total_quantity_kg'] + 1)
        df['sugar_ratio'] = df['sugar_kg'] / (df['total_quantity_kg'] + 1)
        df['kerosene_ratio'] = df['kerosene_l'] / (df['total_commodities'] + 1)
        
        # Rolling quantity statistics
        df['quantity_mean'] = df.groupby('beneficiary_id')['total_quantity_kg'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['quantity_std'] = df.groupby('beneficiary_id')['total_quantity_kg'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        df['quantity_deviation'] = np.abs(df['total_quantity_kg'] - df['quantity_mean']) / (df['quantity_std'] + 1)
        df['quantity_deviation'].fillna(0, inplace=True)
        
        # Authentication strength encoding
        auth_encoding = {'BIOMETRIC': 1.0, 'CARD': 0.6, 'MANUAL': 0.2}
        df['auth_strength'] = df['authentication_method'].map(auth_encoding)
        
        # Geographic features
        df['log_distance'] = np.log1p(df['distance_km'])
        df['is_distant_shop'] = (df['distance_km'] > 15).astype(float)
        df['is_very_distant'] = (df['distance_km'] > 30).astype(float)
        
        # Unusual timing indicators
        df['is_unusual_hour'] = df['hour'].apply(lambda x: 1.0 if x < 6 or x > 20 else 0.0)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(float)
        
        # Value-based features
        df['value_per_kg'] = df['total_value'] / (df['total_quantity_kg'] + 1)
        df['log_total_value'] = np.log1p(df['total_value'])
        
        # Shop risk integration
        df['normalized_shop_risk'] = df['shop_risk_score'] / 100.0
        
        # Interaction features
        df['risk_distance_interaction'] = df['normalized_shop_risk'] * df['log_distance']
        df['auth_distance_interaction'] = df['auth_strength'] * df['log_distance']
        df['quantity_regularity_interaction'] = df['quantity_deviation'] * df['timing_regularity']
        
        return df


class AutoencoderAnomalyDetector:
    """
    Production-grade Autoencoder Neural Network for PDS fraud detection.
    Implements deep learning-based anomaly detection through reconstruction error analysis.
    """
    
    def __init__(self, encoding_dim: int = 4, hidden_layers: List[int] = [32, 16, 8],
                 learning_rate: float = 0.001, batch_size: int = 128, epochs: int = 100):
        """
        Initialize Autoencoder detector.
        
        Args:
            encoding_dim: Dimension of compressed representation (bottleneck)
            hidden_layers: Neuron counts for encoder hidden layers
            learning_rate: Initial learning rate for Adam optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_names = None
        self.threshold = None
        self.training_history = None
        self.metrics = {}
        
    def _build_autoencoder(self, input_dim: int):
        """
        Construct autoencoder architecture with regularization.
        
        Args:
            input_dim: Number of input features
        """
        
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for i, units in enumerate(self.hidden_layers):
            encoded = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name=f'encoder_{i}'
            )(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(0.3)(encoded)
        
        # Bottleneck
        bottleneck = layers.Dense(
            self.encoding_dim,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='bottleneck'
        )(encoded)
        
        # Decoder (mirror of encoder)
        decoded = bottleneck
        for i, units in enumerate(reversed(self.hidden_layers)):
            decoded = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name=f'decoder_{i}'
            )(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(0.3)(decoded)
        
        # Output layer
        output_layer = layers.Dense(
            input_dim,
            activation='linear',
            name='output'
        )(decoded)
        
        # Create models
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer, name='autoencoder')
        self.encoder = Model(inputs=input_layer, outputs=bottleneck, name='encoder')
        
        # Compile
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        print("Autoencoder Architecture:")
        self.autoencoder.summary()
    
    def train(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, y_train: pd.Series = None):
        """
        Train autoencoder on normal transactions only.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            y_train: Training labels (used to filter normal transactions)
        """
        
        print("\nTraining Autoencoder Neural Network...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        
        # Filter to only normal transactions for training
        if y_train is not None:
            X_train_normal = X_train[y_train == 0]
            print(f"Normal transactions for training: {len(X_train_normal):,}")
        else:
            X_train_normal = X_train
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_normal)
        
        # Build architecture
        self._build_autoencoder(X_train.shape[1])
        self.feature_names = X_train.columns.tolist()
        
        # Prepare validation data
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, X_val_scaled)
        else:
            validation_split = 0.15
            validation_data = None
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'autoencoder_best.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train
        history = self.autoencoder.fit(
            X_train_scaled,
            X_train_scaled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data if X_val is not None else None,
            validation_split=validation_split if X_val is None else 0,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        print("\n✓ Training completed successfully")
        
        # Calculate reconstruction error threshold on normal training data
        train_reconstructions = self.autoencoder.predict(X_train_scaled, verbose=0)
        train_mse = np.mean(np.power(X_train_scaled - train_reconstructions, 2), axis=1)
        
        # Set threshold at 95th percentile of normal reconstruction errors
        self.threshold = np.percentile(train_mse, 95)
        print(f"✓ Anomaly threshold set at: {self.threshold:.6f}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predictions, reconstruction_errors)
        """
        
        X_scaled = self.scaler.transform(X)
        
        # Get reconstructions
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction errors
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        
        # Predict anomalies (1 for anomaly, 0 for normal)
        predictions = (mse > self.threshold).astype(int)
        
        return predictions, mse
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive evaluation with multiple metrics.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        print("\nEvaluating model performance...")
        
        predictions, reconstruction_errors = self.predict(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, reconstruction_errors)
        
        # Classification report
        class_report = classification_report(
            y_test, predictions,
            target_names=['Normal', 'Fraud'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'threshold': self.threshold
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generate comprehensive evaluation visualizations."""
        
        predictions, reconstruction_errors = self.predict(X_test)
        
        fig = plt.figure(figsize=(18, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Training History
        ax1 = fig.add_subplot(gs[0, :2])
        if self.training_history:
            ax1.plot(self.training_history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in self.training_history:
                ax1.plot(self.training_history['val_loss'], label='Validation Loss', linewidth=2)
            ax1.set_title('Model Training History', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Mean Squared Error')
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        ax2.set_xticklabels(['Normal', 'Fraud'])
        ax2.set_yticklabels(['Normal', 'Fraud'])
        
        # 3. Reconstruction Error Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(reconstruction_errors[y_test == 0], bins=50, alpha=0.6, label='Normal', color='green', density=True)
        ax3.hist(reconstruction_errors[y_test == 1], bins=50, alpha=0.6, label='Fraud', color='red', density=True)
        ax3.axvline(self.threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={self.threshold:.4f}')
        ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Reconstruction Error (MSE)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. ROC Curve
        ax4 = fig.add_subplot(gs[1, 1])
        fpr, tpr, _ = roc_curve(y_test, reconstruction_errors)
        ax4.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={self.metrics["roc_auc"]:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax4.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.fill_between(fpr, tpr, alpha=0.2)
        
        # 5. Precision-Recall Curve
        ax5 = fig.add_subplot(gs[1, 2])
        precision, recall, _ = precision_recall_curve(y_test, reconstruction_errors)
        ax5.plot(recall, precision, linewidth=2, color='purple')
        ax5.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.grid(alpha=0.3)
        ax5.fill_between(recall, precision, alpha=0.2)
        
        # 6. Performance Metrics
        ax6 = fig.add_subplot(gs[2, :])
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            self.metrics['accuracy'],
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1_score'],
            self.metrics['roc_auc']
        ]
        
        colors = ['#27ae60' if v >= 0.95 else '#f39c12' if v >= 0.85 else '#e74c3c' for v in metric_values]
        bars = ax6.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_title('Comprehensive Performance Metrics', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score')
        ax6.set_ylim(0, 1.1)
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax6.axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='98% Target')
        ax6.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Target')
        ax6.legend(loc='lower right')
        
        plt.suptitle('Autoencoder Anomaly Detection - Comprehensive Evaluation', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('autoencoder_evaluation.png', dpi=300, bbox_inches='tight')
        print("✓ Evaluation plots saved as 'autoencoder_evaluation.png'")
        plt.show()
    
    def save_model(self, filepath: str = 'autoencoder_model'):
        """Save complete model package."""
        
        # Save Keras model
        self.autoencoder.save(f'{filepath}_autoencoder.h5')
        self.encoder.save(f'{filepath}_encoder.h5')
        
        # Save other components
        model_package = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'metrics': self.metrics,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers
        }
        
        joblib.dump(model_package, f'{filepath}_package.pkl')
        print(f"✓ Model saved to {filepath}_*.h5 and {filepath}_package.pkl")
    
    def load_model(self, filepath: str = 'autoencoder_model'):
        """Load complete model package."""
        
        self.autoencoder = keras.models.load_model(f'{filepath}_autoencoder.h5')
        self.encoder = keras.models.load_model(f'{filepath}_encoder.h5')
        
        model_package = joblib.load(f'{filepath}_package.pkl')
        self.scaler = model_package['scaler']
        self.feature_names = model_package['feature_names']
        self.threshold = model_package['threshold']
        self.metrics = model_package.get('metrics', {})
        
        print(f"✓ Model loaded from {filepath}_*.h5 and {filepath}_package.pkl")


def main():
    """
    Execute complete autoencoder training and evaluation pipeline.
    """
    
    print("=" * 90)
    print("GRAINSECURE PDS MONITORING SYSTEM")
    print("Autoencoder Neural Network Anomaly Detection Model")
    print("Deep Learning-Based Fraud Detection through Reconstruction Error Analysis")
    print("=" * 90)
    print()
    
    # Step 1: Generate Enhanced Dataset
    print("STEP 1: Generating Enhanced Synthetic PDS Transaction Dataset")
    print("-" * 90)
    
    generator = AdvancedPDSDataGenerator(
        n_beneficiaries=15000,
        n_shops=600,
        n_transactions=150000,
        fraud_rate=0.05
    )
    
    transactions = generator.generate_complete_dataset()
    print()
    
    # Step 2: Advanced Feature Engineering
    print("STEP 2: Engineering Advanced Features for Neural Network Processing")
    print("-" * 90)
    
    engineer = EnhancedFeatureEngineer()
    transactions_featured = engineer.engineer_features(transactions)
    
    print(f"✓ Total features engineered: {transactions_featured.shape[1]}")
    print()
    
    # Step 3: Feature Selection
    print("STEP 3: Selecting Optimal Feature Set")
    print("-" * 90)
    
    feature_cols = [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'days_since_last_txn', 'timing_regularity', 'timing_mean', 'timing_std',
        'beneficiary_txn_count', 'shop_txn_count', 'unique_shops_used',
        'total_quantity_kg', 'total_commodities',
        'rice_ratio', 'wheat_ratio', 'sugar_ratio', 'kerosene_ratio',
        'quantity_mean', 'quantity_std', 'quantity_deviation',
        'auth_strength', 'log_distance', 'is_distant_shop', 'is_very_distant',
        'is_unusual_hour', 'is_weekend', 'is_night',
        'value_per_kg', 'log_total_value', 'normalized_shop_risk',
        'risk_distance_interaction', 'auth_distance_interaction',
        'quantity_regularity_interaction'
    ]
    
    # Filter to available features
    feature_cols = [f for f in feature_cols if f in transactions_featured.columns]
    
    X = transactions_featured[feature_cols].fillna(0)
    y = transactions_featured['is_fraud']
    
    print(f"✓ Selected features: {len(feature_cols)}")
    print(f"✓ Feature set: {', '.join(feature_cols[:5])}... (showing first 5)")
    print()
    
    # Step 4: Train-Test Split
    print("STEP 4: Preparing Training, Validation, and Test Datasets")
    print("-" * 90)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Training set: {len(X_train):,} transactions")
    print(f"  - Normal: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  - Fraud: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"✓ Validation set: {len(X_val):,} transactions")
    print(f"✓ Test set: {len(X_test):,} transactions")
    print()
    
    # Step 5: Train Autoencoder
    print("STEP 5: Training Deep Learning Autoencoder Model")
    print("-" * 90)
    
    detector = AutoencoderAnomalyDetector(
        encoding_dim=4,
        hidden_layers=[32, 16, 8],
        learning_rate=0.001,
        batch_size=128,
        epochs=100
    )
    
    detector.train(X_train, X_val, y_train)
    print()
    
    # Step 6: Comprehensive Evaluation
    print("STEP 6: Comprehensive Model Performance Evaluation")
    print("-" * 90)
    
    metrics = detector.evaluate(X_test, y_test)
    
    print("\n╔" + "═" * 88 + "╗")
    print("║" + " " * 25 + "FINAL PERFORMANCE METRICS" + " " * 37 + "║")
    print("╠" + "═" * 88 + "╣")
    print(f"║  Accuracy:  {metrics['accuracy']:.6f} ({metrics['accuracy']*100:6.3f}%)  {'✓ EXCELLENT' if metrics['accuracy'] >= 0.98 else '⚠ NEEDS IMPROVEMENT':>45} ║")
    print(f"║  Precision: {metrics['precision']:.6f} ({metrics['precision']*100:6.3f}%)  {'✓ EXCELLENT' if metrics['precision'] >= 0.88 else '⚠ NEEDS IMPROVEMENT':>45} ║")
    print(f"║  Recall:    {metrics['recall']:.6f} ({metrics['recall']*100:6.3f}%)  {'✓ EXCELLENT' if metrics['recall'] >= 0.85 else '⚠ NEEDS IMPROVEMENT':>45} ║")
    print(f"║  F1-Score:  {metrics['f1_score']:.6f} ({metrics['f1_score']*100:6.3f}%)  {'✓ EXCELLENT' if metrics['f1_score'] >= 0.95 else '⚠ NEEDS IMPROVEMENT':>45} ║")
    print(f"║  ROC-AUC:   {metrics['roc_auc']:.6f} ({metrics['roc_auc']*100:6.3f}%)  {'✓ EXCELLENT' if metrics['roc_auc'] >= 0.95 else '⚠ NEEDS IMPROVEMENT':>45} ║")
    print("╚" + "═" * 88 + "╝")
    print()
    
    cm = metrics['confusion_matrix']
    print("CONFUSION MATRIX ANALYSIS:")
    print(f"  True Negatives (Correct Normal):   {cm[0][0]:>8,}")
    print(f"  False Positives (False Alarms):    {cm[0][1]:>8,}")
    print(f"  False Negatives (Missed Fraud):    {cm[1][0]:>8,}")
    print(f"  True Positives (Detected Fraud):   {cm[1][1]:>8,}")
    print()
    
    print("DETAILED CLASSIFICATION PERFORMANCE:")
    class_report = metrics['classification_report']
    print(f"  Normal Transactions:")
    print(f"    Precision: {class_report['Normal']['precision']:.4f} | Recall: {class_report['Normal']['recall']:.4f} | F1: {class_report['Normal']['f1-score']:.4f}")
    print(f"  Fraudulent Transactions:")
    print(f"    Precision: {class_report['Fraud']['precision']:.4f} | Recall: {class_report['Fraud']['recall']:.4f} | F1: {class_report['Fraud']['f1-score']:.4f}")
    print()
    
    # Check targets
    targets_met = (
        metrics['accuracy'] >= 0.98 and
        metrics['f1_score'] >= 0.95 and
        metrics['precision'] >= 0.88
    )
    
    if targets_met:
        print("╔" + "═" * 88 + "╗")
        print("║" + " " * 15 + "🎯 ALL PERFORMANCE TARGETS ACHIEVED! 🎯" + " " * 32 + "║")
        print("║" + " " * 10 + "Model exceeds 98% accuracy and 95% F1-score requirements" + " " * 20 + "║")
        print("╚" + "═" * 88 + "╝")
    else:
        print("⚠ Some metrics below target - Consider hyperparameter optimization")
    print()
    
    # Step 7: Visualizations
    print("STEP 7: Generating Comprehensive Evaluation Visualizations")
    print("-" * 90)
    
    detector.plot_evaluation(X_test, y_test)
    print()
    
    # Step 8: Save Model
    print("STEP 8: Saving Trained Model and Artifacts")
    print("-" * 90)
    
    detector.save_model('grainsecure_autoencoder')
    
    # Save predictions
    predictions, reconstruction_errors = detector.predict(X_test)
    results_df = pd.DataFrame({
        'transaction_id': transactions_featured.iloc[X_test.index]['transaction_id'],
        'true_label': y_test.values,
        'predicted_label': predictions,
        'reconstruction_error': reconstruction_errors,
        'is_correct': (y_test.values == predictions).astype(int)
    })
    
    results_df.to_csv('autoencoder_predictions.csv', index=False)
    print("✓ Predictions saved to 'autoencoder_predictions.csv'")
    print()
    
    print("=" * 90)
    print("AUTOENCODER MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print()
    print("Model Deliverables:")
    print("  1. grainsecure_autoencoder_autoencoder.h5 - Full autoencoder model")
    print("  2. grainsecure_autoencoder_encoder.h5 - Encoder for feature compression")
    print("  3. grainsecure_autoencoder_package.pkl - Complete model package")
    print("  4. autoencoder_evaluation.png - Performance visualizations")
    print("  5. autoencoder_predictions.csv - Sample predictions")
    print()
    print("Next Steps:")
    print("  • Deploy to production inference API")
    print("  • Implement ensemble with Isolation Forest")
    print("  • Setup continuous monitoring pipeline")
    print("  • Begin gradient boosting classifier training")
    print()


if __name__ == "__main__":
    main()
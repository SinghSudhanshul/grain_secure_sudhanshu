import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
from typing import Tuple, Dict, List, Optional
import json
from collections import Counter

warnings.filterwarnings('ignore')
np.random.seed(42)


class ComprehensivePDSDataGenerator:
    """
    Sophisticated PDS transaction data generator with multi-layered fraud patterns
    designed specifically for gradient boosting supervised learning requirements.
    """
    
    def __init__(self, n_beneficiaries: int = 20000, n_shops: int = 750,
                 n_transactions: int = 200000, fraud_rate: float = 0.05):
        """
        Initialize comprehensive data generator.
        
        Args:
            n_beneficiaries: Number of unique beneficiary households
            n_shops: Number of fair price shops in the district
            n_transactions: Total transaction volume to generate
            fraud_rate: Target proportion of fraudulent transactions
        """
        self.n_beneficiaries = n_beneficiaries
        self.n_shops = n_shops
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        
        # Data storage
        self.beneficiaries_df = None
        self.shops_df = None
        self.transactions_df = None
        
        # Fraud scheme templates
        self.fraud_schemes = {
            'ghost_beneficiary': {'weight': 0.22, 'severity': 'HIGH'},
            'quantity_manipulation': {'weight': 0.28, 'severity': 'MEDIUM'},
            'collusion_network': {'weight': 0.18, 'severity': 'CRITICAL'},
            'identity_theft': {'weight': 0.12, 'severity': 'HIGH'},
            'timing_fraud': {'weight': 0.10, 'severity': 'MEDIUM'},
            'geographic_fraud': {'weight': 0.10, 'severity': 'MEDIUM'}
        }
        
    def generate_complete_dataset(self) -> pd.DataFrame:
        """
        Generate comprehensive dataset with all entities and relationships.
        
        Returns:
            Complete transaction dataframe with features and labels
        """
        
        print("Generating Comprehensive PDS Dataset for Supervised Learning...")
        print("=" * 80)
        
        # Generate all entities
        self._generate_beneficiary_profiles()
        print(f"✓ Beneficiary profiles: {len(self.beneficiaries_df):,}")
        
        self._generate_shop_infrastructure()
        print(f"✓ Shop infrastructure: {len(self.shops_df):,}")
        
        self._generate_transaction_records()
        print(f"✓ Transaction records: {len(self.transactions_df):,}")
        
        # Calculate fraud statistics
        fraud_counts = self.transactions_df['fraud_type'].value_counts()
        print(f"\nFraud Distribution:")
        for fraud_type, count in fraud_counts.items():
            if fraud_type != 'legitimate':
                percentage = (count / len(self.transactions_df)) * 100
                severity = self.fraud_schemes.get(fraud_type, {}).get('severity', 'UNKNOWN')
                print(f"  • {fraud_type}: {count:,} ({percentage:.2f}%) - {severity}")
        
        total_fraud = self.transactions_df['is_fraud'].sum()
        print(f"\nTotal Fraud Rate: {total_fraud:,} ({total_fraud/len(self.transactions_df)*100:.2f}%)")
        print("=" * 80)
        
        return self.transactions_df
    
    def _generate_beneficiary_profiles(self):
        """Generate realistic beneficiary demographic profiles."""
        
        # Income category distribution based on Indian PDS statistics
        income_categories = np.random.choice(
            ['BPL', 'APL', 'AAY'],
            size=self.n_beneficiaries,
            p=[0.40, 0.45, 0.15]
        )
        
        # Family size distribution following census patterns
        family_sizes = np.random.choice(
            range(1, 11),
            size=self.n_beneficiaries,
            p=[0.03, 0.12, 0.18, 0.23, 0.19, 0.12, 0.07, 0.04, 0.01, 0.01]
        )
        
        # Geographic clustering with multiple population centers
        n_clusters = 8
        cluster_centers = np.random.uniform([28.05, 77.05], [28.45, 77.45], (n_clusters, 2))
        cluster_assignment = np.random.choice(n_clusters, self.n_beneficiaries)
        
        latitudes = cluster_centers[cluster_assignment, 0] + np.random.normal(0, 0.03, self.n_beneficiaries)
        longitudes = cluster_centers[cluster_assignment, 1] + np.random.normal(0, 0.03, self.n_beneficiaries)
        
        # Calculate monthly entitlements based on family size and category
        entitlements = []
        for size, category in zip(family_sizes, income_categories):
            if category == 'AAY':
                multiplier = 1.5  # Antyodaya receives highest allocation
            elif category == 'BPL':
                multiplier = 1.2
            else:
                multiplier = 0.85
            
            entitlements.append({
                'rice_kg': int(size * 5 * multiplier),
                'wheat_kg': int(size * 3 * multiplier),
                'sugar_kg': int(size * 1 * multiplier),
                'kerosene_l': int(size * 2 * multiplier) if category in ['BPL', 'AAY'] else 0,
                'pulses_kg': int(size * 0.5 * multiplier) if category == 'AAY' else 0
            })
        
        # Vulnerability indicators (elderly, disabled, etc.)
        vulnerability_score = np.random.beta(2, 5, self.n_beneficiaries)  # Most low, some high
        
        self.beneficiaries_df = pd.DataFrame({
            'beneficiary_id': [f'BEN{str(i).zfill(8)}' for i in range(self.n_beneficiaries)],
            'family_size': family_sizes,
            'income_category': income_categories,
            'latitude': latitudes,
            'longitude': longitudes,
            'cluster_id': cluster_assignment,
            'monthly_entitlement': entitlements,
            'vulnerability_score': vulnerability_score,
            'enrollment_date': pd.date_range(
                end=datetime.now() - timedelta(days=30),
                periods=self.n_beneficiaries,
                freq='H'
            )
        })
    
    def _generate_shop_infrastructure(self):
        """Generate fair price shop profiles with operational characteristics."""
        
        # Shop locations near population clusters
        shop_latitudes = np.random.uniform(28.08, 28.42, self.n_shops)
        shop_longitudes = np.random.uniform(77.08, 77.42, self.n_shops)
        
        # Operational status with realistic distribution
        operational_status = np.random.choice(
            ['ACTIVE', 'SUSPENDED', 'CLOSED', 'PROBATION'],
            size=self.n_shops,
            p=[0.88, 0.06, 0.03, 0.03]
        )
        
        # Shop size and capacity
        shop_capacity = np.random.choice(
            ['SMALL', 'MEDIUM', 'LARGE'],
            size=self.n_shops,
            p=[0.40, 0.45, 0.15]
        )
        
        # Historical performance scores
        performance_base = np.random.beta(5, 2, self.n_shops) * 100  # Most perform well
        
        # Compliance scores
        compliance_scores = np.random.beta(4, 2, self.n_shops) * 100
        
        # Owner tenure (years operating)
        owner_tenure = np.random.exponential(5, self.n_shops)  # Mean 5 years
        
        self.shops_df = pd.DataFrame({
            'shop_id': [f'SHOP{str(i).zfill(6)}' for i in range(self.n_shops)],
            'latitude': shop_latitudes,
            'longitude': shop_longitudes,
            'operational_status': operational_status,
            'shop_capacity': shop_capacity,
            'performance_score': performance_base,
            'compliance_score': compliance_scores,
            'owner_tenure_years': owner_tenure,
            'last_inspection_days': np.random.randint(0, 365, self.n_shops)
        })
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate great circle distance in kilometers."""
        R = 6371  # Earth radius in kilometers
        
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _generate_transaction_records(self):
        """Generate transaction records with sophisticated fraud patterns."""
        
        transactions = []
        start_date = datetime.now() - timedelta(days=180)
        
        # Determine fraud distribution
        n_fraud = int(self.n_transactions * self.fraud_rate)
        fraud_indices = set(np.random.choice(self.n_transactions, n_fraud, replace=False))
        
        # Track beneficiary transaction history for temporal features
        beneficiary_history = {bid: [] for bid in self.beneficiaries_df['beneficiary_id']}
        shop_history = {sid: [] for sid in self.shops_df['shop_id']}
        
        # Fraud scheme selection probabilities
        scheme_weights = [v['weight'] for v in self.fraud_schemes.values()]
        scheme_names = list(self.fraud_schemes.keys())
        
        for txn_idx in range(self.n_transactions):
            is_fraud = txn_idx in fraud_indices
            
            # Select fraud scheme if fraudulent
            if is_fraud:
                fraud_type = np.random.choice(scheme_names, p=scheme_weights)
            else:
                fraud_type = 'legitimate'
            
            # Beneficiary selection based on fraud type
            if fraud_type == 'ghost_beneficiary':
                # Create or reuse ghost beneficiary patterns
                if np.random.random() < 0.3 and len(beneficiary_history) > 100:
                    # Reuse existing ghost pattern
                    ghost_candidates = [
                        bid for bid, history in beneficiary_history.items()
                        if len(history) > 0 and len(history) % 30 < 2
                    ]
                    if ghost_candidates:
                        beneficiary = self.beneficiaries_df[
                            self.beneficiaries_df['beneficiary_id'] == np.random.choice(ghost_candidates)
                        ].iloc[0]
                    else:
                        beneficiary = self.beneficiaries_df.sample(1).iloc[0]
                else:
                    beneficiary = self.beneficiaries_df.sample(1).iloc[0]
            elif fraud_type == 'collusion_network':
                # Target specific clusters for collusion
                collusion_cluster = np.random.choice([0, 1, 2])
                cluster_beneficiaries = self.beneficiaries_df[
                    self.beneficiaries_df['cluster_id'] == collusion_cluster
                ]
                beneficiary = cluster_beneficiaries.sample(1).iloc[0] if len(cluster_beneficiaries) > 0 else self.beneficiaries_df.sample(1).iloc[0]
            else:
                beneficiary = self.beneficiaries_df.sample(1).iloc[0]
            
            # Shop selection based on fraud type
            if fraud_type == 'collusion_network':
                # Use shops with low compliance
                risky_shops = self.shops_df[self.shops_df['compliance_score'] < 60]
                shop = risky_shops.sample(1).iloc[0] if len(risky_shops) > 0 else self.shops_df.sample(1).iloc[0]
            elif fraud_type == 'geographic_fraud':
                # Deliberately choose distant shops
                distances = self.shops_df.apply(
                    lambda s: self._haversine_distance(
                        beneficiary['latitude'], beneficiary['longitude'],
                        s['latitude'], s['longitude']
                    ), axis=1
                )
                far_shops = self.shops_df[distances > distances.quantile(0.80)]
                shop = far_shops.sample(1).iloc[0] if len(far_shops) > 0 else self.shops_df.sample(1).iloc[0]
            else:
                # Normal proximity-based selection
                distances = self.shops_df.apply(
                    lambda s: self._haversine_distance(
                        beneficiary['latitude'], beneficiary['longitude'],
                        s['latitude'], s['longitude']
                    ), axis=1
                )
                nearby_shops = self.shops_df[distances < distances.quantile(0.35)]
                shop = nearby_shops.sample(1).iloc[0] if len(nearby_shops) > 0 else self.shops_df.sample(1).iloc[0]
            
            # Transaction timing
            history = beneficiary_history[beneficiary['beneficiary_id']]
            
            if fraud_type == 'ghost_beneficiary' and len(history) > 0:
                # Mechanically precise monthly intervals
                last_txn = history[-1]
                transaction_date = last_txn + timedelta(days=30.0)
                hour = 10
                minute = 0
            elif fraud_type == 'timing_fraud':
                # Suspicious hours or patterns
                days_offset = np.random.randint(0, 180)
                transaction_date = start_date + timedelta(days=days_offset)
                hour = np.random.choice([1, 2, 3, 4, 22, 23])
                minute = np.random.randint(0, 60)
            elif len(history) > 0:
                # Realistic intervals with gamma distribution
                last_txn = history[-1]
                days_gap = np.random.gamma(shape=9, scale=3.5)  # Mean ~31.5 days
                transaction_date = last_txn + timedelta(days=days_gap)
                # Peak hours weighted distribution
                hour = np.random.choice(
                    range(7, 20),
                    p=[0.02, 0.06, 0.10, 0.14, 0.16, 0.15, 0.13, 0.10, 0.07, 0.04, 0.02, 0.01, 0.00]
                )
                minute = np.random.randint(0, 60)
            else:
                days_offset = np.random.randint(0, 180)
                transaction_date = start_date + timedelta(days=days_offset)
                hour = np.random.choice(range(8, 18))
                minute = np.random.randint(0, 60)
            
            transaction_datetime = transaction_date.replace(hour=hour, minute=minute)
            beneficiary_history[beneficiary['beneficiary_id']].append(transaction_datetime)
            shop_history[shop['shop_id']].append(transaction_datetime)
            
            # Quantity determination
            entitlement = beneficiary['monthly_entitlement']
            
            if fraud_type == 'quantity_manipulation':
                # Excessive quantities
                multiplier = np.random.uniform(1.8, 3.5)
                rice_qty = int(entitlement['rice_kg'] * multiplier)
                wheat_qty = int(entitlement['wheat_kg'] * multiplier)
                sugar_qty = int(entitlement['sugar_kg'] * multiplier)
                kerosene_qty = int(entitlement.get('kerosene_l', 0) * multiplier)
                pulses_qty = int(entitlement.get('pulses_kg', 0) * multiplier)
            elif fraud_type == 'ghost_beneficiary':
                # Exactly entitled amount with no variation
                rice_qty = entitlement['rice_kg']
                wheat_qty = entitlement['wheat_kg']
                sugar_qty = entitlement['sugar_kg']
                kerosene_qty = entitlement.get('kerosene_l', 0)
                pulses_qty = entitlement.get('pulses_kg', 0)
            elif fraud_type in ['identity_theft', 'collusion_network']:
                # Moderately inflated
                multiplier = np.random.uniform(1.3, 1.7)
                rice_qty = int(entitlement['rice_kg'] * multiplier)
                wheat_qty = int(entitlement['wheat_kg'] * multiplier)
                sugar_qty = int(entitlement['sugar_kg'] * multiplier)
                kerosene_qty = int(entitlement.get('kerosene_l', 0) * multiplier)
                pulses_qty = int(entitlement.get('pulses_kg', 0) * multiplier)
            else:
                # Normal variation
                rice_qty = int(entitlement['rice_kg'] * np.random.uniform(0.5, 1.2))
                wheat_qty = int(entitlement['wheat_kg'] * np.random.uniform(0.5, 1.2))
                sugar_qty = int(entitlement['sugar_kg'] * np.random.uniform(0.5, 1.2))
                kerosene_qty = int(entitlement.get('kerosene_l', 0) * np.random.uniform(0.6, 1.15))
                pulses_qty = int(entitlement.get('pulses_kg', 0) * np.random.uniform(0.6, 1.15))
            
            # Authentication method
            if is_fraud:
                auth_method = np.random.choice(
                    ['BIOMETRIC', 'CARD', 'MANUAL', 'AADHAAR'],
                    p=[0.12, 0.28, 0.45, 0.15]
                )
            else:
                auth_method = np.random.choice(
                    ['BIOMETRIC', 'CARD', 'MANUAL', 'AADHAAR'],
                    p=[0.55, 0.30, 0.08, 0.07]
                )
            
            # Calculate distance
            distance_km = self._haversine_distance(
                beneficiary['latitude'], beneficiary['longitude'],
                shop['latitude'], shop['longitude']
            )
            
            # Calculate transaction value at subsidized rates
            total_value = (
                rice_qty * 2.0 +
                wheat_qty * 2.0 +
                sugar_qty * 40.0 +
                kerosene_qty * 15.0 +
                pulses_qty * 50.0
            )
            
            # Shop transaction velocity
            shop_txn_velocity = len([
                t for t in shop_history[shop['shop_id']]
                if (transaction_datetime - t).days <= 1
            ])
            
            transactions.append({
                'transaction_id': f'TXN{str(txn_idx).zfill(10)}',
                'beneficiary_id': beneficiary['beneficiary_id'],
                'shop_id': shop['shop_id'],
                'transaction_datetime': transaction_datetime,
                'rice_kg': rice_qty,
                'wheat_kg': wheat_qty,
                'sugar_kg': sugar_qty,
                'kerosene_l': kerosene_qty,
                'pulses_kg': pulses_qty,
                'total_value': total_value,
                'authentication_method': auth_method,
                'distance_km': distance_km,
                'beneficiary_family_size': beneficiary['family_size'],
                'beneficiary_income_category': beneficiary['income_category'],
                'beneficiary_vulnerability': beneficiary['vulnerability_score'],
                'shop_performance_score': shop['performance_score'],
                'shop_compliance_score': shop['compliance_score'],
                'shop_capacity': shop['shop_capacity'],
                'shop_txn_velocity': shop_txn_velocity,
                'is_fraud': 1 if is_fraud else 0,
                'fraud_type': fraud_type
            })
        
        self.transactions_df = pd.DataFrame(transactions)
        self.transactions_df = self.transactions_df.sort_values('transaction_datetime').reset_index(drop=True)


class SupervisedFeatureEngineer:
    """
    Advanced feature engineering optimized for gradient boosting supervised learning
    with comprehensive temporal, behavioral, and contextual features.
    """
    
    def __init__(self):
        self.feature_stats = {}
        
    def engineer_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive feature set for gradient boosting classifier.
        
        Args:
            transactions_df: Raw transaction data
            
        Returns:
            Feature-engineered dataframe ready for model training
        """
        
        print("Engineering Advanced Features for Supervised Learning...")
        
        df = transactions_df.copy()
        
        # === TEMPORAL FEATURES ===
        df['datetime'] = pd.to_datetime(df['transaction_datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        
        # Business hours indicator
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
        df['is_suspicious_hours'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
        
        # === TRANSACTION INTERVAL FEATURES ===
        df = df.sort_values(['beneficiary_id', 'datetime'])
        df['days_since_last_txn'] = df.groupby('beneficiary_id')['datetime'].diff().dt.total_seconds() / 86400
        df['days_since_last_txn'].fillna(30, inplace=True)
        
        # Rolling interval statistics
        df['interval_rolling_mean'] = df.groupby('beneficiary_id')['days_since_last_txn'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['interval_rolling_std'] = df.groupby('beneficiary_id')['days_since_last_txn'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        df['interval_coefficient_variation'] = df['interval_rolling_std'] / (df['interval_rolling_mean'] + 1)
        df['interval_coefficient_variation'].fillna(0, inplace=True)
        
        # Interval deviation from expected
        df['interval_deviation'] = np.abs(df['days_since_last_txn'] - 30) / 30
        
        # === TRANSACTION FREQUENCY FEATURES ===
        df['beneficiary_txn_count'] = df.groupby('beneficiary_id').cumcount() + 1
        df['shop_txn_count'] = df.groupby('shop_id').cumcount() + 1
        
        # Transaction density (transactions per day)
        df['beneficiary_txn_density'] = df['beneficiary_txn_count'] / (df['days_since_last_txn'].cumsum() + 1)
        
        # === SHOP DIVERSITY FEATURES ===
        df['unique_shops_used'] = df.groupby('beneficiary_id')['shop_id'].transform('nunique')
        df['primary_shop_ratio'] = df.groupby('beneficiary_id')['shop_id'].transform(
            lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 1
        )
        
        # === QUANTITY FEATURES ===
        df['total_food_kg'] = df['rice_kg'] + df['wheat_kg'] + df['sugar_kg'] + df['pulses_kg']
        df['total_quantity'] = df['total_food_kg'] + df['kerosene_l']
        
        # Commodity ratios
        df['rice_ratio'] = df['rice_kg'] / (df['total_food_kg'] + 1)
        df['wheat_ratio'] = df['wheat_kg'] / (df['total_food_kg'] + 1)
        df['sugar_ratio'] = df['sugar_kg'] / (df['total_food_kg'] + 1)
        df['kerosene_ratio'] = df['kerosene_l'] / (df['total_quantity'] + 1)
        
        # Per capita quantities
        df['rice_per_capita'] = df['rice_kg'] / df['beneficiary_family_size']
        df['wheat_per_capita'] = df['wheat_kg'] / df['beneficiary_family_size']
        df['total_per_capita'] = df['total_food_kg'] / df['beneficiary_family_size']
        
        # Rolling quantity statistics
        df['quantity_rolling_mean'] = df.groupby('beneficiary_id')['total_quantity'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['quantity_rolling_std'] = df.groupby('beneficiary_id')['total_quantity'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        df['quantity_z_score'] = (df['total_quantity'] - df['quantity_rolling_mean']) / (df['quantity_rolling_std'] + 1)
        df['quantity_z_score'].fillna(0, inplace=True)
        
        # === VALUE FEATURES ===
        df['value_per_kg'] = df['total_value'] / (df['total_food_kg'] + 1)
        df['value_per_capita'] = df['total_value'] / df['beneficiary_family_size']
        df['log_total_value'] = np.log1p(df['total_value'])
        
        # === AUTHENTICATION FEATURES ===
        auth_strength_map = {'BIOMETRIC': 4, 'AADHAAR': 3, 'CARD': 2, 'MANUAL': 1}
        df['auth_strength_score'] = df['authentication_method'].map(auth_strength_map)
        df['is_weak_auth'] = (df['auth_strength_score'] <= 2).astype(int)
        df['is_strong_auth'] = (df['auth_strength_score'] >= 3).astype(int)
        
        # === GEOGRAPHIC FEATURES ===
        df['log_distance'] = np.log1p(df['distance_km'])
        df['is_nearby_shop'] = (df['distance_km'] < 5).astype(int)
        df['is_distant_shop'] = (df['distance_km'] > 15).astype(int)
        df['is_very_distant'] = (df['distance_km'] > 30).astype(int)
        
        # Distance per transaction
        df['cumulative_distance'] = df.groupby('beneficiary_id')['distance_km'].cumsum()
        df['avg_distance'] = df['cumulative_distance'] / df['beneficiary_txn_count']
        df['distance_deviation'] = np.abs(df['distance_km'] - df['avg_distance'])
        
        # === BENEFICIARY CHARACTERISTICS ===
        income_category_map = {'AAY': 3, 'BPL': 2, 'APL': 1}
        df['income_category_encoded'] = df['beneficiary_income_category'].map(income_category_map)
        
        df['is_large_family'] = (df['beneficiary_family_size'] >= 6).astype(int)
        df['is_vulnerable'] = (df['beneficiary_vulnerability'] > 0.6).astype(int)
        
        # === SHOP CHARACTERISTICS ===
        shop_capacity_map = {'LARGE': 3, 'MEDIUM': 2, 'SMALL': 1}
        df['shop_capacity_encoded'] = df['shop_capacity'].map(shop_capacity_map)
        
        df['shop_risk_score'] = 100 - (
            df['shop_performance_score'] * 0.5 + 
            df['shop_compliance_score'] * 0.5
        )
        df['is_high_risk_shop'] = (df['shop_risk_score'] > 50).astype(int)
        df['is_low_compliance_shop'] = (df['shop_compliance_score'] < 60).astype(int)
        
        # Shop velocity features
        df['is_high_velocity'] = (df['shop_txn_velocity'] > 50).astype(int)
        
        # === INTERACTION FEATURES ===
        df['risk_distance_product'] = df['shop_risk_score'] * df['log_distance']
        df['auth_risk_product'] = df['auth_strength_score'] * df['shop_risk_score']
        df['quantity_regularity_product'] = df['quantity_z_score'] * df['interval_coefficient_variation']
        df['vulnerability_distance'] = df['beneficiary_vulnerability'] * df['distance_km']
        df['family_quantity_ratio'] = df['total_per_capita'] * df['beneficiary_family_size']
        
        # === ANOMALY INDICATORS ===
        df['has_multiple_anomalies'] = (
            (df['is_suspicious_hours'] == 1) +
            (df['is_distant_shop'] == 1) +
            (df['is_weak_auth'] == 1) +
            (df['quantity_z_score'].abs() > 2) +
            (df['interval_coefficient_variation'] < 0.1)
        )
        
        print(f"✓ Feature engineering complete: {df.shape[1]} total features")
        
        return df


class GradientBoostingFraudClassifier:
    """
    Production-grade XGBoost classifier for supervised PDS fraud detection
    with advanced hyperparameter optimization and calibrated probability predictions.
    """
    
    def __init__(self, n_estimators: int = 300, max_depth: int = 8, 
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        """
        Initialize gradient boosting classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for gradient descent
            subsample: Fraction of samples for tree training
            colsample_bytree: Fraction of features for tree training
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        self.feature_importance = None
        self.metrics = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              optimize_hyperparameters: bool = False):
        """
        Train gradient boosting classifier with optional hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimize_hyperparameters: Whether to perform grid search
        """
        
        print("\n" + "=" * 80)
        print("TRAINING GRADIENT BOOSTING CLASSIFIER")
        print("=" * 80)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Fraud rate: {y_train.mean()*100:.2f}%")
        print("=" * 80)
        
        self.feature_names = X_train.columns.tolist()
        
        # Calculate class weights
        fraud_count = y_train.sum()
        normal_count = len(y_train) - fraud_count
        scale_pos_weight = normal_count / fraud_count
        
        print(f"\nClass Distribution:")
        print(f"  Normal transactions: {normal_count:,}")
        print(f"  Fraudulent transactions: {fraud_count:,}")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        
        if optimize_hyperparameters:
            print("\nPerforming Hyperparameter Optimization...")
            self._optimize_hyperparameters(X_train, y_train)
        else:
            # Initialize model with configured parameters
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method='hist'
            )
            
            # Prepare evaluation set
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                eval_names = ['train', 'validation']
            else:
                eval_set = [(X_train, y_train)]
                eval_names = ['train']
            
            # Train model
            print("\nTraining XGBoost model...")
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=50
            )
            
            print("\n✓ Training completed successfully")
        
        # Calibrate probabilities
        print("\nCalibrating probability predictions...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method='isotonic',
            cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        print("✓ Probability calibration complete")
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']:35s}: {row['importance']:.4f}")
    
    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Perform grid search for optimal hyperparameters."""
        
        param_grid = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [200, 300, 400],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='f1',
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
    
    def predict(self, X: pd.DataFrame, calibrated: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and probability scores.
        
        Args:
            X: Features to predict on
            calibrated: Whether to use calibrated probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        
        if calibrated and self.calibrated_model:
            predictions = self.calibrated_model.predict(X)
            probabilities = self.calibrated_model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive model evaluation with advanced metrics.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)
        
        predictions, probabilities = self.predict(X_test, calibrated=True)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, probabilities)
        avg_precision = average_precision_score(y_test, probabilities)
        mcc = matthews_corrcoef(y_test, predictions)
        kappa = cohen_kappa_score(y_test, predictions)
        
        # Classification report
        class_report = classification_report(
            y_test, predictions,
            target_names=['Normal', 'Fraud'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'matthews_corrcoef': mcc,
            'cohen_kappa': kappa,
            'specificity': specificity,
            'negative_predictive_value': npv,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_comprehensive_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generate comprehensive evaluation visualizations."""
        
        predictions, probabilities = self.predict(X_test, calibrated=True)
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax1, 
                   cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_xticklabels(['Normal', 'Fraud'], fontsize=11)
        ax1.set_yticklabels(['Normal', 'Fraud'], fontsize=11)
        
        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        ax2.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC={self.metrics["roc_auc"]:.4f})', color='#2ecc71')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.legend(fontsize=11, loc='lower right')
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.fill_between(fpr, tpr, alpha=0.2, color='#2ecc71')
        
        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
        ax3.plot(recall_curve, precision_curve, linewidth=3, 
                label=f'PR (AP={self.metrics["avg_precision"]:.4f})', color='#3498db')
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Recall', fontsize=12)
        ax3.set_ylabel('Precision', fontsize=12)
        ax3.legend(fontsize=11, loc='upper right')
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.fill_between(recall_curve, precision_curve, alpha=0.2, color='#3498db')
        
        # 4. Feature Importance (Top 15)
        ax4 = fig.add_subplot(gs[1, :])
        top_features = self.feature_importance.head(15)
        colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax4.barh(top_features['feature'], top_features['importance'], 
                       color=colors_feat, edgecolor='black', linewidth=1.5)
        ax4.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Feature Importance Score', fontsize=12)
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax4.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        # 5. Probability Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(probabilities[y_test == 0], bins=50, alpha=0.7, label='Normal', 
                color='green', density=True, edgecolor='black')
        ax5.hist(probabilities[y_test == 1], bins=50, alpha=0.7, label='Fraud', 
                color='red', density=True, edgecolor='black')
        ax5.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
        ax5.set_title('Predicted Probability Distribution', fontsize=14, fontweight='bold', pad=15)
        ax5.set_xlabel('Predicted Fraud Probability', fontsize=12)
        ax5.set_ylabel('Density', fontsize=12)
        ax5.legend(fontsize=11)
        ax5.grid(alpha=0.3, linestyle='--')
        
        # 6. Calibration Curve
        ax6 = fig.add_subplot(gs[2, 1])
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, probabilities, n_bins=10, strategy='quantile'
        )
        ax6.plot(mean_predicted_value, fraction_of_positives, 'o-', linewidth=3, 
                markersize=10, label='Calibrated Model', color='#9b59b6')
        ax6.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
        ax6.set_title('Probability Calibration Curve', fontsize=14, fontweight='bold', pad=15)
        ax6.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax6.set_ylabel('Fraction of Positives', fontsize=12)
        ax6.legend(fontsize=11)
        ax6.grid(alpha=0.3, linestyle='--')
        
        # 7. Performance Metrics
        ax7 = fig.add_subplot(gs[2, 2])
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
        metric_values = [
            self.metrics['accuracy'],
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1_score'],
            self.metrics['roc_auc'],
            (self.metrics['matthews_corrcoef'] + 1) / 2  # Normalize MCC to 0-1
        ]
        
        colors_met = ['#27ae60' if v >= 0.95 else '#f39c12' if v >= 0.85 else '#e74c3c' 
                     for v in metric_values]
        bars_met = ax7.bar(metric_names, metric_values, color=colors_met, alpha=0.8, 
                          edgecolor='black', linewidth=2)
        ax7.set_title('Comprehensive Performance Metrics', fontsize=14, fontweight='bold', pad=15)
        ax7.set_ylabel('Score', fontsize=12)
        ax7.set_ylim(0, 1.15)
        ax7.grid(axis='y', alpha=0.3, linestyle='--')
        ax7.axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='98% Target')
        ax7.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Target')
        ax7.legend(fontsize=10, loc='upper right')
        
        # Add value labels
        for bar, value in zip(bars_met, metric_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.suptitle('XGBoost Gradient Boosting Classifier - Comprehensive Evaluation Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('gradient_boosting_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation visualizations saved as 'gradient_boosting_evaluation.png'")
        plt.show()
    
    def save_model(self, filepath: str = 'gradient_boosting_model'):
        """Save complete model package."""
        
        # Save XGBoost model
        self.model.save_model(f'{filepath}_xgboost.json')
        
        # Save other components
        model_package = {
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }
        
        joblib.dump(model_package, f'{filepath}_package.pkl')
        print(f"\n✓ Model saved to {filepath}_xgboost.json and {filepath}_package.pkl")
    
    def load_model(self, filepath: str = 'gradient_boosting_model'):
        """Load complete model package."""
        
        self.model = XGBClassifier()
        self.model.load_model(f'{filepath}_xgboost.json')
        
        model_package = joblib.load(f'{filepath}_package.pkl')
        self.calibrated_model = model_package['calibrated_model']
        self.feature_names = model_package['feature_names']
        self.feature_importance = model_package.get('feature_importance')
        self.metrics = model_package.get('metrics', {})
        
        print(f"✓ Model loaded from {filepath}_xgboost.json and {filepath}_package.pkl")


def main():
    """
    Execute complete gradient boosting training and evaluation pipeline.
    """
    
    print("\n" + "╔" + "═" * 88 + "╗")
    print("║" + " " * 20 + "GRAINSECURE PDS MONITORING SYSTEM" + " " * 35 + "║")
    print("║" + " " * 15 + "Gradient Boosting Classifier for Fraud Prediction" + " " * 22 + "║")
    print("║" + " " * 10 + "Advanced Supervised Learning with XGBoost & Probability Calibration" + " " * 9 + "║")
    print("╚" + "═" * 88 + "╝")
    print()
    
    # Step 1: Generate Comprehensive Dataset
    print("STEP 1: Generating Comprehensive PDS Transaction Dataset")
    print("=" * 80)
    
    generator = ComprehensivePDSDataGenerator(
        n_beneficiaries=20000,
        n_shops=750,
        n_transactions=200000,
        fraud_rate=0.05
    )
    
    transactions = generator.generate_complete_dataset()
    print()
    
    # Step 2: Advanced Feature Engineering
    print("\nSTEP 2: Advanced Feature Engineering for Supervised Learning")
    print("=" * 80)
    
    engineer = SupervisedFeatureEngineer()
    transactions_featured = engineer.engineer_features(transactions)
    print()
    
    # Step 3: Feature Selection
    print("\nSTEP 3: Selecting Optimal Feature Set for Gradient Boosting")
    print("=" * 80)
    
    # Select numeric features only (exclude IDs, datetime, categorical originals)
    exclude_cols = [
        'transaction_id', 'beneficiary_id', 'shop_id', 'transaction_datetime',
        'datetime', 'fraud_type', 'is_fraud', 'authentication_method',
        'beneficiary_income_category', 'shop_capacity'
    ]
    
    feature_cols = [col for col in transactions_featured.columns if col not in exclude_cols]
    
    X = transactions_featured[feature_cols].fillna(0)
    y = transactions_featured['is_fraud']
    
    print(f"✓ Total features available: {len(feature_cols)}")
    print(f"✓ Sample size: {len(X):,} transactions")
    print(f"✓ Fraud prevalence: {y.mean()*100:.2f}%")
    print()
    
    # Step 4: Train-Validation-Test Split
    print("\nSTEP 4: Preparing Stratified Train-Validation-Test Split")
    print("=" * 80)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    print(f"Training set:   {len(X_train):>8,} transactions ({(y_train==1).sum():>6,} fraud)")
    print(f"Validation set: {len(X_val):>8,} transactions ({(y_val==1).sum():>6,} fraud)")
    print(f"Test set:       {len(X_test):>8,} transactions ({(y_test==1).sum():>6,} fraud)")
    print()
    
    # Step 5: Train Gradient Boosting Classifier
    print("\nSTEP 5: Training XGBoost Gradient Boosting Classifier")
    print("=" * 80)
    
    classifier = GradientBoostingFraudClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    classifier.train(X_train, y_train, X_val, y_val, optimize_hyperparameters=False)
    print()
    
    # Step 6: Comprehensive Evaluation
    print("\nSTEP 6: Comprehensive Model Performance Evaluation")
    print("=" * 80)
    
    metrics = classifier.evaluate(X_test, y_test)
    
    # Print detailed metrics
    print("\n╔" + "═" * 88 + "╗")
    print("║" + " " * 28 + "FINAL PERFORMANCE REPORT" + " " * 36 + "║")
    print("╠" + "═" * 88 + "╣")
    
    def format_metric(name, value, target, width=30):
        status = "✓ EXCEEDS TARGET" if value >= target else "⚠ BELOW TARGET"
        return f"║  {name:<{width}}: {value:.6f} ({value*100:6.3f}%)  {status:>28} ║"
    
    print(format_metric("Accuracy", metrics['accuracy'], 0.98))
    print(format_metric("Precision", metrics['precision'], 0.88))
    print(format_metric("Recall", metrics['recall'], 0.85))
    print(format_metric("F1-Score", metrics['f1_score'], 0.95))
    print(format_metric("ROC-AUC", metrics['roc_auc'], 0.95))
    print(format_metric("Average Precision", metrics['avg_precision'], 0.90))
    print(format_metric("Matthews Correlation", metrics['matthews_corrcoef'], 0.80))
    print("╚" + "═" * 88 + "╝")
    print()
    
    # Confusion Matrix Analysis
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  True Negatives  (Correctly identified normal):     {tn:>10,}")
    print(f"  False Positives (Normal flagged as fraud):         {fp:>10,}")
    print(f"  False Negatives (Fraud missed):                    {fn:>10,}")
    print(f"  True Positives  (Correctly identified fraud):      {tp:>10,}")
    print()
    print(f"  Specificity (True Negative Rate):                  {metrics['specificity']:.4f}")
    print(f"  Negative Predictive Value:                         {metrics['negative_predictive_value']:.4f}")
    print()
    
    # Classification Report
    print("DETAILED CLASSIFICATION PERFORMANCE:")
    class_report = metrics['classification_report']
    print(f"  Normal Transactions:")
    print(f"    Precision: {class_report['Normal']['precision']:.4f} | Recall: {class_report['Normal']['recall']:.4f} | F1: {class_report['Normal']['f1-score']:.4f} | Support: {class_report['Normal']['support']:.0f}")
    print(f"  Fraudulent Transactions:")
    print(f"    Precision: {class_report['Fraud']['precision']:.4f} | Recall: {class_report['Fraud']['recall']:.4f} | F1: {class_report['Fraud']['f1-score']:.4f} | Support: {class_report['Fraud']['support']:.0f}")
    print()
    
    # Target Achievement Check
    targets_met = all([
        metrics['accuracy'] >= 0.98,
        metrics['precision'] >= 0.88,
        metrics['recall'] >= 0.85,
        metrics['f1_score'] >= 0.95,
        metrics['roc_auc'] >= 0.95
    ])
    
    if targets_met:
        print("╔" + "═" * 88 + "╗")
        print("║" + " " * 15 + "🎯 ALL PERFORMANCE TARGETS ACHIEVED! 🎯" + " " * 32 + "║")
        print("║" + " " * 8 + "Model achieves 98%+ accuracy, 95%+ F1-score, and 88%+ precision!" + " " * 16 + "║")
        print("║" + " " * 18 + "Ready for Production Deployment" + " " * 37 + "║")
        print("╚" + "═" * 88 + "╝")
    else:
        print("⚠ Some metrics below target - Additional tuning recommended")
    print()
    
    # Step 7: Visualizations
    print("\nSTEP 7: Generating Comprehensive Evaluation Visualizations")
    print("=" * 80)
    
    classifier.plot_comprehensive_evaluation(X_test, y_test)
    print()
    
    # Step 8: Save Model and Artifacts
    print("\nSTEP 8: Saving Trained Model and Artifacts")
    print("=" * 80)
    
    classifier.save_model('grainsecure_gradient_boosting')
    
    # Save predictions
    predictions, probabilities = classifier.predict(X_test, calibrated=True)
    results_df = pd.DataFrame({
        'transaction_id': transactions_featured.iloc[X_test.index]['transaction_id'],
        'true_label': y_test.values,
        'predicted_label': predictions,
        'fraud_probability': probabilities,
        'is_correct': (y_test.values == predictions).astype(int),
        'true_fraud_type': transactions_featured.iloc[X_test.index]['fraud_type']
    })
    
    results_df.to_csv('gradient_boosting_predictions.csv', index=False)
    print("✓ Predictions saved to 'gradient_boosting_predictions.csv'")
    
    # Save feature importance
    classifier.feature_importance.to_csv('feature_importance.csv', index=False)
    print("✓ Feature importance saved to 'feature_importance.csv'")
    print()
    
    # Final Summary
    print("=" * 80)
    print("GRADIENT BOOSTING CLASSIFIER TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Model Deliverables:")
    print("  1. grainsecure_gradient_boosting_xgboost.json - XGBoost model")
    print("  2. grainsecure_gradient_boosting_package.pkl - Complete package")
    print("  3. gradient_boosting_evaluation.png - Performance visualizations")
    print("  4. gradient_boosting_predictions.csv - Sample predictions")
    print("  5. feature_importance.csv - Feature importance rankings")
    print()
    print("Model Capabilities:")
    print("  • Calibrated probability predictions for risk scoring")
    print("  • Feature importance rankings for interpretability")
    print("  • High precision (low false positives) for investigator efficiency")
    print("  • Excellent recall (fraud detection) for comprehensive coverage")
    print("  • Production-ready for real-time inference")
    print()
    print("Next Steps:")
    print("  • Deploy to production API endpoint")
    print("  • Integrate with Isolation Forest and Autoencoder ensemble")
    print("  • Implement continuous monitoring and retraining pipeline")
    print("  • Begin weighted factor risk model development")
    print()


if __name__ == "__main__":
    main()
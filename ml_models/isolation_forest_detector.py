import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
from typing import Tuple, Dict, List
import json

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class GrainSecurePDSDetector:
    """
    Synthetic PDS transaction data generator that creates realistic
    beneficiary profiles, shop operations, and transaction patterns
    with controlled fraud injection for model training and validation.
    """
    
    def __init__(self, n_beneficiaries: int = 10000, n_shops: int = 500,
                 n_transactions: int = 100000, fraud_rate: float = 0.05):
        """
        Initialize data generator with specified parameters.
        
        Args:
            n_beneficiaries: Number of beneficiary households to generate
            n_shops: Number of fair price shops to simulate
            n_transactions: Total number of transactions to create
            fraud_rate: Percentage of transactions that should be fraudulent
        """
        self.n_beneficiaries = n_beneficiaries
        self.n_shops = n_shops
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        
        self.beneficiaries_df = None
        self.shops_df = None
        self.transactions_df = None
        
    def generate_beneficiaries(self) -> pd.DataFrame:
        """Generate beneficiary profiles with realistic demographic attributes."""
        
        # Income categories following actual PDS distribution
        income_categories = np.random.choice(
            ['BPL', 'APL', 'AAY'],
            size=self.n_beneficiaries,
            p=[0.45, 0.40, 0.15]  # Below Poverty Line, Above Poverty Line, Antyodaya
        )
        
        # Family sizes following Indian household distribution
        family_sizes = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8],
            size=self.n_beneficiaries,
            p=[0.05, 0.15, 0.20, 0.25, 0.18, 0.10, 0.05, 0.02]
        )
        
        # Geographic distribution across district (normalized coordinates)
        latitudes = np.random.uniform(28.0, 28.5, self.n_beneficiaries)
        longitudes = np.random.uniform(77.0, 77.5, self.n_beneficiaries)
        
        # Calculate monthly entitlements based on family size and category
        entitlements = []
        for size, category in zip(family_sizes, income_categories):
            base_rice = size * 5  # 5 kg per person
            base_wheat = size * 3
            base_sugar = size * 1
            
            # BPL and AAY get higher subsidies
            if category in ['BPL', 'AAY']:
                multiplier = 1.2
            else:
                multiplier = 0.8
                
            entitlements.append({
                'rice_kg': int(base_rice * multiplier),
                'wheat_kg': int(base_wheat * multiplier),
                'sugar_kg': int(base_sugar * multiplier)
            })
        
        self.beneficiaries_df = pd.DataFrame({
            'beneficiary_id': [f'BEN{str(i).zfill(6)}' for i in range(self.n_beneficiaries)],
            'family_size': family_sizes,
            'income_category': income_categories,
            'latitude': latitudes,
            'longitude': longitudes,
            'monthly_entitlement': entitlements,
            'enrollment_date': pd.date_range(
                end=datetime.now(),
                periods=self.n_beneficiaries,
                freq='D'
            )
        })
        
        return self.beneficiaries_df
    
    def generate_shops(self) -> pd.DataFrame:
      """Generate fair price shop profiles with location and capacity data."""
      
      # Shop locations clustered in population centers
      shop_latitudes = np.random.uniform(28.0, 28.5, self.n_shops)
      shop_longitudes = np.random.uniform(77.0, 77.5, self.n_shops)
      
      # Operational status (most active, some suspended)
      operational_status = np.random.choice(
        ['ACTIVE', 'SUSPENDED', 'CLOSED'],
        size=self.n_shops,
        p=[0.92, 0.05, 0.03]
      )
      
      # Monthly quotas based on shop capacity
      monthly_quotas = []
      for _ in range(self.n_shops):
        quota = {
          'rice_kg': np.random.randint(5000, 20000),
          'wheat_kg': np.random.randint(3000, 15000),
          'sugar_kg': np.random.randint(1000, 5000)
        }
        monthly_quotas.append(quota)

      today = pd.Timestamp.today()
      license_expiry = today + pd.to_timedelta(
        np.random.randint(180, 5 * 365, size=self.n_shops), unit="D"
      )

      self.shops_df = pd.DataFrame(
        {
          "shop_id": [f"SHOP{str(i).zfill(4)}" for i in range(self.n_shops)],
          "latitude": shop_latitudes,
          "longitude": shop_longitudes,
          "operational_status": operational_status,
          "monthly_quota": monthly_quotas,
          "license_expiry": license_expiry,
        }
      )
      return self.shops_df
    
    def calculate_distance(self, lat1: float, lon1: float, 
                lat2: float, lon2: float) -> float:
      """Calculate approximate distance in kilometers using Euclidean distance."""
      # Simplified distance for synthetic data
      return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # ~111 km per degree
    
    def generate_transactions(self) -> pd.DataFrame:
      """Generate synthetic PDS transactions with controlled fraud injection."""
      
      print("→ Inside generate_transactions()", flush=True)

      # Ensure beneficiaries and shops exist
      if self.beneficiaries_df is None:
        self.generate_beneficiaries()

      if self.shops_df is None:
        self.generate_shops()

      transactions = []
      start_date = datetime.now() - timedelta(days=180)  # 6 months of data

      # Determine which transactions will be fraudulent
      n_fraud = int(self.n_transactions * self.fraud_rate)
      fraud_indices = set(np.random.choice(self.n_transactions, n_fraud, replace=False))

      for i in range(self.n_transactions):

        # Progress logging every 10k transactions
        if i % 10000 == 0 and i != 0:
          print(f"  → Generated {i} transactions...", flush=True)

        is_fraud = i in fraud_indices

        # Select beneficiary
        beneficiary = self.beneficiaries_df.sample(1).iloc[0]

        # Determine fraud type
        if is_fraud:
          fraud_type = np.random.choice([
            'ghost_beneficiary',
            'quantity_manipulation',
            'distant_shop',
            'timing_anomaly'
          ])
        else:
          fraud_type = 'legitimate'

        # Select shop based on fraud type
        if fraud_type == 'distant_shop' or np.random.random() < 0.15:
          shop = self.shops_df.sample(1).iloc[0]
        else:
          distances = self.shops_df.apply(
            lambda x: self.calculate_distance(
              beneficiary['latitude'], beneficiary['longitude'],
              x['latitude'], x['longitude']
            ),
            axis=1
          )
          nearby_shops = self.shops_df[distances < distances.quantile(0.3)]
          shop = nearby_shops.sample(1).iloc[0] if len(nearby_shops) > 0 else self.shops_df.sample(1).iloc[0]

        # Transaction timing
        if fraud_type == 'ghost_beneficiary':
          days_offset = i * 30.5 / (n_fraud / 5)
          transaction_date = start_date + timedelta(days=days_offset)
          hour = 10
        elif fraud_type == 'timing_anomaly':
          days_offset = np.random.randint(0, 180)
          transaction_date = start_date + timedelta(days=days_offset)
          hour = np.random.choice([2, 3, 23])
        else:
          days_offset = np.random.randint(0, 180)
          transaction_date = start_date + timedelta(days=days_offset)
          hour = np.random.choice(
            range(8, 18),
            p=[0.05, 0.10, 0.15, 0.20, 0.15, 0.15, 0.10, 0.05, 0.03, 0.02]
          )

        transaction_datetime = transaction_date.replace(hour=hour)

        # Quantities
        entitlement = beneficiary['monthly_entitlement']

        if fraud_type == 'quantity_manipulation':
          rice_qty = int(entitlement['rice_kg'] * np.random.uniform(1.5, 2.5))
          wheat_qty = int(entitlement['wheat_kg'] * np.random.uniform(1.5, 2.5))
          sugar_qty = int(entitlement['sugar_kg'] * np.random.uniform(1.5, 2.5))
        elif fraud_type == 'ghost_beneficiary':
          rice_qty = entitlement['rice_kg']
          wheat_qty = entitlement['wheat_kg']
          sugar_qty = entitlement['sugar_kg']
        else:
          rice_qty = int(entitlement['rice_kg'] * np.random.uniform(0.7, 1.1))
          wheat_qty = int(entitlement['wheat_kg'] * np.random.uniform(0.7, 1.1))
          sugar_qty = int(entitlement['sugar_kg'] * np.random.uniform(0.7, 1.1))

        # Authentication method
        if is_fraud:
          auth_method = np.random.choice(['BIOMETRIC', 'CARD', 'MANUAL'], p=[0.20, 0.30, 0.50])
        else:
          auth_method = np.random.choice(['BIOMETRIC', 'CARD', 'MANUAL'], p=[0.60, 0.30, 0.10])

        # Distance calculation
        distance_km = self.calculate_distance(
          beneficiary['latitude'], beneficiary['longitude'],
          shop['latitude'], shop['longitude']
        )

        # Append transaction
        transactions.append({
          'transaction_id': f'TXN{str(i).zfill(8)}',
          'beneficiary_id': beneficiary['beneficiary_id'],
          'shop_id': shop['shop_id'],
          'transaction_datetime': transaction_datetime,
          'rice_kg': rice_qty,
          'wheat_kg': wheat_qty,
          'sugar_kg': sugar_qty,
          'total_value': rice_qty * 2 + wheat_qty * 2 + sugar_qty * 40,
          'authentication_method': auth_method,
          'distance_km': distance_km,
          'is_fraud': 1 if is_fraud else 0,
          'fraud_type': fraud_type
        })

      # Convert to DataFrame and sort
      self.transactions_df = pd.DataFrame(transactions)
      self.transactions_df = self.transactions_df.sort_values('transaction_datetime').reset_index(drop=True)

      return self.transactions_df

class FeatureEngineer:
    """
    Advanced feature engineering for PDS transaction anomaly detection.
    Extracts temporal, quantity, geographic, and behavioral features.
    """
    
    def __init__(self):
        self.beneficiary_stats = {}
        self.shop_stats = {}
        
    def engineer_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from raw transaction data.
        
        Args:
            transactions_df: Raw transaction dataframe
            
        Returns:
            DataFrame with engineered features
        """
        
        df = transactions_df.copy()
        
        # Temporal features
        df['hour'] = pd.to_datetime(df['transaction_datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['transaction_datetime']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['transaction_datetime']).dt.day
        
        # Calculate days since last transaction per beneficiary
        df = df.sort_values('transaction_datetime')
        df['days_since_last_txn'] = df.groupby('beneficiary_id')['transaction_datetime'].diff().dt.days
        df['days_since_last_txn'].fillna(30, inplace=True)  # First transaction assumption
        
        # Timing regularity (coefficient of variation in intervals)
        beneficiary_intervals = df.groupby('beneficiary_id')['days_since_last_txn'].agg(['std', 'mean'])
        beneficiary_intervals['timing_regularity'] = beneficiary_intervals['std'] / (beneficiary_intervals['mean'] + 1)
        df = df.merge(
            beneficiary_intervals[['timing_regularity']],
            left_on='beneficiary_id',
            right_index=True,
            how='left'
        )
        df['timing_regularity'].fillna(0.5, inplace=True)
        
        # Transaction count per beneficiary
        txn_counts = df.groupby('beneficiary_id').size()
        df['beneficiary_txn_count'] = df['beneficiary_id'].map(txn_counts)
        
        # Shop diversity (number of unique shops used)
        shop_diversity = df.groupby('beneficiary_id')['shop_id'].nunique()
        df['shop_diversity'] = df['beneficiary_id'].map(shop_diversity)
        
        # Quantity features (total commodities)
        df['total_quantity_kg'] = df['rice_kg'] + df['wheat_kg'] + df['sugar_kg']
        
        # Authentication method encoding
        auth_encoding = {'BIOMETRIC': 3, 'CARD': 2, 'MANUAL': 1}
        df['auth_strength'] = df['authentication_method'].map(auth_encoding)
        
        # Geographic features - distance already present
        df['is_distant_shop'] = (df['distance_km'] > 10).astype(int)
        
        # Timing anomalies
        df['is_unusual_hour'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 20 else 0)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Commodity mix features
        df['rice_ratio'] = df['rice_kg'] / (df['total_quantity_kg'] + 1)
        df['wheat_ratio'] = df['wheat_kg'] / (df['total_quantity_kg'] + 1)
        df['sugar_ratio'] = df['sugar_kg'] / (df['total_quantity_kg'] + 1)
        
        return df


class IsolationForestDetector:
    """
    Production-grade Isolation Forest implementation for PDS fraud detection.
    Optimized for high-volume transaction processing with comprehensive evaluation.
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 max_samples: int = 256, random_state: int = 42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers in dataset
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_metrics = {}
        
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for anomaly detection."""
        
        features = [
            'days_since_last_txn',
            'timing_regularity',
            'beneficiary_txn_count',
            'shop_diversity',
            'total_quantity_kg',
            'auth_strength',
            'distance_km',
            'is_distant_shop',
            'is_unusual_hour',
            'is_weekend',
            'hour',
            'day_of_week',
            'rice_ratio',
            'wheat_ratio',
            'sugar_ratio'
        ]
        
        return [f for f in features if f in df.columns]
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        Train the Isolation Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels (optional, for evaluation only)
        """
        
        print("Training Isolation Forest Anomaly Detector...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train_scaled)
        self.feature_names = X_train.columns.tolist()
        
        print("✓ Training completed successfully")
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies and generate anomaly scores.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: -1 for anomaly, 1 for normal
            anomaly_scores: Lower scores indicate higher anomaly likelihood
        """
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        return predictions, anomaly_scores
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        
        predictions, anomaly_scores = self.predict(X_test)
        
        # Convert predictions to binary (1 for anomaly, 0 for normal)
        y_pred = (predictions == -1).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC AUC using anomaly scores (negate because lower scores = anomaly)
        roc_auc = roc_auc_score(y_test, -anomaly_scores)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=['Normal', 'Fraud'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'precision': class_report['Fraud']['precision'],
            'recall': class_report['Fraud']['recall'],
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        self.training_metrics = metrics
        
        return metrics
    
    def plot_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generate comprehensive evaluation visualizations."""
        
        predictions, anomaly_scores = self.predict(X_test)
        y_pred = (predictions == -1).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_xticklabels(['Normal', 'Fraud'])
        axes[0, 0].set_yticklabels(['Normal', 'Fraud'])
        
        # Anomaly Score Distribution
        axes[0, 1].hist(-anomaly_scores[y_test == 0], bins=50, alpha=0.6, label='Normal', color='green')
        axes[0, 1].hist(-anomaly_scores[y_test == 1], bins=50, alpha=0.6, label='Fraud', color='red')
        axes[0, 1].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Anomaly Score (higher = more anomalous)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test, -anomaly_scores)
        axes[1, 0].plot(recall, precision, linewidth=2, color='blue')
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].fill_between(recall, precision, alpha=0.2)
        
        # Performance Metrics Bar Chart
        metrics = self.training_metrics
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['roc_auc']
        ]
        
        colors = ['#2ecc71' if v >= 0.90 else '#f39c12' if v >= 0.80 else '#e74c3c' for v in metric_values]
        bars = axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal line at 98% target
        axes[1, 1].axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='98% Target')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('isolation_forest_evaluation.png', dpi=300, bbox_inches='tight')
        print("✓ Evaluation plots saved as 'isolation_forest_evaluation.png'")
        plt.show()
    
    def save_model(self, filepath: str = 'isolation_forest_model.pkl'):
        """Save trained model to disk."""
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.training_metrics,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }
        
        joblib.dump(model_package, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'isolation_forest_model.pkl'):
        """Load trained model from disk."""
        
        model_package = joblib.load(filepath)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_names = model_package['feature_names']
        self.training_metrics = model_package.get('metrics', {})
        
        print(f"✓ Model loaded from {filepath}")


def main():
    """
    Main execution pipeline for Isolation Forest anomaly detection.
    Demonstrates complete workflow from data generation to model deployment.
    """
    
    print("=" * 80)
    print("GRAINSECURE PDS MONITORING SYSTEM")
    print("Isolation Forest Anomaly Detection Model")
    print("=" * 80)
    print()
    
    # Step 1: Generate synthetic PDS transaction data
    print("STEP 1: Generating Synthetic PDS Transaction Data")
    print("-" * 80)
    
    generator = GrainSecurePDSDetector(
        n_beneficiaries=10000,
        n_shops=500,
        n_transactions=100000,
        fraud_rate=0.05  # 5% fraud rate
    )
    
    beneficiaries = generator.generate_beneficiaries()
    shops = generator.generate_shops()
    transactions = generator.generate_transactions()
    
    print(f"✓ Generated {len(beneficiaries):,} beneficiary profiles")
    print(f"✓ Generated {len(shops):,} fair price shops")
    print(f"✓ Generated {len(transactions):,} transactions")
    print(f"✓ Fraud transactions: {transactions['is_fraud'].sum():,} ({transactions['is_fraud'].mean()*100:.2f}%)")
    print()
    
    # Step 2: Feature Engineering
    print("STEP 2: Engineering Features for Anomaly Detection")
    print("-" * 80)
    
    engineer = FeatureEngineer()
    transactions_featured = engineer.engineer_features(transactions)
    
    print(f"✓ Engineered {transactions_featured.shape[1]} features")
    print(f"✓ Feature engineering completed")
    print()
    
    # Step 3: Prepare training and test datasets
    print("STEP 3: Preparing Training and Test Datasets")
    print("-" * 80)
    
    detector = IsolationForestDetector(
        contamination=0.05,
        n_estimators=200,
        max_samples=256
    )
    
    feature_cols = detector.select_features(transactions_featured)
    X = transactions_featured[feature_cols]
    y = transactions_featured['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {len(X_train):,} transactions")
    print(f"✓ Test set: {len(X_test):,} transactions")
    print(f"✓ Features selected: {len(feature_cols)}")
    print(f"  Features: {', '.join(feature_cols)}")
    print()
    
    # Step 4: Train the model
    print("STEP 4: Training Isolation Forest Model")
    print("-" * 80)
    
    detector.train(X_train, y_train)
    print()
    
    # Step 5: Evaluate the model
    print("STEP 5: Evaluating Model Performance")
    print("-" * 80)
    
    metrics = detector.evaluate(X_test, y_test)
    
    print("PERFORMANCE METRICS:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    print()
    
    print("CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    print(f"  True Negatives:  {cm[0][0]:,}")
    print(f"  False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,}")
    print(f"  True Positives:  {cm[1][1]:,}")
    print()
    
    print("DETAILED CLASSIFICATION REPORT:")
    class_report = metrics['classification_report']
    print(f"  Normal Transactions:")
    print(f"    Precision: {class_report['Normal']['precision']:.4f}")
    print(f"    Recall:    {class_report['Normal']['recall']:.4f}")
    print(f"    F1-Score:  {class_report['Normal']['f1-score']:.4f}")
    print(f"  Fraudulent Transactions:")
    print(f"    Precision: {class_report['Fraud']['precision']:.4f}")
    print(f"    Recall:    {class_report['Fraud']['recall']:.4f}")
    print(f"    F1-Score:  {class_report['Fraud']['f1-score']:.4f}")
    print()
    
    # Check if targets are met
    target_accuracy = 0.98
    target_f1 = 0.95
    
    if metrics['accuracy'] >= target_accuracy and metrics['f1_score'] >= target_f1:
        print(f"✓ TARGET ACHIEVED: Model exceeds {target_accuracy*100}% accuracy and {target_f1*100}% F1-score!")
    else:
        print(f"⚠ Performance below target. Consider hyperparameter tuning.")
    print()
    
    # Step 6: Visualize results
    print("STEP 6: Generating Evaluation Visualizations")
    print("-" * 80)
    
    detector.plot_evaluation(X_test, y_test)
    print()
    
    # Step 7: Save the model
    print("STEP 7: Saving Trained Model")
    print("-" * 80)
    
    detector.save_model('grainsecure_isolation_forest.pkl')
    
    # Save sample predictions
    predictions, anomaly_scores = detector.predict(X_test)
    results_df = pd.DataFrame({
        'transaction_id': transactions_featured.iloc[X_test.index]['transaction_id'],
        'true_label': y_test.values,
        'predicted_label': (predictions == -1).astype(int),
        'anomaly_score': -anomaly_scores,  # Negate for intuitive interpretation
        'is_correct': (y_test.values == (predictions == -1).astype(int)).astype(int)
    })
    
    results_df.to_csv('sample_predictions.csv', index=False)
    print("✓ Sample predictions saved to 'sample_predictions.csv'")
    print()
    
    print("=" * 80)
    print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()


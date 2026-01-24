"""
Stock Reconciliation Model for Fraud Detection System
Detects discrepancies between inventory records and transaction flows
Uses multiple detection methods for comprehensive reconciliation analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReconciliationConfig:
    """Configuration for stock reconciliation analysis"""
    variance_threshold: float = 0.05  # 5% variance threshold
    z_score_threshold: float = 3.0
    isolation_contamination: float = 0.1
    elliptic_contamination: float = 0.1
    lookback_period_days: int = 30
    min_transaction_count: int = 5
    confidence_level: float = 0.95
    moving_average_window: int = 7


class StockReconciliationFeatures:
    """Feature engineering for stock reconciliation"""
    
    @staticmethod
    def calculate_expected_stock(transactions: pd.DataFrame, 
                                 initial_stock: float) -> pd.DataFrame:
        """
        Calculate expected stock levels based on transaction flow
        
        Args:
            transactions: Transaction data with amount and type
            initial_stock: Starting inventory level
            
        Returns:
            DataFrame with expected stock levels
        """
        transactions = transactions.sort_values('timestamp').copy()
        
        # Calculate cumulative stock changes
        transactions['stock_change'] = transactions.apply(
            lambda x: x['amount'] if x['transaction_type'] == 'receipt' 
            else -x['amount'], axis=1
        )
        
        transactions['expected_stock'] = (
            initial_stock + transactions['stock_change'].cumsum()
        )
        
        # Add rolling statistics
        transactions['rolling_mean'] = (
            transactions['expected_stock']
            .rolling(window=7, min_periods=1)
            .mean()
        )
        
        transactions['rolling_std'] = (
            transactions['expected_stock']
            .rolling(window=7, min_periods=1)
            .std()
            .fillna(0)
        )
        
        return transactions
    
    @staticmethod
    def extract_reconciliation_features(
        transactions: pd.DataFrame,
        stock_records: pd.DataFrame,
        item_id: str
    ) -> pd.DataFrame:
        """
        Extract comprehensive reconciliation features
        
        Args:
            transactions: Transaction history
            stock_records: Recorded stock levels
            item_id: Item identifier
            
        Returns:
            DataFrame with reconciliation features
        """
        # Merge transactions with stock records
        merged = pd.merge_asof(
            transactions.sort_values('timestamp'),
            stock_records.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            suffixes=('_txn', '_stock')
        )
        
        features = pd.DataFrame()
        
        # 1. Variance features
        features['absolute_variance'] = (
            merged['expected_stock'] - merged['recorded_stock']
        )
        features['relative_variance'] = (
            features['absolute_variance'] / 
            merged['expected_stock'].replace(0, 1)
        )
        features['variance_squared'] = features['absolute_variance'] ** 2
        
        # 2. Rate of change features
        features['stock_velocity'] = (
            merged['recorded_stock'].diff() / 
            merged['timestamp'].diff().dt.total_seconds().replace(0, 1)
        )
        features['expected_velocity'] = (
            merged['expected_stock'].diff() / 
            merged['timestamp'].diff().dt.total_seconds().replace(0, 1)
        )
        features['velocity_mismatch'] = (
            features['stock_velocity'] - features['expected_velocity']
        )
        
        # 3. Statistical features
        features['z_score'] = (
            (merged['recorded_stock'] - merged['expected_stock']) / 
            merged['rolling_std'].replace(0, 1)
        )
        features['deviation_from_mean'] = (
            merged['recorded_stock'] - merged['rolling_mean']
        )
        
        # 4. Transaction pattern features
        features['txn_frequency'] = (
            transactions.groupby(transactions['timestamp'].dt.date)
            .size()
            .reindex(merged['timestamp'].dt.date, fill_value=0)
            .values
        )
        features['txn_volume'] = (
            transactions.groupby(transactions['timestamp'].dt.date)
            ['amount']
            .sum()
            .reindex(merged['timestamp'].dt.date, fill_value=0)
            .values
        )
        
        # 5. Cumulative features
        features['cumulative_variance'] = features['absolute_variance'].cumsum()
        features['variance_trend'] = (
            features['absolute_variance']
            .rolling(window=5, min_periods=1)
            .mean()
        )
        
        # 6. Time-based features
        features['hour'] = merged['timestamp'].dt.hour
        features['day_of_week'] = merged['timestamp'].dt.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # 7. Consistency features
        features['consecutive_mismatches'] = (
            (features['absolute_variance'].abs() > 0)
            .groupby((features['absolute_variance'].abs() <= 0).cumsum())
            .cumsum()
        )
        
        features['timestamp'] = merged['timestamp']
        features['item_id'] = item_id
        features['expected_stock'] = merged['expected_stock']
        features['recorded_stock'] = merged['recorded_stock']
        
        return features.fillna(0)


class StockReconciliationModel:
    """
    Multi-method stock reconciliation model for fraud detection
    Combines statistical analysis, machine learning, and business rules
    """
    
    def __init__(self, config: Optional[ReconciliationConfig] = None):
        """
        Initialize the stock reconciliation model
        
        Args:
            config: Configuration parameters
        """
        self.config = config or ReconciliationConfig()
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=self.config.isolation_contamination,
            random_state=42,
            n_estimators=100
        )
        self.elliptic_envelope = EllipticEnvelope(
            contamination=self.config.elliptic_contamination,
            random_state=42
        )
        self.feature_extractor = StockReconciliationFeatures()
        self.is_fitted = False
        self.feature_importance = {}
        
        logger.info("Stock Reconciliation Model initialized")
    
    def fit(self, transactions: pd.DataFrame, 
            stock_records: pd.DataFrame,
            item_metadata: Optional[pd.DataFrame] = None) -> 'StockReconciliationModel':
        """
        Train the reconciliation model
        
        Args:
            transactions: Historical transaction data
            stock_records: Historical stock level records
            item_metadata: Optional item-level metadata
            
        Returns:
            Self for method chaining
        """
        logger.info("Training Stock Reconciliation Model...")
        
        # Validate input data
        self._validate_data(transactions, stock_records)
        
        # Extract features for all items
        all_features = []
        for item_id in transactions['item_id'].unique():
            item_txns = transactions[transactions['item_id'] == item_id]
            item_stock = stock_records[stock_records['item_id'] == item_id]
            
            if len(item_txns) >= self.config.min_transaction_count:
                # Calculate expected stock
                initial_stock = item_stock.iloc[0]['recorded_stock']
                item_txns = self.feature_extractor.calculate_expected_stock(
                    item_txns, initial_stock
                )
                
                # Extract features
                features = self.feature_extractor.extract_reconciliation_features(
                    item_txns, item_stock, item_id
                )
                all_features.append(features)
        
        if not all_features:
            raise ValueError("Insufficient data for model training")
        
        self.training_features = pd.concat(all_features, ignore_index=True)
        
        # Select numerical features for modeling
        feature_cols = [
            'absolute_variance', 'relative_variance', 'variance_squared',
            'stock_velocity', 'expected_velocity', 'velocity_mismatch',
            'z_score', 'deviation_from_mean', 'txn_frequency', 'txn_volume',
            'cumulative_variance', 'variance_trend', 'consecutive_mismatches'
        ]
        
        X = self.training_features[feature_cols].fillna(0)
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit anomaly detection models
        self.isolation_forest.fit(X_scaled)
        
        # Fit elliptic envelope on normal data (filter obvious anomalies first)
        z_scores = np.abs(X['z_score'])
        normal_mask = z_scores < self.config.z_score_threshold
        if normal_mask.sum() > 10:
            self.elliptic_envelope.fit(X_scaled[normal_mask])
        else:
            self.elliptic_envelope.fit(X_scaled)
        
        # Calculate feature importance based on variance contribution
        self._calculate_feature_importance(X, feature_cols)
        
        self.is_fitted = True
        logger.info(f"Model trained on {len(self.training_features)} records")
        
        return self
    
    def predict(self, transactions: pd.DataFrame,
                stock_records: pd.DataFrame,
                item_metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Detect stock reconciliation anomalies
        
        Args:
            transactions: Transaction data to analyze
            stock_records: Stock level records to analyze
            item_metadata: Optional item-level metadata
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info("Predicting stock reconciliation anomalies...")
        
        # Extract features for all items
        all_predictions = []
        for item_id in transactions['item_id'].unique():
            item_txns = transactions[transactions['item_id'] == item_id]
            item_stock = stock_records[stock_records['item_id'] == item_id]
            
            if len(item_txns) >= self.config.min_transaction_count:
                # Calculate expected stock
                initial_stock = item_stock.iloc[0]['recorded_stock']
                item_txns = self.feature_extractor.calculate_expected_stock(
                    item_txns, initial_stock
                )
                
                # Extract features
                features = self.feature_extractor.extract_reconciliation_features(
                    item_txns, item_stock, item_id
                )
                
                # Generate predictions
                predictions = self._predict_item(features)
                all_predictions.append(predictions)
        
        if not all_predictions:
            return pd.DataFrame()
        
        results = pd.concat(all_predictions, ignore_index=True)
        
        logger.info(f"Detected {results['is_anomaly'].sum()} anomalies "
                   f"out of {len(results)} records")
        
        return results
    
    def _predict_item(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for a single item"""
        feature_cols = [
            'absolute_variance', 'relative_variance', 'variance_squared',
            'stock_velocity', 'expected_velocity', 'velocity_mismatch',
            'z_score', 'deviation_from_mean', 'txn_frequency', 'txn_volume',
            'cumulative_variance', 'variance_trend', 'consecutive_mismatches'
        ]
        
        X = features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Method 1: Statistical threshold
        statistical_anomaly = (
            np.abs(features['relative_variance']) > self.config.variance_threshold
        ) | (
            np.abs(features['z_score']) > self.config.z_score_threshold
        )
        
        # Method 2: Isolation Forest
        iso_predictions = self.isolation_forest.predict(X_scaled)
        iso_scores = self.isolation_forest.score_samples(X_scaled)
        iso_anomaly = iso_predictions == -1
        
        # Method 3: Elliptic Envelope
        elliptic_predictions = self.elliptic_envelope.predict(X_scaled)
        elliptic_scores = self.elliptic_envelope.score_samples(X_scaled)
        elliptic_anomaly = elliptic_predictions == -1
        
        # Ensemble: Combine methods (voting)
        anomaly_votes = (
            statistical_anomaly.astype(int) +
            iso_anomaly.astype(int) +
            elliptic_anomaly.astype(int)
        )
        
        # Consider anomaly if 2 or more methods agree
        is_anomaly = anomaly_votes >= 2
        
        # Calculate composite anomaly score (0-1 scale)
        anomaly_score = (
            0.4 * np.abs(features['relative_variance']) / 
            (self.config.variance_threshold * 3) +
            0.3 * (1 - (iso_scores - iso_scores.min()) / 
                   (iso_scores.max() - iso_scores.min() + 1e-10)) +
            0.3 * (1 - (elliptic_scores - elliptic_scores.min()) / 
                   (elliptic_scores.max() - elliptic_scores.min() + 1e-10))
        )
        anomaly_score = np.clip(anomaly_score, 0, 1)
        
        # Calculate severity based on variance magnitude
        severity = pd.cut(
            np.abs(features['relative_variance']),
            bins=[0, 0.05, 0.15, 0.30, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        # Calculate confidence based on data quality
        data_quality = self._assess_data_quality(features)
        confidence = data_quality * (1 - np.abs(features['rolling_std']) / 
                                     (features['expected_stock'] + 1e-10))
        confidence = np.clip(confidence, 0, 1)
        
        # Build results DataFrame
        results = pd.DataFrame({
            'timestamp': features['timestamp'],
            'item_id': features['item_id'],
            'expected_stock': features['expected_stock'],
            'recorded_stock': features['recorded_stock'],
            'variance': features['absolute_variance'],
            'variance_pct': features['relative_variance'] * 100,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'severity': severity,
            'confidence': confidence,
            'statistical_flag': statistical_anomaly,
            'isolation_flag': iso_anomaly,
            'elliptic_flag': elliptic_anomaly,
            'z_score': features['z_score'],
            'consecutive_mismatches': features['consecutive_mismatches']
        })
        
        return results
    
    def _validate_data(self, transactions: pd.DataFrame, 
                       stock_records: pd.DataFrame):
        """Validate input data format and quality"""
        required_txn_cols = ['item_id', 'timestamp', 'amount', 'transaction_type']
        required_stock_cols = ['item_id', 'timestamp', 'recorded_stock']
        
        missing_txn = set(required_txn_cols) - set(transactions.columns)
        missing_stock = set(required_stock_cols) - set(stock_records.columns)
        
        if missing_txn:
            raise ValueError(f"Missing transaction columns: {missing_txn}")
        if missing_stock:
            raise ValueError(f"Missing stock record columns: {missing_stock}")
        
        # Ensure datetime type
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        stock_records['timestamp'] = pd.to_datetime(stock_records['timestamp'])
    
    def _calculate_feature_importance(self, X: pd.DataFrame, 
                                      feature_cols: List[str]):
        """Calculate feature importance based on variance contribution"""
        variances = X.var()
        total_variance = variances.sum()
        
        for col in feature_cols:
            self.feature_importance[col] = variances[col] / total_variance
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), 
                   key=lambda x: x[1], reverse=True)
        )
    
    def _assess_data_quality(self, features: pd.DataFrame) -> np.ndarray:
        """Assess data quality for confidence calculation"""
        quality_score = np.ones(len(features))
        
        # Penalize sparse transaction patterns
        quality_score *= np.clip(features['txn_frequency'] / 10, 0.5, 1.0)
        
        # Penalize high volatility
        volatility = features['rolling_std'] / (features['expected_stock'] + 1e-10)
        quality_score *= np.clip(1 - volatility, 0.3, 1.0)
        
        return quality_score
    
    def get_reconciliation_report(self, predictions: pd.DataFrame) -> Dict:
        """
        Generate comprehensive reconciliation report
        
        Args:
            predictions: Prediction results from predict()
            
        Returns:
            Dictionary containing reconciliation metrics and insights
        """
        if predictions.empty:
            return {"status": "no_data", "message": "No predictions available"}
        
        # Overall statistics
        total_records = len(predictions)
        total_anomalies = predictions['is_anomaly'].sum()
        anomaly_rate = total_anomalies / total_records
        
        # Severity breakdown
        severity_counts = predictions[predictions['is_anomaly']]['severity'].value_counts()
        
        # Top problematic items
        item_anomaly_rates = (
            predictions.groupby('item_id')
            .agg({
                'is_anomaly': ['sum', 'mean'],
                'anomaly_score': 'mean',
                'variance_pct': 'mean'
            })
            .round(3)
        )
        item_anomaly_rates.columns = [
            'anomaly_count', 'anomaly_rate', 'avg_score', 'avg_variance_pct'
        ]
        top_items = item_anomaly_rates.nlargest(10, 'anomaly_rate')
        
        # Temporal patterns
        predictions['date'] = predictions['timestamp'].dt.date
        daily_anomalies = (
            predictions.groupby('date')['is_anomaly']
            .agg(['sum', 'mean'])
            .round(3)
        )
        
        # Variance statistics
        variance_stats = {
            'mean_variance_pct': predictions['variance_pct'].mean(),
            'median_variance_pct': predictions['variance_pct'].median(),
            'max_variance_pct': predictions['variance_pct'].max(),
            'std_variance_pct': predictions['variance_pct'].std()
        }
        
        report = {
            'summary': {
                'total_records': int(total_records),
                'total_anomalies': int(total_anomalies),
                'anomaly_rate': float(anomaly_rate),
                'average_anomaly_score': float(predictions['anomaly_score'].mean()),
                'high_confidence_anomalies': int(
                    ((predictions['is_anomaly']) & 
                     (predictions['confidence'] > 0.8)).sum()
                )
            },
            'severity_distribution': severity_counts.to_dict(),
            'variance_statistics': variance_stats,
            'top_problematic_items': top_items.to_dict('index'),
            'daily_trends': daily_anomalies.tail(7).to_dict('index'),
            'feature_importance': self.feature_importance,
            'model_config': {
                'variance_threshold': self.config.variance_threshold,
                'z_score_threshold': self.config.z_score_threshold,
                'isolation_contamination': self.config.isolation_contamination,
                'lookback_period_days': self.config.lookback_period_days
            }
        }
        
        return report
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        import pickle
        
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'elliptic_envelope': self.elliptic_envelope,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.isolation_forest = model_data['isolation_forest']
        self.elliptic_envelope = model_data['elliptic_envelope']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")


def generate_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample transaction and stock data for testing"""
    np.random.seed(42)
    
    # Generate transactions
    n_transactions = 500
    item_ids = ['ITEM_' + str(i) for i in range(1, 6)]
    
    transactions = []
    for _ in range(n_transactions):
        transactions.append({
            'item_id': np.random.choice(item_ids),
            'timestamp': datetime.now() - timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24)
            ),
            'amount': np.random.randint(1, 20),
            'transaction_type': np.random.choice(['receipt', 'issue'], p=[0.6, 0.4])
        })
    
    transactions_df = pd.DataFrame(transactions).sort_values('timestamp')
    
    # Generate stock records with some discrepancies
    stock_records = []
    for item_id in item_ids:
        item_txns = transactions_df[transactions_df['item_id'] == item_id]
        stock = 100  # Initial stock
        
        for idx, row in item_txns.iterrows():
            # Update stock based on transaction
            if row['transaction_type'] == 'receipt':
                stock += row['amount']
            else:
                stock -= row['amount']
            
            # Introduce random discrepancies (10% chance)
            recorded_stock = stock
            if np.random.random() < 0.1:
                recorded_stock += np.random.randint(-10, 10)
            
            stock_records.append({
                'item_id': item_id,
                'timestamp': row['timestamp'],
                'recorded_stock': max(0, recorded_stock)
            })
    
    stock_records_df = pd.DataFrame(stock_records).sort_values('timestamp')
    
    return transactions_df, stock_records_df


if __name__ == "__main__":
    # Example usage and demonstration
    print("=" * 80)
    print("Stock Reconciliation Model - Demonstration")
    print("=" * 80)
    
    # Generate sample data
    print("\nGenerating sample transaction and stock data...")
    transactions, stock_records = generate_sample_data()
    print(f"Generated {len(transactions)} transactions and {len(stock_records)} stock records")
    
    # Initialize and train model
    print("\nInitializing and training model...")
    config = ReconciliationConfig(
        variance_threshold=0.05,
        z_score_threshold=3.0,
        isolation_contamination=0.1
    )
    
    model = StockReconciliationModel(config)
    model.fit(transactions, stock_records)
    
    # Make predictions
    print("\nDetecting reconciliation anomalies...")
    predictions = model.predict(transactions, stock_records)
    
    # Display results
    print(f"\nPrediction Results:")
    print(f"Total records analyzed: {len(predictions)}")
    print(f"Anomalies detected: {predictions['is_anomaly'].sum()}")
    print(f"Anomaly rate: {predictions['is_anomaly'].mean():.2%}")
    
    print("\nSample anomalies:")
    anomalies = predictions[predictions['is_anomaly']].head(5)
    print(anomalies[['item_id', 'expected_stock', 'recorded_stock', 
                     'variance_pct', 'anomaly_score', 'severity']])
    
    # Generate comprehensive report
    print("\nGenerating reconciliation report...")
    report = model.get_reconciliation_report(predictions)
    
    print("\nReconciliation Summary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    print("\nTop 3 Feature Importance:")
    for i, (feature, importance) in enumerate(list(report['feature_importance'].items())[:3]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("=" * 80)
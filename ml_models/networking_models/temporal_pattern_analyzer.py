"""
GrainSecure PDS Monitoring System
Temporal Pattern Analyzer for Time Series Fraud Detection

This module implements advanced time series analysis for detecting temporal
fraud patterns, seasonal anomalies, behavioral changes, and coordinated
timing schemes in PDS transaction data.

Author: GrainSecure Development Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import joblib
from typing import Dict, List, Tuple, Optional
import time

warnings.filterwarnings('ignore')
np.random.seed(42)


class TemporalFraudDataGenerator:
    """
    Generates time series transaction data with sophisticated temporal fraud
    patterns for model training and validation.
    """
    
    def __init__(self, n_beneficiaries: int = 1000, n_days: int = 365,
                 fraud_rate: float = 0.08, seasonal_fraud_rate: float = 0.03):
        """
        Initialize temporal fraud data generator.
        
        Args:
            n_beneficiaries: Number of beneficiary time series
            n_days: Time series length in days
            fraud_rate: Base fraud rate
            seasonal_fraud_rate: Additional fraud during peak seasons
        """
        self.n_beneficiaries = n_beneficiaries
        self.n_days = n_days
        self.fraud_rate = fraud_rate
        self.seasonal_fraud_rate = seasonal_fraud_rate
        
        self.transactions_df = None
        self.fraud_patterns = {
            'ghost_beneficiary': [],
            'seasonal_fraud': [],
            'progressive_fraud': [],
            'coordinated_timing': [],
            'change_point_fraud': []
        }
        
    def generate_temporal_data(self) -> pd.DataFrame:
        """
        Generate complete temporal transaction dataset.
        
        Returns:
            DataFrame with temporal transaction patterns
        """
        
        print("=" * 80)
        print("GENERATING TEMPORAL FRAUD PATTERNS FOR TIME SERIES ANALYSIS")
        print("=" * 80)
        
        start_date = datetime(2024, 1, 1)
        
        all_transactions = []
        
        # Determine fraud types for each beneficiary
        n_fraud = int(self.n_beneficiaries * self.fraud_rate)
        fraud_indices = np.random.choice(self.n_beneficiaries, n_fraud, replace=False)
        
        fraud_types = np.random.choice(
            ['ghost', 'seasonal', 'progressive', 'coordinated', 'change_point'],
            size=n_fraud,
            p=[0.25, 0.20, 0.20, 0.20, 0.15]
        )
        
        fraud_assignment = dict(zip(fraud_indices, fraud_types))
        
        for ben_id in range(self.n_beneficiaries):
            fraud_type = fraud_assignment.get(ben_id, 'legitimate')
            
            if fraud_type == 'ghost':
                txns = self._generate_ghost_pattern(ben_id, start_date)
            elif fraud_type == 'seasonal':
                txns = self._generate_seasonal_fraud(ben_id, start_date)
            elif fraud_type == 'progressive':
                txns = self._generate_progressive_fraud(ben_id, start_date)
            elif fraud_type == 'coordinated':
                txns = self._generate_coordinated_timing(ben_id, start_date)
            elif fraud_type == 'change_point':
                txns = self._generate_change_point_fraud(ben_id, start_date)
            else:
                txns = self._generate_legitimate_pattern(ben_id, start_date)
            
            all_transactions.extend(txns)
        
        self.transactions_df = pd.DataFrame(all_transactions)
        self.transactions_df = self.transactions_df.sort_values('date').reset_index(drop=True)
        
        print(f"\n✓ Generated {len(self.transactions_df):,} transactions")
        print(f"  Time range: {self.transactions_df['date'].min()} to {self.transactions_df['date'].max()}")
        print(f"  Fraud transactions: {self.transactions_df['is_fraud'].sum():,} ({self.transactions_df['is_fraud'].mean()*100:.1f}%)")
        
        print(f"\nFraud Pattern Distribution:")
        for pattern, count in self.transactions_df[self.transactions_df['is_fraud']==1]['fraud_type'].value_counts().items():
            print(f"  {pattern}: {count:,} transactions")
        
        print("=" * 80)
        
        return self.transactions_df
    
    def _generate_ghost_pattern(self, ben_id: int, start_date: datetime) -> List[Dict]:
        """Generate ghost beneficiary with mechanically precise timing."""
        
        transactions = []
        current_date = start_date + timedelta(days=np.random.randint(0, 30))
        
        # Ghost beneficiaries: exactly 30-day intervals
        n_transactions = int(self.n_days / 30)
        
        for i in range(n_transactions):
            transactions.append({
                'beneficiary_id': f'BEN_{ben_id:05d}',
                'date': current_date,
                'quantity': 25.0,  # Always same quantity
                'days_since_last': 30.0 if i > 0 else 30.0,
                'hour': 10,  # Always same hour
                'is_fraud': 1,
                'fraud_type': 'ghost'
            })
            current_date += timedelta(days=30)
        
        self.fraud_patterns['ghost_beneficiary'].append(ben_id)
        return transactions
    
    def _generate_seasonal_fraud(self, ben_id: int, start_date: datetime) -> List[Dict]:
        """Generate seasonal fraud pattern (higher fraud during certain months)."""
        
        transactions = []
        current_date = start_date
        
        # Higher fraud during months 6-8 (mid-year) and 12 (year-end)
        fraud_months = {6, 7, 8, 12}
        
        for day in range(0, self.n_days, 30):
            transaction_date = start_date + timedelta(days=day + np.random.randint(-3, 4))
            month = transaction_date.month
            
            is_fraud_month = month in fraud_months
            
            if is_fraud_month:
                quantity = np.random.uniform(35, 50)  # Excessive during fraud season
            else:
                quantity = np.random.uniform(18, 28)  # Normal otherwise
            
            if day > 0:
                days_since = (transaction_date - transactions[-1]['date']).days
            else:
                days_since = 30
            
            transactions.append({
                'beneficiary_id': f'BEN_{ben_id:05d}',
                'date': transaction_date,
                'quantity': quantity,
                'days_since_last': float(days_since),
                'hour': np.random.choice(range(8, 18)),
                'is_fraud': int(is_fraud_month),
                'fraud_type': 'seasonal'
            })
        
        self.fraud_patterns['seasonal_fraud'].append(ben_id)
        return transactions
    
    def _generate_progressive_fraud(self, ben_id: int, start_date: datetime) -> List[Dict]:
        """Generate progressive fraud (gradually increasing quantities)."""
        
        transactions = []
        current_date = start_date
        
        base_quantity = 20.0
        n_transactions = int(self.n_days / 30)
        
        for i in range(n_transactions):
            transaction_date = current_date + timedelta(days=np.random.randint(-3, 4))
            
            # Gradually increase quantity (fraud escalates over time)
            quantity_multiplier = 1.0 + (i / n_transactions) * 1.5  # Up to 2.5x by end
            quantity = base_quantity * quantity_multiplier
            
            if i > 0:
                days_since = (transaction_date - transactions[-1]['date']).days
            else:
                days_since = 30
            
            # Fraud starts after 3 months
            is_fraud = 1 if i >= 3 else 0
            
            transactions.append({
                'beneficiary_id': f'BEN_{ben_id:05d}',
                'date': transaction_date,
                'quantity': quantity,
                'days_since_last': float(days_since),
                'hour': np.random.choice(range(8, 18)),
                'is_fraud': is_fraud,
                'fraud_type': 'progressive'
            })
            
            current_date += timedelta(days=30)
        
        self.fraud_patterns['progressive_fraud'].append(ben_id)
        return transactions
    
    def _generate_coordinated_timing(self, ben_id: int, start_date: datetime) -> List[Dict]:
        """Generate coordinated timing pattern (synchronized with other fraudsters)."""
        
        transactions = []
        
        # Coordinated collection on specific days each month (5th, 15th, 25th)
        coordinated_days = [5, 15, 25]
        
        for month in range(12):
            for day in coordinated_days:
                try:
                    transaction_date = datetime(2024, month + 1, day)
                    if transaction_date >= start_date and transaction_date < start_date + timedelta(days=self.n_days):
                        
                        if transactions:
                            days_since = (transaction_date - transactions[-1]['date']).days
                        else:
                            days_since = 10
                        
                        transactions.append({
                            'beneficiary_id': f'BEN_{ben_id:05d}',
                            'date': transaction_date,
                            'quantity': np.random.uniform(28, 35),
                            'days_since_last': float(days_since),
                            'hour': 14,  # Always same time window
                            'is_fraud': 1,
                            'fraud_type': 'coordinated'
                        })
                except:
                    pass
        
        self.fraud_patterns['coordinated_timing'].append(ben_id)
        return transactions
    
    def _generate_change_point_fraud(self, ben_id: int, start_date: datetime) -> List[Dict]:
        """Generate change point fraud (sudden behavior change mid-series)."""
        
        transactions = []
        current_date = start_date
        
        n_transactions = int(self.n_days / 30)
        change_point = n_transactions // 2  # Change at midpoint
        
        for i in range(n_transactions):
            transaction_date = current_date + timedelta(days=np.random.randint(-3, 4))
            
            if i < change_point:
                # Before change: normal behavior
                quantity = np.random.uniform(18, 25)
                interval_variation = np.random.randint(28, 33)
                is_fraud_txn = 0
            else:
                # After change: fraudulent behavior (account takeover)
                quantity = np.random.uniform(35, 45)
                interval_variation = 30  # More regular
                is_fraud_txn = 1
            
            if i > 0:
                days_since = interval_variation
            else:
                days_since = 30
            
            transactions.append({
                'beneficiary_id': f'BEN_{ben_id:05d}',
                'date': transaction_date,
                'quantity': quantity,
                'days_since_last': float(days_since),
                'hour': np.random.choice(range(8, 18)),
                'is_fraud': is_fraud_txn,
                'fraud_type': 'change_point'
            })
            
            current_date += timedelta(days=interval_variation)
        
        self.fraud_patterns['change_point_fraud'].append(ben_id)
        return transactions
    
    def _generate_legitimate_pattern(self, ben_id: int, start_date: datetime) -> List[Dict]:
        """Generate legitimate transaction pattern with natural variation."""
        
        transactions = []
        current_date = start_date + timedelta(days=np.random.randint(0, 30))
        
        for _ in range(int(self.n_days / 30)):
            # Natural variation in intervals (28-35 days)
            interval = np.random.randint(28, 36)
            transaction_date = current_date + timedelta(days=interval)
            
            # Natural quantity variation
            quantity = np.random.uniform(15, 28)
            
            if transactions:
                days_since = (transaction_date - transactions[-1]['date']).days
            else:
                days_since = 30
            
            transactions.append({
                'beneficiary_id': f'BEN_{ben_id:05d}',
                'date': transaction_date,
                'quantity': quantity,
                'days_since_last': float(days_since),
                'hour': np.random.choice(range(7, 19)),
                'is_fraud': 0,
                'fraud_type': 'legitimate'
            })
            
            current_date = transaction_date
        
        return transactions


class TemporalPatternAnalyzer:
    """
    Comprehensive temporal pattern analyzer implementing multiple time series
    analysis techniques for fraud detection.
    """
    
    def __init__(self, transactions_df: pd.DataFrame):
        """
        Initialize temporal analyzer.
        
        Args:
            transactions_df: Transaction data with temporal information
        """
        self.df = transactions_df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        self.beneficiary_series = {}
        self.anomaly_scores = {}
        self.detected_anomalies = {}
        self.evaluation_metrics = {}
        
        self._prepare_time_series()
        
    def _prepare_time_series(self):
        """Prepare individual beneficiary time series."""
        
        print("\nPreparing individual beneficiary time series...")
        
        for ben_id in self.df['beneficiary_id'].unique():
            ben_data = self.df[self.df['beneficiary_id'] == ben_id].sort_values('date')
            
            self.beneficiary_series[ben_id] = {
                'dates': ben_data['date'].values,
                'quantities': ben_data['quantity'].values,
                'intervals': ben_data['days_since_last'].values,
                'is_fraud': ben_data['is_fraud'].values,
                'fraud_type': ben_data['fraud_type'].iloc[0]
            }
        
        print(f"✓ Prepared {len(self.beneficiary_series)} beneficiary time series")
    
    def detect_interval_regularity_anomalies(self, threshold: float = 0.15) -> Dict:
        """
        Detect anomalies based on unusual interval regularity.
        
        Args:
            threshold: CV threshold below which is considered suspicious
            
        Returns:
            Dictionary of anomaly scores
        """
        
        print("\n" + "=" * 80)
        print("DETECTING INTERVAL REGULARITY ANOMALIES")
        print("=" * 80)
        
        scores = {}
        
        for ben_id, series in self.beneficiary_series.items():
            intervals = series['intervals']
            
            if len(intervals) > 2:
                # Coefficient of variation
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                cv = std_interval / mean_interval if mean_interval > 0 else 1.0
                
                # Lower CV = more regular = more suspicious
                anomaly_score = 1.0 - min(cv / 0.5, 1.0)  # Normalize
                
                scores[ben_id] = {
                    'score': anomaly_score,
                    'is_anomaly': cv < threshold,
                    'cv': cv,
                    'mean_interval': mean_interval
                }
        
        self.anomaly_scores['interval_regularity'] = scores
        
        n_anomalies = sum(1 for s in scores.values() if s['is_anomaly'])
        print(f"✓ Detected {n_anomalies} beneficiaries with suspicious interval regularity")
        print(f"  Mean CV: {np.mean([s['cv'] for s in scores.values()]):.3f}")
        print("=" * 80)
        
        return scores
    
    def detect_quantity_trend_anomalies(self, min_slope: float = 0.5) -> Dict:
        """
        Detect progressive fraud through quantity trend analysis.
        
        Args:
            min_slope: Minimum slope to consider anomalous
            
        Returns:
            Dictionary of anomaly scores
        """
        
        print("\n" + "=" * 80)
        print("DETECTING QUANTITY TREND ANOMALIES")
        print("=" * 80)
        
        scores = {}
        
        for ben_id, series in self.beneficiary_series.items():
            quantities = series['quantities']
            
            if len(quantities) > 3:
                # Linear trend
                x = np.arange(len(quantities))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, quantities)
                
                # Positive slope indicates increasing quantities
                anomaly_score = min(max(slope, 0) / 2.0, 1.0)
                
                scores[ben_id] = {
                    'score': anomaly_score,
                    'is_anomaly': slope > min_slope and p_value < 0.05,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value
                }
        
        self.anomaly_scores['quantity_trend'] = scores
        
        n_anomalies = sum(1 for s in scores.values() if s['is_anomaly'])
        print(f"✓ Detected {n_anomalies} beneficiaries with suspicious quantity trends")
        print(f"  Mean slope: {np.mean([s['slope'] for s in scores.values()]):.3f}")
        print("=" * 80)
        
        return scores
    
    def detect_seasonal_anomalies(self) -> Dict:
        """
        Detect seasonal fraud patterns using monthly aggregation.
        
        Returns:
            Dictionary of anomaly scores
        """
        
        print("\n" + "=" * 80)
        print("DETECTING SEASONAL ANOMALIES")
        print("=" * 80)
        
        scores = {}
        
        for ben_id, series in self.beneficiary_series.items():
            dates = pd.to_datetime(series['dates'])
            quantities = series['quantities']
            
            if len(quantities) >= 6:  # Need enough data
                # Group by month
                monthly_data = defaultdict(list)
                for date, qty in zip(dates, quantities):
                    monthly_data[date.month].append(qty)
                
                # Calculate monthly averages
                monthly_avg = {month: np.mean(qtys) for month, qtys in monthly_data.items()}
                
                if len(monthly_avg) >= 3:
                    # Check for high variation across months
                    avg_values = list(monthly_avg.values())
                    overall_mean = np.mean(avg_values)
                    overall_std = np.std(avg_values)
                    cv = overall_std / overall_mean if overall_mean > 0 else 0
                    
                    # High CV indicates seasonal variation
                    anomaly_score = min(cv / 0.5, 1.0)
                    
                    scores[ben_id] = {
                        'score': anomaly_score,
                        'is_anomaly': cv > 0.30,  # 30% variation
                        'seasonal_cv': cv,
                        'monthly_avg': monthly_avg
                    }
        
        self.anomaly_scores['seasonal'] = scores
        
        n_anomalies = sum(1 for s in scores.values() if s['is_anomaly'])
        print(f"✓ Detected {n_anomalies} beneficiaries with seasonal patterns")
        print(f"  Mean seasonal CV: {np.mean([s['seasonal_cv'] for s in scores.values()]):.3f}")
        print("=" * 80)
        
        return scores
    
    def detect_change_points(self, min_change: float = 10.0) -> Dict:
        """
        Detect abrupt changes in behavior using CUSUM algorithm.
        
        Args:
            min_change: Minimum change to consider significant
            
        Returns:
            Dictionary of change point detections
        """
        
        print("\n" + "=" * 80)
        print("DETECTING CHANGE POINTS")
        print("=" * 80)
        
        scores = {}
        
        for ben_id, series in self.beneficiary_series.items():
            quantities = series['quantities']
            
            if len(quantities) > 5:
                # CUSUM algorithm
                mean_qty = np.mean(quantities[:len(quantities)//2])  # Baseline from first half
                cumsum = np.cumsum(quantities - mean_qty)
                
                # Find maximum deviation
                max_cumsum = np.max(np.abs(cumsum))
                change_idx = np.argmax(np.abs(cumsum))
                
                # Check if significant change
                is_anomaly = max_cumsum > min_change
                anomaly_score = min(max_cumsum / 50, 1.0)
                
                scores[ben_id] = {
                    'score': anomaly_score,
                    'is_anomaly': is_anomaly,
                    'change_magnitude': max_cumsum,
                    'change_index': change_idx
                }
        
        self.anomaly_scores['change_point'] = scores
        
        n_anomalies = sum(1 for s in scores.values() if s['is_anomaly'])
        print(f"✓ Detected {n_anomalies} beneficiaries with change points")
        print(f"  Mean change magnitude: {np.mean([s['change_magnitude'] for s in scores.values()]):.3f}")
        print("=" * 80)
        
        return scores
    
    def detect_coordinated_timing(self) -> Dict:
        """
        Detect coordinated timing patterns across beneficiaries.
        
        Returns:
            Dictionary of coordination scores
        """
        
        print("\n" + "=" * 80)
        print("DETECTING COORDINATED TIMING PATTERNS")
        print("=" * 80)
        
        # Aggregate all transactions by day
        daily_txns = defaultdict(set)
        
        for ben_id, series in self.beneficiary_series.items():
            for date in series['dates']:
                date_day = pd.to_datetime(date).day
                daily_txns[date_day].add(ben_id)
        
        # Find days with unusually high transaction counts
        day_counts = {day: len(bens) for day, bens in daily_txns.items()}
        mean_count = np.mean(list(day_counts.values()))
        std_count = np.std(list(day_counts.values()))
        
        high_activity_days = {
            day: count for day, count in day_counts.items()
            if count > mean_count + 2 * std_count
        }
        
        # Score beneficiaries by how often they transact on high-activity days
        scores = {}
        
        for ben_id, series in self.beneficiary_series.items():
            transaction_days = [pd.to_datetime(d).day for d in series['dates']]
            coordinated_count = sum(1 for day in transaction_days if day in high_activity_days)
            
            if len(transaction_days) > 0:
                coordination_ratio = coordinated_count / len(transaction_days)
                anomaly_score = min(coordination_ratio * 2, 1.0)
                
                scores[ben_id] = {
                    'score': anomaly_score,
                    'is_anomaly': coordination_ratio > 0.5,
                    'coordination_ratio': coordination_ratio,
                    'coordinated_count': coordinated_count
                }
        
        self.anomaly_scores['coordinated_timing'] = scores
        
        n_anomalies = sum(1 for s in scores.values() if s['is_anomaly'])
        print(f"✓ Detected {n_anomalies} beneficiaries with coordinated timing")
        print(f"  High-activity days: {len(high_activity_days)}")
        print("=" * 80)
        
        return scores
    
    def create_composite_temporal_score(self, weights: Dict[str, float] = None) -> Dict:
        """
        Create composite score from all temporal anomaly detections.
        
        Args:
            weights: Dictionary of detection method weights
            
        Returns:
            Dictionary of composite scores
        """
        
        if weights is None:
            weights = {
                'interval_regularity': 0.25,
                'quantity_trend': 0.25,
                'seasonal': 0.20,
                'change_point': 0.20,
                'coordinated_timing': 0.10
            }
        
        composite_scores = {}
        
        all_ben_ids = set()
        for scores_dict in self.anomaly_scores.values():
            all_ben_ids.update(scores_dict.keys())
        
        for ben_id in all_ben_ids:
            weighted_score = 0.0
            
            for method, weight in weights.items():
                if method in self.anomaly_scores:
                    ben_score = self.anomaly_scores[method].get(ben_id, {})
                    weighted_score += ben_score.get('score', 0.0) * weight
            
            composite_scores[ben_id] = weighted_score
        
        self.anomaly_scores['composite'] = {
            ben_id: {'score': score, 'is_anomaly': score > 0.6}
            for ben_id, score in composite_scores.items()
        }
        
        print(f"\n✓ Composite temporal scores created")
        print(f"  Mean composite score: {np.mean(list(composite_scores.values())):.3f}")
        
        return composite_scores
    
    def evaluate_detection_performance(self, method: str = 'composite') -> Dict:
        """
        Evaluate detection performance against ground truth.
        
        Args:
            method: Which detection method to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        print(f"\n{'='*80}")
        print(f"EVALUATING {method.upper()} DETECTION PERFORMANCE")
        print(f"{'='*80}")
        
        scores_dict = self.anomaly_scores.get(method, {})
        
        y_true = []
        y_pred = []
        y_scores = []
        
        for ben_id in self.beneficiary_series.keys():
            # Ground truth: any fraud in this beneficiary's series
            true_label = 1 if np.any(self.beneficiary_series[ben_id]['is_fraud']) else 0
            
            # Prediction
            ben_score = scores_dict.get(ben_id, {})
            predicted_label = 1 if ben_score.get('is_anomaly', False) else 0
            score_value = ben_score.get('score', 0.0)
            
            y_true.append(true_label)
            y_pred.append(predicted_label)
            y_scores.append(score_value)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        cm = confusion_matrix(y_true, y_pred)
        
        class_report = classification_report(
            y_true, y_pred,
            target_names=['Legitimate', 'Fraud'],
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        self.evaluation_metrics[method] = metrics
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]:>4}  FP: {cm[0][1]:>4}")
        print(f"  FN: {cm[1][0]:>4}  TP: {cm[1][1]:>4}")
        
        print(f"{'='*80}")
        
        return metrics
    
    def visualize_temporal_patterns(self, figsize: Tuple[int, int] = (20, 16)):
        """Generate comprehensive temporal pattern visualizations."""
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Transaction volume over time
        ax1 = axes[0, 0]
        daily_counts = self.df.groupby('date').size()
        ax1.plot(daily_counts.index, daily_counts.values, linewidth=1.5)
        ax1.set_title('Daily Transaction Volume', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Transactions')
        ax1.grid(alpha=0.3)
        
        # 2. Interval regularity distribution
        ax2 = axes[0, 1]
        if 'interval_regularity' in self.anomaly_scores:
            cvs = [s['cv'] for s in self.anomaly_scores['interval_regularity'].values()]
            ax2.hist(cvs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.axvline(0.15, color='red', linestyle='--', linewidth=2, label='Threshold')
            ax2.set_title('Interval Regularity Distribution (CV)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Coefficient of Variation')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. Quantity trends
        ax3 = axes[1, 0]
        if 'quantity_trend' in self.anomaly_scores:
            slopes = [s['slope'] for s in self.anomaly_scores['quantity_trend'].values()]
            ax3.hist(slopes, bins=50, color='coral', alpha=0.7, edgecolor='black')
            ax3.axvline(0, color='black', linestyle='-', linewidth=1)
            ax3.set_title('Quantity Trend Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Slope (quantity change per transaction)')
            ax3.set_ylabel('Frequency')
            ax3.grid(alpha=0.3)
        
        # 4. Seasonal pattern
        ax4 = axes[1, 1]
        monthly_fraud = self.df[self.df['is_fraud']==1].groupby(
            pd.to_datetime(self.df[self.df['is_fraud']==1]['date']).dt.month
        ).size()
        monthly_total = self.df.groupby(pd.to_datetime(self.df['date']).dt.month).size()
        fraud_rate_by_month = (monthly_fraud / monthly_total * 100).fillna(0)
        
        ax4.bar(fraud_rate_by_month.index, fraud_rate_by_month.values, 
               color='darkred', alpha=0.7, edgecolor='black')
        ax4.set_title('Fraud Rate by Month', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Fraud Rate (%)')
        ax4.set_xticks(range(1, 13))
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Example time series
        ax5 = axes[2, 0]
        # Find one of each fraud type
        examples = {}
        for ben_id, series in self.beneficiary_series.items():
            fraud_type = series['fraud_type']
            if fraud_type not in examples and fraud_type != 'legitimate':
                examples[fraud_type] = ben_id
            if len(examples) >= 3:
                break
        
        colors = {'ghost': 'red', 'progressive': 'orange', 'seasonal': 'purple'}
        for fraud_type, ben_id in examples.items():
            series = self.beneficiary_series[ben_id]
            ax5.plot(range(len(series['quantities'])), series['quantities'], 
                    marker='o', label=fraud_type, color=colors.get(fraud_type, 'blue'))
        
        ax5.set_title('Example Fraud Patterns', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Transaction Number')
        ax5.set_ylabel('Quantity')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Performance comparison
        ax6 = axes[2, 1]
        if self.evaluation_metrics:
            methods = list(self.evaluation_metrics.keys())
            metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
            metric_labels = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            data = []
            for method in methods:
                row = [self.evaluation_metrics[method].get(m, 0) for m in metrics_to_plot]
                data.append(row)
            
            if data:
                sns.heatmap(
                    np.array(data).T,
                    xticklabels=methods,
                    yticklabels=metric_labels,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=1,
                    ax=ax6,
                    cbar_kws={'label': 'Score'}
                )
                ax6.set_title('Detection Method Performance', fontsize=14, fontweight='bold')
        
        plt.suptitle('Temporal Pattern Analysis - Comprehensive View', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig('temporal_pattern_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as 'temporal_pattern_analysis.png'")
        plt.show()
    
    def save_results(self, filepath: str = 'temporal_analysis_results.pkl'):
        """Save all analysis results."""
        
        results = {
            'anomaly_scores': self.anomaly_scores,
            'evaluation_metrics': self.evaluation_metrics
        }
        
        joblib.dump(results, filepath)
        print(f"\n✓ Results saved to {filepath}")


def main():
    """
    Execute complete temporal pattern analysis pipeline.
    """
    
    print("\n" + "╔" + "═"*88 + "╗")
    print("║" + " "*20 + "GRAINSECURE PDS MONITORING SYSTEM" + " "*35 + "║")
    print("║" + " "*15 + "Temporal Pattern Analyzer for Fraud Detection" + " "*26 + "║")
    print("║" + " "*10 + "Advanced Time Series Analysis and Behavioral Change Detection" + " "*15 + "║")
    print("╚" + "═"*88 + "╝")
    print()
    
    # Step 1: Generate Temporal Data
    print("STEP 1: Generating Temporal Transaction Data with Fraud Patterns")
    print("="*80)
    
    generator = TemporalFraudDataGenerator(
        n_beneficiaries=1000,
        n_days=365,
        fraud_rate=0.08,
        seasonal_fraud_rate=0.03
    )
    
    transactions_df = generator.generate_temporal_data()
    print()
    
    # Step 2: Initialize Analyzer
    print("\nSTEP 2: Initializing Temporal Pattern Analyzer")
    print("="*80)
    
    analyzer = TemporalPatternAnalyzer(transactions_df)
    print()
    
    # Step 3: Run All Detection Methods
    print("\nSTEP 3: Running Temporal Anomaly Detection Algorithms")
    
    analyzer.detect_interval_regularity_anomalies()
    analyzer.detect_quantity_trend_anomalies()
    analyzer.detect_seasonal_anomalies()
    analyzer.detect_change_points()
    analyzer.detect_coordinated_timing()
    
    # Create composite score
    analyzer.create_composite_temporal_score()
    print()
    
    # Step 4: Evaluate All Methods
    print("\nSTEP 4: Evaluating Detection Performance")
    
    methods = ['interval_regularity', 'quantity_trend', 'seasonal', 
               'change_point', 'coordinated_timing', 'composite']
    
    for method in methods:
        analyzer.evaluate_detection_performance(method)
    
    print()
    
    # Step 5: Generate Report
    print("\nSTEP 5: Generating Comprehensive Performance Report")
    print("="*80)
    
    print(f"\n{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for method in methods:
        if method in analyzer.evaluation_metrics:
            m = analyzer.evaluation_metrics[method]
            print(f"{method:<25} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
                  f"{m['recall']:<12.4f} {m['f1_score']:<12.4f}")
    
    # Find best method
    best_method = max(
        methods,
        key=lambda m: analyzer.evaluation_metrics[m]['f1_score']
    )
    
    best_metrics = analyzer.evaluation_metrics[best_method]
    
    print("\n" + "="*80)
    print("PERFORMANCE TARGET VALIDATION")
    print("="*80)
    
    targets_met = (
        best_metrics['f1_score'] >= 0.95 and
        best_metrics['precision'] >= 0.88 and
        best_metrics['recall'] >= 0.85 and
        best_metrics['accuracy'] >= 0.98
    )
    
    print(f"\nBest Method: {best_method.upper()}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f} ({'✓ EXCEEDS' if best_metrics['accuracy'] >= 0.98 else '✗ BELOW'} 0.98 target)")
    print(f"  Precision: {best_metrics['precision']:.4f} ({'✓ EXCEEDS' if best_metrics['precision'] >= 0.88 else '✗ BELOW'} 0.88 target)")
    print(f"  Recall:    {best_metrics['recall']:.4f} ({'✓ EXCEEDS' if best_metrics['recall'] >= 0.85 else '✗ BELOW'} 0.85 target)")
    print(f"  F1-Score:  {best_metrics['f1_score']:.4f} ({'✓ EXCEEDS' if best_metrics['f1_score'] >= 0.95 else '✗ BELOW'} 0.95 target)")
    
    if targets_met:
        print("\n╔" + "═"*88 + "╗")
        print("║" + " "*15 + "🎯 ALL PERFORMANCE TARGETS ACHIEVED! 🎯" + " "*32 + "║")
        print("║" + " "*18 + "Ready for Production Deployment" + " "*37 + "║")
        print("╚" + "═"*88 + "╝")
    
    print()
    
    # Step 6: Visualizations
    print("\nSTEP 6: Generating Comprehensive Visualizations")
    print("="*80)
    
    analyzer.visualize_temporal_patterns()
    print()
    
    # Step 7: Save Results
    print("\nSTEP 7: Saving Analysis Results")
    print("="*80)
    
    analyzer.save_results('grainsecure_temporal_analysis.pkl')
    
    # Save detection results
    results_df = []
    for ben_id in analyzer.beneficiary_series.keys():
        composite_score = analyzer.anomaly_scores['composite'].get(ben_id, {})
        is_fraud = np.any(analyzer.beneficiary_series[ben_id]['is_fraud'])
        fraud_type = analyzer.beneficiary_series[ben_id]['fraud_type']
        
        results_df.append({
            'beneficiary_id': ben_id,
            'true_fraud': is_fraud,
            'fraud_type': fraud_type,
            'predicted_fraud': composite_score.get('is_anomaly', False),
            'anomaly_score': composite_score.get('score', 0.0)
        })
    
    results_df = pd.DataFrame(results_df)
    results_df.to_csv('temporal_detection_results.csv', index=False)
    print("✓ Detection results saved to 'temporal_detection_results.csv'")
    print()
    
    print("="*80)
    print("TEMPORAL PATTERN ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print()
    print("Deliverables:")
    print("  1. grainsecure_temporal_analysis.pkl - Complete analysis results")
    print("  2. temporal_detection_results.csv - Beneficiary-level fraud predictions")
    print("  3. temporal_pattern_analysis.png - Six-panel visualization")
    print()
    print("Key Findings:")
    print(f"  • Best detection method: {best_method}")
    print(f"  • Overall F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"  • Detected {sum(results_df['predicted_fraud'])} fraudulent beneficiaries")
    print(f"  • False positive rate: {(1-best_metrics['precision'])*100:.1f}%")
    print()
    print("Fraud Patterns Identified:")
    for pattern, count in results_df[results_df['true_fraud']==1]['fraud_type'].value_counts().items():
        print(f"  • {pattern}: {count} cases")
    print()
    print("Next Steps:")
    print("  • Deploy to production temporal monitoring")
    print("  • Integrate with real-time transaction streams")
    print("  • Implement automated alerting for temporal anomalies")
    print("  • Begin ensemble predictor development")
    print()


if __name__ == "__main__":
    main()
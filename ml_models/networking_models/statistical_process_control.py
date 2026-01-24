"""
Statistical Process Control (SPC) Module for Fraud Detection System
Implements control charts and aggregate metric monitoring for detecting
systematic anomalies and process deviations in transaction patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ControlChartType(Enum):
    """Types of control charts available"""
    XBAR = "x_bar"  # Mean chart
    R_CHART = "r_chart"  # Range chart
    S_CHART = "s_chart"  # Standard deviation chart
    EWMA = "ewma"  # Exponentially Weighted Moving Average
    CUSUM = "cusum"  # Cumulative Sum
    P_CHART = "p_chart"  # Proportion chart
    C_CHART = "c_chart"  # Count chart
    INDIVIDUALS = "individuals"  # Individual values chart


class ViolationType(Enum):
    """Types of control limit violations"""
    BEYOND_LIMITS = "beyond_control_limits"
    TWO_OF_THREE = "two_of_three_beyond_2sigma"
    FOUR_OF_FIVE = "four_of_five_beyond_1sigma"
    EIGHT_CONSECUTIVE = "eight_consecutive_same_side"
    SIX_TRENDING = "six_consecutive_trending"
    FOURTEEN_ALTERNATING = "fourteen_alternating"
    FIFTEEN_WITHIN_1SIGMA = "fifteen_within_1sigma"
    EIGHT_BEYOND_1SIGMA = "eight_beyond_1sigma"


@dataclass
class SPCConfig:
    """Configuration for Statistical Process Control"""
    # Control limit parameters
    sigma_multiplier: float = 3.0  # Standard deviation multiplier for control limits
    ewma_lambda: float = 0.2  # EWMA smoothing parameter
    cusum_h: float = 5.0  # CUSUM decision interval
    cusum_k: float = 0.5  # CUSUM reference value (shift detection)
    
    # Process parameters
    subgroup_size: int = 5  # Size of subgroups for control charts
    min_baseline_points: int = 20  # Minimum points for establishing control limits
    baseline_period_days: int = 30  # Days to use for baseline calculation
    
    # Sensitivity parameters
    enable_western_electric: bool = True  # Enable Western Electric rules
    enable_nelson_rules: bool = True  # Enable Nelson rules
    detect_trends: bool = True  # Enable trend detection
    detect_shifts: bool = True  # Enable shift detection
    
    # Thresholds
    trend_min_points: int = 6  # Minimum points for trend detection
    shift_min_points: int = 8  # Minimum points for shift detection
    outlier_threshold: float = 3.0  # Z-score threshold for outliers
    
    # Update parameters
    recalculate_limits_frequency: int = 100  # Recalculate limits every N points
    adaptive_limits: bool = True  # Enable adaptive control limits


@dataclass
class ControlLimits:
    """Container for control chart limits"""
    center_line: float
    upper_control_limit: float
    lower_control_limit: float
    upper_warning_limit: float
    lower_warning_limit: float
    sigma: float
    
    def is_beyond_control(self, value: float) -> bool:
        """Check if value exceeds control limits"""
        return value > self.upper_control_limit or value < self.lower_control_limit
    
    def is_beyond_warning(self, value: float) -> bool:
        """Check if value exceeds warning limits"""
        return value > self.upper_warning_limit or value < self.lower_warning_limit


class AggregateMetricCalculator:
    """Calculate various aggregate metrics from transaction data"""
    
    @staticmethod
    def calculate_volume_metrics(data: pd.DataFrame, 
                                 time_period: str = 'D') -> pd.DataFrame:
        """
        Calculate transaction volume metrics over time periods
        
        Args:
            data: Transaction data with timestamp
            time_period: Pandas frequency string (D=day, H=hour, W=week)
            
        Returns:
            DataFrame with volume metrics
        """
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['period'] = data['timestamp'].dt.floor(time_period)
        
        metrics = data.groupby('period').agg({
            'transaction_id': 'count',  # Transaction count
            'amount': ['sum', 'mean', 'std', 'min', 'max']
        }).reset_index()
        
        metrics.columns = ['period', 'count', 'total_amount', 'mean_amount', 
                          'std_amount', 'min_amount', 'max_amount']
        
        # Fill missing periods with zeros
        metrics = metrics.set_index('period').asfreq(time_period, fill_value=0).reset_index()
        
        return metrics
    
    @staticmethod
    def calculate_pattern_metrics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern-based metrics (hourly, daily, weekly patterns)
        
        Args:
            data: Transaction data with timestamp
            
        Returns:
            DataFrame with pattern metrics
        """
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['date'] = data['timestamp'].dt.date
        
        # Calculate hourly distribution
        hourly_dist = data.groupby(['date', 'hour']).size().reset_index(name='count')
        hourly_metrics = hourly_dist.groupby('date').agg({
            'count': ['mean', 'std', 'max', 'min']
        }).reset_index()
        hourly_metrics.columns = ['date', 'hourly_mean', 'hourly_std', 
                                  'hourly_max', 'hourly_min']
        
        # Calculate day-of-week distribution
        dow_dist = data.groupby(['date', 'day_of_week']).size().reset_index(name='count')
        dow_metrics = dow_dist.groupby('date')['count'].agg(['mean', 'std']).reset_index()
        dow_metrics.columns = ['date', 'dow_mean', 'dow_std']
        
        # Merge metrics
        pattern_metrics = pd.merge(hourly_metrics, dow_metrics, on='date', how='outer')
        pattern_metrics['date'] = pd.to_datetime(pattern_metrics['date'])
        
        return pattern_metrics
    
    @staticmethod
    def calculate_user_metrics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate user-level aggregate metrics
        
        Args:
            data: Transaction data with user_id and timestamp
            
        Returns:
            DataFrame with user metrics aggregated by time period
        """
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['date'] = data['timestamp'].dt.date
        
        user_metrics = data.groupby('date').agg({
            'user_id': lambda x: x.nunique(),  # Unique users
            'transaction_id': 'count',  # Total transactions
            'amount': ['mean', 'sum']
        }).reset_index()
        
        user_metrics.columns = ['date', 'unique_users', 'total_transactions', 
                               'avg_amount_per_txn', 'total_amount']
        
        # Calculate transactions per user
        user_metrics['txn_per_user'] = (
            user_metrics['total_transactions'] / user_metrics['unique_users']
        )
        
        # Calculate amount per user
        user_metrics['amount_per_user'] = (
            user_metrics['total_amount'] / user_metrics['unique_users']
        )
        
        user_metrics['date'] = pd.to_datetime(user_metrics['date'])
        
        return user_metrics


class ControlChart:
    """Base class for control charts with common functionality"""
    
    def __init__(self, chart_type: ControlChartType, config: SPCConfig):
        """
        Initialize control chart
        
        Args:
            chart_type: Type of control chart
            config: SPC configuration
        """
        self.chart_type = chart_type
        self.config = config
        self.limits: Optional[ControlLimits] = None
        self.baseline_data: List[float] = []
        self.is_calibrated = False
        
    def calibrate(self, baseline_data: np.ndarray):
        """
        Establish control limits from baseline data
        
        Args:
            baseline_data: Historical data for establishing limits
        """
        if len(baseline_data) < self.config.min_baseline_points:
            raise ValueError(
                f"Insufficient baseline data. Need at least "
                f"{self.config.min_baseline_points} points"
            )
        
        self.baseline_data = baseline_data.copy()
        self._calculate_control_limits(baseline_data)
        self.is_calibrated = True
        
        logger.info(f"{self.chart_type.value} chart calibrated with "
                   f"{len(baseline_data)} baseline points")
    
    def _calculate_control_limits(self, data: np.ndarray):
        """Calculate control limits (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _calculate_control_limits")
    
    def monitor(self, value: float) -> Tuple[bool, Dict]:
        """
        Monitor a single value against control limits
        
        Args:
            value: Value to monitor
            
        Returns:
            Tuple of (is_anomaly, details_dict)
        """
        if not self.is_calibrated:
            raise ValueError("Chart must be calibrated before monitoring")
        
        is_beyond_control = self.limits.is_beyond_control(value)
        is_beyond_warning = self.limits.is_beyond_warning(value)
        
        z_score = (value - self.limits.center_line) / (self.limits.sigma + 1e-10)
        
        details = {
            'value': value,
            'center_line': self.limits.center_line,
            'ucl': self.limits.upper_control_limit,
            'lcl': self.limits.lower_control_limit,
            'z_score': z_score,
            'beyond_control': is_beyond_control,
            'beyond_warning': is_beyond_warning,
            'chart_type': self.chart_type.value
        }
        
        return is_beyond_control, details


class XBarChart(ControlChart):
    """X-bar (mean) control chart for monitoring process mean"""
    
    def __init__(self, config: SPCConfig):
        super().__init__(ControlChartType.XBAR, config)
    
    def _calculate_control_limits(self, data: np.ndarray):
        """Calculate control limits for X-bar chart"""
        center_line = np.mean(data)
        sigma = np.std(data, ddof=1)
        
        # Calculate control limits
        ucl = center_line + self.config.sigma_multiplier * sigma
        lcl = center_line - self.config.sigma_multiplier * sigma
        
        # Warning limits at 2 sigma
        uwl = center_line + 2 * sigma
        lwl = center_line - 2 * sigma
        
        self.limits = ControlLimits(
            center_line=center_line,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=uwl,
            lower_warning_limit=lwl,
            sigma=sigma
        )


class EWMAChart(ControlChart):
    """Exponentially Weighted Moving Average control chart"""
    
    def __init__(self, config: SPCConfig):
        super().__init__(ControlChartType.EWMA, config)
        self.ewma_value = None
        self.observation_count = 0
    
    def _calculate_control_limits(self, data: np.ndarray):
        """Calculate control limits for EWMA chart"""
        center_line = np.mean(data)
        sigma = np.std(data, ddof=1)
        
        # Initialize EWMA with process mean
        self.ewma_value = center_line
        
        # EWMA control limits (asymptotic)
        lambda_param = self.config.ewma_lambda
        limit_multiplier = self.config.sigma_multiplier * sigma * np.sqrt(
            lambda_param / (2 - lambda_param)
        )
        
        ucl = center_line + limit_multiplier
        lcl = center_line - limit_multiplier
        
        uwl = center_line + (2/3) * limit_multiplier
        lwl = center_line - (2/3) * limit_multiplier
        
        self.limits = ControlLimits(
            center_line=center_line,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=uwl,
            lower_warning_limit=lwl,
            sigma=sigma
        )
    
    def monitor(self, value: float) -> Tuple[bool, Dict]:
        """Monitor value using EWMA"""
        if not self.is_calibrated:
            raise ValueError("Chart must be calibrated before monitoring")
        
        # Update EWMA
        lambda_param = self.config.ewma_lambda
        self.ewma_value = lambda_param * value + (1 - lambda_param) * self.ewma_value
        self.observation_count += 1
        
        # Check against control limits
        is_beyond_control = self.limits.is_beyond_control(self.ewma_value)
        is_beyond_warning = self.limits.is_beyond_warning(self.ewma_value)
        
        z_score = (self.ewma_value - self.limits.center_line) / (self.limits.sigma + 1e-10)
        
        details = {
            'value': value,
            'ewma_value': self.ewma_value,
            'center_line': self.limits.center_line,
            'ucl': self.limits.upper_control_limit,
            'lcl': self.limits.lower_control_limit,
            'z_score': z_score,
            'beyond_control': is_beyond_control,
            'beyond_warning': is_beyond_warning,
            'chart_type': self.chart_type.value
        }
        
        return is_beyond_control, details


class CUSUMChart(ControlChart):
    """Cumulative Sum control chart for detecting small shifts"""
    
    def __init__(self, config: SPCConfig):
        super().__init__(ControlChartType.CUSUM, config)
        self.cusum_high = 0.0
        self.cusum_low = 0.0
    
    def _calculate_control_limits(self, data: np.ndarray):
        """Calculate parameters for CUSUM chart"""
        center_line = np.mean(data)
        sigma = np.std(data, ddof=1)
        
        # CUSUM uses h (decision interval) instead of traditional control limits
        h_value = self.config.cusum_h * sigma
        
        self.limits = ControlLimits(
            center_line=center_line,
            upper_control_limit=h_value,
            lower_control_limit=-h_value,
            upper_warning_limit=0.75 * h_value,
            lower_warning_limit=-0.75 * h_value,
            sigma=sigma
        )
        
        # Initialize CUSUM values
        self.cusum_high = 0.0
        self.cusum_low = 0.0
    
    def monitor(self, value: float) -> Tuple[bool, Dict]:
        """Monitor value using CUSUM"""
        if not self.is_calibrated:
            raise ValueError("Chart must be calibrated before monitoring")
        
        # Calculate CUSUM
        k_value = self.config.cusum_k * self.limits.sigma
        
        # High-side CUSUM (detecting upward shifts)
        self.cusum_high = max(0, self.cusum_high + value - self.limits.center_line - k_value)
        
        # Low-side CUSUM (detecting downward shifts)
        self.cusum_low = min(0, self.cusum_low + value - self.limits.center_line + k_value)
        
        # Check for violations
        is_beyond_control = (
            self.cusum_high > self.limits.upper_control_limit or
            self.cusum_low < self.limits.lower_control_limit
        )
        
        details = {
            'value': value,
            'cusum_high': self.cusum_high,
            'cusum_low': self.cusum_low,
            'center_line': self.limits.center_line,
            'h_value': self.limits.upper_control_limit,
            'beyond_control': is_beyond_control,
            'chart_type': self.chart_type.value
        }
        
        return is_beyond_control, details


class SPCRulesEngine:
    """Implements Western Electric and Nelson rules for control charts"""
    
    def __init__(self, config: SPCConfig):
        """
        Initialize rules engine
        
        Args:
            config: SPC configuration
        """
        self.config = config
        self.history: List[Dict] = []
        self.max_history = 15  # Keep last 15 points for rule checking
    
    def check_rules(self, value: float, limits: ControlLimits) -> List[ViolationType]:
        """
        Check all enabled SPC rules
        
        Args:
            value: Current value
            limits: Control limits
            
        Returns:
            List of violated rules
        """
        # Add to history
        z_score = (value - limits.center_line) / (limits.sigma + 1e-10)
        self.history.append({
            'value': value,
            'z_score': z_score,
            'above_center': value > limits.center_line
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        violations = []
        
        # Rule 1: Point beyond 3 sigma
        if abs(z_score) > 3:
            violations.append(ViolationType.BEYOND_LIMITS)
        
        if self.config.enable_western_electric:
            violations.extend(self._check_western_electric_rules())
        
        if self.config.enable_nelson_rules:
            violations.extend(self._check_nelson_rules())
        
        return violations
    
    def _check_western_electric_rules(self) -> List[ViolationType]:
        """Check Western Electric rules"""
        violations = []
        
        if len(self.history) < 3:
            return violations
        
        # Rule 2: Two out of three consecutive points beyond 2 sigma
        recent_3 = self.history[-3:]
        beyond_2sigma = sum(1 for p in recent_3 if abs(p['z_score']) > 2)
        if beyond_2sigma >= 2:
            violations.append(ViolationType.TWO_OF_THREE)
        
        if len(self.history) < 5:
            return violations
        
        # Rule 3: Four out of five consecutive points beyond 1 sigma
        recent_5 = self.history[-5:]
        beyond_1sigma = sum(1 for p in recent_5 if abs(p['z_score']) > 1)
        if beyond_1sigma >= 4:
            violations.append(ViolationType.FOUR_OF_FIVE)
        
        if len(self.history) < 8:
            return violations
        
        # Rule 4: Eight consecutive points on same side of center line
        recent_8 = self.history[-8:]
        all_above = all(p['above_center'] for p in recent_8)
        all_below = all(not p['above_center'] for p in recent_8)
        if all_above or all_below:
            violations.append(ViolationType.EIGHT_CONSECUTIVE)
        
        return violations
    
    def _check_nelson_rules(self) -> List[ViolationType]:
        """Check Nelson rules"""
        violations = []
        
        if len(self.history) < 6:
            return violations
        
        # Nelson Rule 5: Six consecutive points increasing or decreasing
        if self.config.detect_trends and len(self.history) >= 6:
            recent_6 = self.history[-6:]
            values = [p['value'] for p in recent_6]
            
            increasing = all(values[i] < values[i+1] for i in range(5))
            decreasing = all(values[i] > values[i+1] for i in range(5))
            
            if increasing or decreasing:
                violations.append(ViolationType.SIX_TRENDING)
        
        if len(self.history) < 8:
            return violations
        
        # Nelson Rule 8: Eight consecutive points beyond 1 sigma
        recent_8 = self.history[-8:]
        all_beyond_1sigma = all(abs(p['z_score']) > 1 for p in recent_8)
        if all_beyond_1sigma:
            violations.append(ViolationType.EIGHT_BEYOND_1SIGMA)
        
        if len(self.history) < 14:
            return violations
        
        # Nelson Rule 6: Fourteen consecutive alternating points
        recent_14 = self.history[-14:]
        alternating = all(
            recent_14[i]['above_center'] != recent_14[i+1]['above_center']
            for i in range(13)
        )
        if alternating:
            violations.append(ViolationType.FOURTEEN_ALTERNATING)
        
        if len(self.history) < 15:
            return violations
        
        # Nelson Rule 7: Fifteen consecutive points within 1 sigma
        recent_15 = self.history[-15:]
        all_within_1sigma = all(abs(p['z_score']) < 1 for p in recent_15)
        if all_within_1sigma:
            violations.append(ViolationType.FIFTEEN_WITHIN_1SIGMA)
        
        return violations


class StatisticalProcessControl:
    """
    Main SPC system for monitoring aggregate metrics
    Manages multiple control charts and provides comprehensive monitoring
    """
    
    def __init__(self, config: Optional[SPCConfig] = None):
        """
        Initialize SPC system
        
        Args:
            config: SPC configuration
        """
        self.config = config or SPCConfig()
        self.charts: Dict[str, ControlChart] = {}
        self.rules_engines: Dict[str, SPCRulesEngine] = {}
        self.metric_calculator = AggregateMetricCalculator()
        self.monitoring_results: Dict[str, List[Dict]] = defaultdict(list)
        self.is_calibrated = False
        
        logger.info("Statistical Process Control system initialized")
    
    def setup_monitoring(self, metrics: List[str], 
                        chart_types: Optional[Dict[str, ControlChartType]] = None):
        """
        Setup control charts for specified metrics
        
        Args:
            metrics: List of metric names to monitor
            chart_types: Optional mapping of metric to chart type
        """
        if chart_types is None:
            chart_types = {metric: ControlChartType.XBAR for metric in metrics}
        
        for metric in metrics:
            chart_type = chart_types.get(metric, ControlChartType.XBAR)
            
            if chart_type == ControlChartType.XBAR:
                self.charts[metric] = XBarChart(self.config)
            elif chart_type == ControlChartType.EWMA:
                self.charts[metric] = EWMAChart(self.config)
            elif chart_type == ControlChartType.CUSUM:
                self.charts[metric] = CUSUMChart(self.config)
            else:
                self.charts[metric] = XBarChart(self.config)
            
            self.rules_engines[metric] = SPCRulesEngine(self.config)
        
        logger.info(f"Set up monitoring for {len(metrics)} metrics")
    
    def calibrate(self, baseline_data: pd.DataFrame, metrics: List[str]):
        """
        Calibrate control charts with baseline data
        
        Args:
            baseline_data: Historical data for establishing control limits
            metrics: List of metric columns to calibrate
        """
        for metric in metrics:
            if metric not in baseline_data.columns:
                logger.warning(f"Metric {metric} not found in baseline data")
                continue
            
            if metric not in self.charts:
                logger.warning(f"No chart configured for metric {metric}")
                continue
            
            data = baseline_data[metric].dropna().values
            
            if len(data) >= self.config.min_baseline_points:
                self.charts[metric].calibrate(data)
            else:
                logger.warning(f"Insufficient baseline data for {metric}")
        
        self.is_calibrated = True
        logger.info(f"Calibrated {len(metrics)} control charts")
    
    def monitor_transactions(self, transactions: pd.DataFrame, 
                           time_period: str = 'D') -> pd.DataFrame:
        """
        Monitor transaction data for process anomalies
        
        Args:
            transactions: Transaction data to monitor
            time_period: Aggregation period
            
        Returns:
            DataFrame with monitoring results
        """
        if not self.is_calibrated:
            raise ValueError("System must be calibrated before monitoring")
        
        # Calculate aggregate metrics
        volume_metrics = self.metric_calculator.calculate_volume_metrics(
            transactions, time_period
        )
        user_metrics = self.metric_calculator.calculate_user_metrics(transactions)
        
        # Merge metrics
        metrics_df = pd.merge(
            volume_metrics,
            user_metrics,
            left_on='period',
            right_on='date',
            how='outer'
        ).fillna(0)
        
        # Monitor each metric
        results = []
        for idx, row in metrics_df.iterrows():
            period = row['period'] if 'period' in row else row['date']
            period_results = {
                'period': period,
                'anomalies': [],
                'violations': [],
                'severity': 'normal'
            }
            
            for metric_name, chart in self.charts.items():
                if metric_name not in metrics_df.columns:
                    continue
                
                value = row[metric_name]
                
                # Monitor with control chart
                is_anomaly, details = chart.monitor(value)
                
                # Check SPC rules
                violations = self.rules_engines[metric_name].check_rules(
                    value, chart.limits
                )
                
                if is_anomaly or violations:
                    period_results['anomalies'].append({
                        'metric': metric_name,
                        'value': value,
                        'details': details,
                        'violations': [v.value for v in violations]
                    })
                    
                    # Update severity
                    if len(violations) >= 2:
                        period_results['severity'] = 'critical'
                    elif is_anomaly:
                        period_results['severity'] = 'high'
                    elif period_results['severity'] == 'normal':
                        period_results['severity'] = 'medium'
            
            results.append(period_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df['anomaly_count'] = results_df['anomalies'].apply(len)
        results_df['has_anomaly'] = results_df['anomaly_count'] > 0
        
        logger.info(f"Monitored {len(results_df)} periods, "
                   f"found {results_df['has_anomaly'].sum()} anomalous periods")
        
        return results_df
    
    def get_control_chart_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all control charts
        
        Returns:
            Dictionary with chart summaries
        """
        summary = {}
        
        for metric_name, chart in self.charts.items():
            if not chart.is_calibrated:
                continue
            
            summary[metric_name] = {
                'chart_type': chart.chart_type.value,
                'center_line': chart.limits.center_line,
                'ucl': chart.limits.upper_control_limit,
                'lcl': chart.limits.lower_control_limit,
                'sigma': chart.limits.sigma,
                'baseline_points': len(chart.baseline_data)
            }
        
        return summary
    
    def generate_spc_report(self, monitoring_results: pd.DataFrame) -> Dict:
        """
        Generate comprehensive SPC monitoring report
        
        Args:
            monitoring_results: Results from monitor_transactions()
            
        Returns:
            Dictionary with SPC report
        """
        total_periods = len(monitoring_results)
        anomalous_periods = monitoring_results['has_anomaly'].sum()
        anomaly_rate = anomalous_periods / total_periods if total_periods > 0 else 0
        
        # Severity distribution
        severity_dist = monitoring_results['severity'].value_counts().to_dict()
        
        # Most problematic metrics
        all_anomalies = []
        for anomalies in monitoring_results['anomalies']:
            all_anomalies.extend(anomalies)
        
        metric_anomaly_counts = defaultdict(int)
        metric_violations = defaultdict(list)
        
        for anomaly in all_anomalies:
            metric = anomaly['metric']
            metric_anomaly_counts[metric] += 1
            metric_violations[metric].extend(anomaly['violations'])
        
        # Most common violations
        all_violations = []
        for violations in metric_violations.values():
            all_violations.extend(violations)
        
        violation_counts = defaultdict(int)
        for violation in all_violations:
            violation_counts[violation] += 1
        
        report = {
            'summary': {
                'total_periods_monitored': int(total_periods),
                'anomalous_periods': int(anomalous_periods),
                'anomaly_rate': float(anomaly_rate),
                'total_anomalies_detected': len(all_anomalies)
            },
            'severity_distribution': severity_dist,
            'metric_anomaly_counts': dict(metric_anomaly_counts),
            'top_violations': dict(
                sorted(violation_counts.items(), 
                      key=lambda x: x[1], reverse=True)[:5]
            ),
            'control_charts': self.get_control_chart_summary(),
            'configuration': {
                'sigma_multiplier': self.config.sigma_multiplier,
                'min_baseline_points': self.config.min_baseline_points,
                'western_electric_enabled': self.config.enable_western_electric,
                'nelson_rules_enabled': self.config.enable_nelson_rules
            }
        }
        
        return report


def generate_sample_transaction_data(n_days: int = 60, 
                                     base_volume: int = 1000) -> pd.DataFrame:
    """Generate sample transaction data with some anomalies"""
    np.random.seed(42)
    
    transactions = []
    start_date = datetime.now() - timedelta(days=n_days)
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        
        # Normal daily volume with some variation
        daily_volume = int(base_volume * (1 + 0.1 * np.random.randn()))
        
        # Inject anomalies
        if day in [20, 35, 50]:  # Spike days
            daily_volume = int(daily_volume * 1.5)
        elif day in [25, 45]:  # Drop days
            daily_volume = int(daily_volume * 0.6)
        
        # Generate transactions for the day
        for _ in range(daily_volume):
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            timestamp = current_date.replace(hour=hour, minute=minute)
            
            amount = max(10, np.random.gamma(2, 50))
            
            transactions.append({
                'transaction_id': f"TXN_{len(transactions)}",
                'user_id': f"USER_{np.random.randint(1, 500)}",
                'timestamp': timestamp,
                'amount': amount
            })
    
    return pd.DataFrame(transactions)


if __name__ == "__main__":
    # Demonstration
    print("=" * 80)
    print("Statistical Process Control - Demonstration")
    print("=" * 80)
    
    # Generate sample data
    print("\nGenerating sample transaction data...")
    transactions = generate_sample_transaction_data(n_days=60, base_volume=1000)
    print(f"Generated {len(transactions)} transactions over 60 days")
    
    # Split into baseline and monitoring periods
    baseline_cutoff = datetime.now() - timedelta(days=30)
    baseline_txns = transactions[
        pd.to_datetime(transactions['timestamp']) < baseline_cutoff
    ]
    monitoring_txns = transactions[
        pd.to_datetime(transactions['timestamp']) >= baseline_cutoff
    ]
    
    print(f"Baseline period: {len(baseline_txns)} transactions")
    print(f"Monitoring period: {len(monitoring_txns)} transactions")
    
    # Initialize SPC system
    print("\nInitializing SPC system...")
    config = SPCConfig(
        sigma_multiplier=3.0,
        enable_western_electric=True,
        enable_nelson_rules=True,
        min_baseline_points=20
    )
    
    spc = StatisticalProcessControl(config)
    
    # Setup monitoring for key metrics
    metrics_to_monitor = ['count', 'total_amount', 'mean_amount', 
                          'unique_users', 'txn_per_user']
    chart_types = {
        'count': ControlChartType.EWMA,
        'total_amount': ControlChartType.EWMA,
        'mean_amount': ControlChartType.XBAR,
        'unique_users': ControlChartType.CUSUM,
        'txn_per_user': ControlChartType.XBAR
    }
    
    spc.setup_monitoring(metrics_to_monitor, chart_types)
    
    # Calibrate with baseline data
    print("\nCalibrating control charts...")
    calc = AggregateMetricCalculator()
    baseline_volume = calc.calculate_volume_metrics(baseline_txns, 'D')
    baseline_user = calc.calculate_user_metrics(baseline_txns)
    
    baseline_combined = pd.merge(
        baseline_volume,
        baseline_user,
        left_on='period',
        right_on='date',
        how='outer'
    ).fillna(0)
    
    spc.calibrate(baseline_combined, metrics_to_monitor)
    
    # Monitor new data
    print("\nMonitoring transactions...")
    results = spc.monitor_transactions(monitoring_txns, time_period='D')
    
    # Display results
    print(f"\nMonitoring Results:")
    print(f"Total periods: {len(results)}")
    print(f"Anomalous periods: {results['has_anomaly'].sum()}")
    print(f"Anomaly rate: {results['has_anomaly'].mean():.2%}")
    
    print("\nSeverity Distribution:")
    print(results['severity'].value_counts())
    
    # Generate report
    print("\nGenerating SPC report...")
    report = spc.generate_spc_report(results)
    
    print("\nReport Summary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    print("\nTop Violations:")
    for violation, count in list(report['top_violations'].items())[:3]:
        print(f"  {violation}: {count}")
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("=" * 80)
    
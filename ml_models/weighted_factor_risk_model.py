import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
import joblib
import json

warnings.filterwarnings('ignore')
np.random.seed(42)


class WeightedFactorRiskModel:
    """
    Interpretable risk scoring model combining multiple fraud indicators
    through weighted aggregation with configurable normalization strategies.
    
    The model produces risk scores on a 0-100 scale where higher values
    indicate elevated fraud likelihood. Factor contributions are fully
    transparent, enabling detailed investigation guidance.
    """
    
    def __init__(self, entity_type: str = 'shop'):
        """
        Initialize weighted factor risk model.
        
        Args:
            entity_type: Type of entity to score ('shop' or 'beneficiary')
        """
        if entity_type not in ['shop', 'beneficiary']:
            raise ValueError("entity_type must be 'shop' or 'beneficiary'")
        
        self.entity_type = entity_type
        self.factor_weights = {}
        self.factor_scalers = {}
        self.risk_thresholds = {}
        self.calibration_data = {}
        self.evaluation_metrics = {}
        
        # Initialize default factor configurations
        self._initialize_factor_weights()
        self._initialize_risk_thresholds()
        
    def _initialize_factor_weights(self):
        """Initialize default factor weights based on entity type."""
        
        if self.entity_type == 'shop':
            # Shop-level risk factors with empirically derived weights
            self.factor_weights = {
                'anomaly_rate': 0.30,              # Proportion of flagged transactions
                'complaint_rate': 0.20,            # Beneficiary complaints per transaction
                'stock_discrepancy_rate': 0.20,    # Inventory vs transaction mismatch
                'inspection_overdue_days': 0.15,   # Days since last inspection
                'network_suspicion_score': 0.10,   # Graph analysis risk indicator
                'compliance_history_score': 0.05   # Historical compliance record
            }
        else:  # beneficiary
            # Beneficiary-level risk factors
            self.factor_weights = {
                'anomaly_rate': 0.35,              # Proportion of flagged transactions
                'transaction_regularity': 0.20,    # Mechanical vs organic patterns
                'shop_diversity_score': 0.15,      # Multiple shop usage indicator
                'quantity_deviation': 0.15,        # Distance from entitlement
                'verification_dispute_rate': 0.10, # Self-reported issues
                'geographic_anomaly_score': 0.05   # Distance-based suspicion
            }
    
    def _initialize_risk_thresholds(self):
        """Initialize risk score classification thresholds."""
        
        self.risk_thresholds = {
            'critical': 80,    # Immediate action required
            'high': 60,        # Priority investigation within 7 days
            'medium': 40,      # Routine inspection scheduling
            'low': 20,         # Standard monitoring
            'minimal': 0       # No immediate concern
        }
    
    def configure_weights(self, custom_weights: Dict[str, float]):
        """
        Configure custom factor weights.
        
        Args:
            custom_weights: Dictionary mapping factor names to weights (must sum to 1.0)
        """
        # Validate weights sum to 1.0
        weight_sum = sum(custom_weights.values())
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.4f}")
        
        # Validate factor names match expected factors
        expected_factors = set(self.factor_weights.keys())
        provided_factors = set(custom_weights.keys())
        
        if expected_factors != provided_factors:
            missing = expected_factors - provided_factors
            extra = provided_factors - expected_factors
            error_msg = []
            if missing:
                error_msg.append(f"Missing factors: {missing}")
            if extra:
                error_msg.append(f"Unknown factors: {extra}")
            raise ValueError("; ".join(error_msg))
        
        self.factor_weights = custom_weights.copy()
        print(f"✓ Custom weights configured for {self.entity_type} risk model")
    
    def fit(self, data: pd.DataFrame, labels: Optional[pd.Series] = None):
        """
        Fit normalization scalers to training data.
        
        Args:
            data: DataFrame containing raw factor values
            labels: Optional ground truth labels for calibration (1 for fraud, 0 for normal)
        """
        print(f"\nFitting {self.entity_type.capitalize()} Risk Model...")
        print("=" * 80)
        
        # Validate required columns
        required_factors = list(self.factor_weights.keys())
        missing_factors = set(required_factors) - set(data.columns)
        if missing_factors:
            raise ValueError(f"Missing required factors: {missing_factors}")
        
        # Fit robust scalers for each factor
        for factor in required_factors:
            scaler = RobustScaler()
            
            # Handle missing values
            factor_data = data[factor].fillna(data[factor].median())
            
            # Fit scaler
            scaler.fit(factor_data.values.reshape(-1, 1))
            self.factor_scalers[factor] = scaler
            
            print(f"  ✓ Fitted scaler for {factor}")
            print(f"    Median: {scaler.center_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")
        
        # Calibrate risk thresholds if labels provided
        if labels is not None:
            self._calibrate_thresholds(data, labels)
        
        print("\n✓ Model fitting completed successfully")
        return self
    
    def _calibrate_thresholds(self, data: pd.DataFrame, labels: pd.Series):
        """
        Calibrate risk thresholds based on labeled training data.
        
        Args:
            data: Factor values
            labels: Ground truth labels
        """
        print("\nCalibrating risk thresholds...")
        
        # Calculate risk scores for training data
        scores = self.predict_risk_scores(data)
        
        # Find optimal thresholds using ROC analysis
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Critical threshold: 95% specificity (5% FPR)
        critical_idx = np.argmin(np.abs(fpr - 0.05))
        self.risk_thresholds['critical'] = min(thresholds[critical_idx], 80)
        
        # High threshold: 90% specificity (10% FPR)
        high_idx = np.argmin(np.abs(fpr - 0.10))
        self.risk_thresholds['high'] = min(thresholds[high_idx], 60)
        
        # Medium threshold: 75% specificity (25% FPR)
        medium_idx = np.argmin(np.abs(fpr - 0.25))
        self.risk_thresholds['medium'] = min(thresholds[medium_idx], 40)
        
        print(f"  Calibrated thresholds:")
        print(f"    Critical: {self.risk_thresholds['critical']:.2f}")
        print(f"    High:     {self.risk_thresholds['high']:.2f}")
        print(f"    Medium:   {self.risk_thresholds['medium']:.2f}")
        
        # Store calibration statistics
        self.calibration_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc_score(labels, scores)
        }
        
        print(f"  ROC-AUC: {self.calibration_data['auc']:.4f}")
    
    def predict_risk_scores(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk scores for entities.
        
        Args:
            data: DataFrame containing factor values
            
        Returns:
            Array of risk scores (0-100 scale)
        """
        scores = np.zeros(len(data))
        
        # Calculate weighted contribution from each factor
        for factor, weight in self.factor_weights.items():
            # Get factor values
            factor_values = data[factor].fillna(data[factor].median()).values
            
            # Normalize using fitted scaler
            if factor in self.factor_scalers:
                normalized = self.factor_scalers[factor].transform(
                    factor_values.reshape(-1, 1)
                ).flatten()
            else:
                # Fallback to min-max if scaler not fitted
                normalized = (factor_values - factor_values.min()) / (
                    factor_values.max() - factor_values.min() + 1e-10
                )
            
            # Clip to [0, 1] range
            normalized = np.clip(normalized, 0, 1)
            
            # Add weighted contribution
            scores += normalized * weight * 100
        
        # Ensure scores are in valid range
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def predict_risk_categories(self, data: pd.DataFrame) -> np.ndarray:
        """
        Classify entities into risk categories.
        
        Args:
            data: DataFrame containing factor values
            
        Returns:
            Array of risk category labels
        """
        scores = self.predict_risk_scores(data)
        
        categories = np.full(len(scores), 'minimal', dtype=object)
        categories[scores >= self.risk_thresholds['low']] = 'low'
        categories[scores >= self.risk_thresholds['medium']] = 'medium'
        categories[scores >= self.risk_thresholds['high']] = 'high'
        categories[scores >= self.risk_thresholds['critical']] = 'critical'
        
        return categories
    
    def explain_risk_score(self, data: pd.DataFrame, entity_index: int) -> Dict:
        """
        Generate detailed explanation for a specific entity's risk score.
        
        Args:
            data: DataFrame containing factor values
            entity_index: Index of entity to explain
            
        Returns:
            Dictionary containing score breakdown and interpretation
        """
        entity_data = data.iloc[entity_index]
        
        # Calculate overall score
        total_score = self.predict_risk_scores(data.iloc[[entity_index]])[0]
        category = self.predict_risk_categories(data.iloc[[entity_index]])[0]
        
        # Calculate factor contributions
        contributions = {}
        for factor, weight in self.factor_weights.items():
            raw_value = entity_data[factor] if not pd.isna(entity_data[factor]) else data[factor].median()
            
            # Normalize
            if factor in self.factor_scalers:
                normalized = self.factor_scalers[factor].transform(
                    np.array([[raw_value]])
                ).flatten()[0]
            else:
                normalized = 0.5
            
            normalized = np.clip(normalized, 0, 1)
            contribution = normalized * weight * 100
            
            contributions[factor] = {
                'raw_value': float(raw_value),
                'normalized_value': float(normalized),
                'weight': float(weight),
                'contribution': float(contribution),
                'percentage_of_total': float(contribution / total_score * 100) if total_score > 0 else 0
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(contributions, category)
        
        explanation = {
            'entity_type': self.entity_type,
            'entity_index': entity_index,
            'risk_score': float(total_score),
            'risk_category': category,
            'factor_contributions': contributions,
            'top_risk_factors': sorted(
                contributions.items(),
                key=lambda x: x[1]['contribution'],
                reverse=True
            )[:3],
            'recommendations': recommendations
        }
        
        return explanation
    
    def _generate_recommendations(self, contributions: Dict, category: str) -> List[str]:
        """Generate actionable recommendations based on risk factors."""
        
        recommendations = []
        
        # Sort factors by contribution
        sorted_factors = sorted(
            contributions.items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        
        # Generate category-specific recommendations
        if category == 'critical':
            recommendations.append("IMMEDIATE ACTION REQUIRED: Suspend operations pending investigation")
            recommendations.append("Assign senior investigator for comprehensive audit")
        elif category == 'high':
            recommendations.append("Priority investigation required within 7 days")
            recommendations.append("Review all transactions from past 30 days")
        elif category == 'medium':
            recommendations.append("Schedule routine inspection within 30 days")
            recommendations.append("Monitor for pattern changes")
        
        # Add factor-specific recommendations
        for factor_name, factor_data in sorted_factors[:2]:
            if factor_data['contribution'] > 15:
                if 'anomaly' in factor_name:
                    recommendations.append(f"High anomaly rate detected - review flagged transactions")
                elif 'complaint' in factor_name:
                    recommendations.append(f"Elevated complaints - interview affected beneficiaries")
                elif 'stock' in factor_name:
                    recommendations.append(f"Stock discrepancies found - conduct physical inventory audit")
                elif 'inspection' in factor_name:
                    recommendations.append(f"Overdue inspection - schedule immediate visit")
        
        return recommendations
    
    def evaluate(self, data: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        Evaluate model performance against ground truth labels.
        
        Args:
            data: Factor values
            labels: Ground truth labels (1 for fraud, 0 for normal)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {self.entity_type.capitalize()} Risk Model...")
        print("=" * 80)
        
        # Calculate scores and predictions
        risk_scores = self.predict_risk_scores(data)
        risk_categories = self.predict_risk_categories(data)
        
        # Convert categories to binary predictions (critical/high = fraud)
        binary_predictions = np.isin(risk_categories, ['critical', 'high']).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, binary_predictions)
        f1 = f1_score(labels, binary_predictions, zero_division=0)
        roc_auc = roc_auc_score(labels, risk_scores)
        avg_precision = average_precision_score(labels, risk_scores)
        
        # Classification report
        class_report = classification_report(
            labels,
            binary_predictions,
            target_names=['Normal', 'Fraud'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, binary_predictions)
        
        # Category distribution
        category_dist = pd.Series(risk_categories).value_counts().to_dict()
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'precision': class_report['Fraud']['precision'],
            'recall': class_report['Fraud']['recall'],
            'confusion_matrix': cm,
            'classification_report': class_report,
            'category_distribution': category_dist,
            'mean_risk_score': float(risk_scores.mean()),
            'median_risk_score': float(np.median(risk_scores))
        }
        
        self.evaluation_metrics = metrics
        
        # Print summary
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1-Score:          {f1:.4f}")
        print(f"  ROC-AUC:           {roc_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        
        print(f"\nRisk Category Distribution:")
        for category, count in sorted(category_dist.items(), 
                                     key=lambda x: self.risk_thresholds.get(x[0], 0)):
            percentage = count / len(data) * 100
            print(f"  {category.capitalize():12s}: {count:6d} ({percentage:5.2f}%)")
        
        return metrics
    
    def plot_comprehensive_evaluation(self, data: pd.DataFrame, labels: pd.Series):
        """Generate comprehensive evaluation visualizations."""
        
        risk_scores = self.predict_risk_scores(data)
        risk_categories = self.predict_risk_categories(data)
        binary_predictions = np.isin(risk_categories, ['critical', 'high']).astype(int)
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. Risk Score Distribution by True Label
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(risk_scores[labels == 0], bins=50, alpha=0.6, label='Normal', 
                color='green', density=True, edgecolor='black')
        ax1.hist(risk_scores[labels == 1], bins=50, alpha=0.6, label='Fraud', 
                color='red', density=True, edgecolor='black')
        ax1.axvline(self.risk_thresholds['critical'], color='darkred', 
                   linestyle='--', linewidth=2, label='Critical Threshold')
        ax1.axvline(self.risk_thresholds['high'], color='orange', 
                   linestyle='--', linewidth=2, label='High Threshold')
        ax1.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        cm = confusion_matrix(labels, binary_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax2, 
                   cbar_kws={'label': 'Count'}, linewidths=2)
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        ax2.set_xticklabels(['Normal', 'Fraud'])
        ax2.set_yticklabels(['Normal', 'Fraud'])
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[0, 2])
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, risk_scores)
        ax3.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC={self.evaluation_metrics["roc_auc"]:.4f})', 
                color='#2ecc71')
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
        ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.fill_between(fpr, tpr, alpha=0.2, color='#2ecc71')
        
        # 4. Factor Weights Visualization
        ax4 = fig.add_subplot(gs[1, 0])
        factors = list(self.factor_weights.keys())
        weights = list(self.factor_weights.values())
        colors_weights = plt.cm.viridis(np.linspace(0.3, 0.9, len(factors)))
        bars = ax4.barh(factors, weights, color=colors_weights, edgecolor='black', linewidth=1.5)
        ax4.set_title('Factor Weights', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Weight (fraction of total score)')
        ax4.grid(axis='x', alpha=0.3)
        
        for bar, weight in zip(bars, weights):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{weight:.2f}', ha='left', va='center', fontweight='bold')
        
        # 5. Precision-Recall Curve
        ax5 = fig.add_subplot(gs[1, 1])
        precision, recall, _ = precision_recall_curve(labels, risk_scores)
        ax5.plot(recall, precision, linewidth=3, 
                label=f'PR (AP={self.evaluation_metrics["average_precision"]:.4f})', 
                color='#3498db')
        ax5.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.legend()
        ax5.grid(alpha=0.3)
        ax5.fill_between(recall, precision, alpha=0.2, color='#3498db')
        
        # 6. Risk Category Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        category_counts = pd.Series(risk_categories).value_counts()
        category_order = ['minimal', 'low', 'medium', 'high', 'critical']
        category_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        
        ordered_counts = [category_counts.get(cat, 0) for cat in category_order]
        bars_cat = ax6.bar(category_order, ordered_counts, color=category_colors, 
                          alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Count')
        ax6.set_xlabel('Risk Category')
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars_cat, ordered_counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max(ordered_counts)*0.02,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Average Factor Contributions
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Calculate average normalized values for each factor
        avg_contributions = {}
        for factor in self.factor_weights.keys():
            factor_data = data[factor].fillna(data[factor].median()).values
            if factor in self.factor_scalers:
                normalized = self.factor_scalers[factor].transform(
                    factor_data.reshape(-1, 1)
                ).flatten()
            else:
                normalized = np.zeros_like(factor_data)
            
            avg_contributions[factor] = np.clip(normalized, 0, 1).mean() * self.factor_weights[factor] * 100
        
        factors_contrib = list(avg_contributions.keys())
        contrib_values = list(avg_contributions.values())
        colors_contrib = plt.cm.plasma(np.linspace(0.2, 0.9, len(factors_contrib)))
        
        bars_contrib = ax7.barh(factors_contrib, contrib_values, color=colors_contrib, 
                               edgecolor='black', linewidth=1.5)
        ax7.set_title('Average Factor Contributions to Risk Scores', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Average Contribution (points)')
        ax7.grid(axis='x', alpha=0.3)
        
        for bar, value in zip(bars_contrib, contrib_values):
            width = bar.get_width()
            ax7.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        # 8. Performance Metrics
        ax8 = fig.add_subplot(gs[2, 2])
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            self.evaluation_metrics['accuracy'],
            self.evaluation_metrics['precision'],
            self.evaluation_metrics['recall'],
            self.evaluation_metrics['f1_score'],
            self.evaluation_metrics['roc_auc']
        ]
        
        colors_metrics = ['#27ae60' if v >= 0.90 else '#f39c12' if v >= 0.80 else '#e74c3c' 
                         for v in metric_values]
        bars_metrics = ax8.bar(metric_names, metric_values, color=colors_metrics, 
                              alpha=0.8, edgecolor='black', linewidth=2)
        ax8.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Score')
        ax8.set_ylim(0, 1.1)
        ax8.grid(axis='y', alpha=0.3)
        ax8.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='90% Target')
        ax8.legend()
        
        for bar, value in zip(bars_metrics, metric_values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Weighted Factor Risk Model - {self.entity_type.capitalize()} Evaluation', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        filename = f'weighted_factor_{self.entity_type}_evaluation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Evaluation plots saved as '{filename}'")
        plt.show()
    
    def save_model(self, filepath: str = None):
        """Save model configuration and scalers."""
        
        if filepath is None:
            filepath = f'weighted_factor_{self.entity_type}_model'
        
        model_package = {
            'entity_type': self.entity_type,
            'factor_weights': self.factor_weights,
            'factor_scalers': self.factor_scalers,
            'risk_thresholds': self.risk_thresholds,
            'calibration_data': self.calibration_data,
            'evaluation_metrics': self.evaluation_metrics
        }
        
        joblib.dump(model_package, f'{filepath}.pkl')
        print(f"\n✓ Model saved to {filepath}.pkl")
    
    def load_model(self, filepath: str = None):
        """Load model configuration and scalers."""
        
        if filepath is None:
            filepath = f'weighted_factor_{self.entity_type}_model'
        
        model_package = joblib.load(f'{filepath}.pkl')
        
        self.entity_type = model_package['entity_type']
        self.factor_weights = model_package['factor_weights']
        self.factor_scalers = model_package['factor_scalers']
        self.risk_thresholds = model_package['risk_thresholds']
        self.calibration_data = model_package.get('calibration_data', {})
        self.evaluation_metrics = model_package.get('evaluation_metrics', {})
        
        print(f"✓ Model loaded from {filepath}.pkl")


def generate_risk_factor_data(n_samples: int = 10000, 
                              fraud_rate: float = 0.05,
                              entity_type: str = 'shop') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic risk factor data for model training and evaluation.
    
    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent samples
        entity_type: Type of entity ('shop' or 'beneficiary')
        
    Returns:
        Tuple of (factor_dataframe, labels)
    """
    print(f"\nGenerating synthetic {entity_type} risk factor data...")
    print(f"  Samples: {n_samples:,}")
    print(f"  Fraud rate: {fraud_rate*100:.1f}%")
    
    # Determine fraud samples
    n_fraud = int(n_samples * fraud_rate)
    labels = np.concatenate([
        np.ones(n_fraud),
        np.zeros(n_samples - n_fraud)
    ])
    np.random.shuffle(labels)
    
    if entity_type == 'shop':
        # Shop risk factors
        data = pd.DataFrame({
            'anomaly_rate': np.where(
                labels == 1,
                np.random.beta(8, 2, n_samples) * 0.8,  # High for fraud
                np.random.beta(2, 8, n_samples) * 0.2   # Low for normal
            ),
            'complaint_rate': np.where(
                labels == 1,
                np.random.gamma(5, 0.02, n_samples),    # Elevated for fraud
                np.random.gamma(2, 0.01, n_samples)     # Lower for normal
            ),
            'stock_discrepancy_rate': np.where(
                labels == 1,
                np.random.beta(6, 4, n_samples) * 0.5,  # Higher for fraud
                np.random.beta(2, 8, n_samples) * 0.1   # Lower for normal
            ),
            'inspection_overdue_days': np.where(
                labels == 1,
                np.random.exponential(180, n_samples),  # More overdue for fraud
                np.random.exponential(60, n_samples)    # Less for normal
            ),
            'network_suspicion_score': np.where(
                labels == 1,
                np.random.beta(7, 3, n_samples),        # Higher for fraud
                np.random.beta(2, 5, n_samples)         # Lower for normal
            ),
            'compliance_history_score': np.where(
                labels == 1,
                np.random.beta(2, 8, n_samples) * 100,  # Lower for fraud
                np.random.beta(8, 2, n_samples) * 100   # Higher for normal
            )
        })
    else:  # beneficiary
        # Beneficiary risk factors
        data = pd.DataFrame({
            'anomaly_rate': np.where(
                labels == 1,
                np.random.beta(7, 3, n_samples) * 0.9,
                np.random.beta(2, 8, n_samples) * 0.15
            ),
            'transaction_regularity': np.where(
                labels == 1,
                np.random.beta(2, 8, n_samples),        # Very regular (mechanical)
                np.random.beta(5, 5, n_samples)         # Moderate variation
            ),
            'shop_diversity_score': np.where(
                labels == 1,
                np.random.poisson(4, n_samples),        # Multiple shops
                np.random.poisson(1.2, n_samples)       # Few shops
            ),
            'quantity_deviation': np.where(
                labels == 1,
                np.random.gamma(3, 0.3, n_samples),     # Large deviations
                np.random.gamma(1, 0.1, n_samples)      # Small deviations
            ),
            'verification_dispute_rate': np.where(
                labels == 1,
                np.random.beta(6, 4, n_samples) * 0.3,
                np.random.beta(2, 8, n_samples) * 0.05
            ),
            'geographic_anomaly_score': np.where(
                labels == 1,
                np.random.beta(5, 5, n_samples),
                np.random.beta(2, 8, n_samples)
            )
        })
    
    # Add some noise and clip values
    for col in data.columns:
        data[col] = np.clip(data[col], 0, data[col].quantile(0.99))
    
    print(f"✓ Generated {len(data):,} samples with {labels.sum():.0f} fraud cases")
    
    return data, pd.Series(labels, name='is_fraud')


def main():
    """
    Execute complete weighted factor risk model training and evaluation pipeline.
    """
    
    print("\n" + "╔" + "═" * 88 + "╗")
    print("║" + " " * 20 + "GRAINSECURE PDS MONITORING SYSTEM" + " " * 35 + "║")
    print("║" + " " * 12 + "Weighted Factor Risk Scoring Model - Interpretable Fraud Assessment" + " " * 8 + "║")
    print("╚" + "═" * 88 + "╝")
    print()
    
    # Train models for both entity types
    for entity_type in ['shop', 'beneficiary']:
        print("\n" + "=" * 88)
        print(f"TRAINING {entity_type.upper()} RISK MODEL")
        print("=" * 88)
        
        # Step 1: Generate synthetic data
        print(f"\nSTEP 1: Generating Synthetic {entity_type.capitalize()} Risk Factor Data")
        print("-" * 88)
        
        # Training data
        train_data, train_labels = generate_risk_factor_data(
            n_samples=15000,
            fraud_rate=0.05,
            entity_type=entity_type
        )
        
        # Test data
        test_data, test_labels = generate_risk_factor_data(
            n_samples=5000,
            fraud_rate=0.05,
            entity_type=entity_type
        )
        
        # Step 2: Initialize and fit model
        print(f"\nSTEP 2: Initializing and Fitting {entity_type.capitalize()} Risk Model")
        print("-" * 88)
        
        model = WeightedFactorRiskModel(entity_type=entity_type)
        model.fit(train_data, train_labels)
        
        # Step 3: Evaluate model
        print(f"\nSTEP 3: Evaluating Model Performance")
        print("-" * 88)
        
        metrics = model.evaluate(test_data, test_labels)
        
        # Step 4: Generate visualizations
        print(f"\nSTEP 4: Generating Evaluation Visualizations")
        print("-" * 88)
        
        model.plot_comprehensive_evaluation(test_data, test_labels)
        
        # Step 5: Demonstrate score explanation
        print(f"\nSTEP 5: Demonstrating Risk Score Explanation")
        print("-" * 88)
        
        # Find a high-risk example
        risk_scores = model.predict_risk_scores(test_data)
        high_risk_idx = np.argmax(risk_scores)
        
        explanation = model.explain_risk_score(test_data, high_risk_idx)
        
        print(f"\nDetailed Risk Explanation for {entity_type.capitalize()} #{high_risk_idx}:")
        print(f"  Risk Score: {explanation['risk_score']:.2f}")
        print(f"  Risk Category: {explanation['risk_category'].upper()}")
        print(f"\nTop Risk Factors:")
        for i, (factor, data) in enumerate(explanation['top_risk_factors'], 1):
            print(f"  {i}. {factor}:")
            print(f"     Raw Value: {data['raw_value']:.4f}")
            print(f"     Contribution: {data['contribution']:.2f} points ({data['percentage_of_total']:.1f}% of total)")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(explanation['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Step 6: Save model
        print(f"\nSTEP 6: Saving Trained Model")
        print("-" * 88)
        
        model.save_model()
        
        # Export sample predictions
        predictions_df = pd.DataFrame({
            f'{entity_type}_index': range(len(test_data)),
            'risk_score': model.predict_risk_scores(test_data),
            'risk_category': model.predict_risk_categories(test_data),
            'true_label': test_labels.values
        })
        
        predictions_df.to_csv(f'weighted_factor_{entity_type}_predictions.csv', index=False)
        print(f"✓ Predictions saved to 'weighted_factor_{entity_type}_predictions.csv'")
    
    # Final summary
    print("\n" + "=" * 88)
    print("WEIGHTED FACTOR RISK MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 88)
    print()
    print("Model Deliverables:")
    print("  1. weighted_factor_shop_model.pkl - Shop risk scoring model")
    print("  2. weighted_factor_beneficiary_model.pkl - Beneficiary risk scoring model")
    print("  3. weighted_factor_shop_evaluation.png - Shop model evaluation charts")
    print("  4. weighted_factor_beneficiary_evaluation.png - Beneficiary model charts")
    print("  5. weighted_factor_shop_predictions.csv - Shop risk predictions")
    print("  6. weighted_factor_beneficiary_predictions.csv - Beneficiary predictions")
    print()
    print("Model Capabilities:")
    print("  • Fully interpretable risk scores with factor breakdowns")
    print("  • Configurable factor weights for domain expertise integration")
    print("  • Calibrated risk thresholds for actionable categorization")
    print("  • Detailed risk explanations with investigation recommendations")
    print("  • Transparent scoring enabling regulatory compliance")
    print()
    print("Next Steps:")
    print("  • Integrate with ensemble prediction system")
    print("  • Deploy risk scoring API endpoints")
    print("  • Configure alert generation based on risk thresholds")
    print("  • Implement continuous monitoring dashboard")
    print()


if __name__ == "__main__":
    main()
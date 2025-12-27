"""
Utility Functions for Health XAI Project

This module contains shared utility functions used across the project,
including configuration loading, logging setup, file I/O operations,
and common helper functions.
"""

import os
import logging
import yaml
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_config(config_path: str = "src/config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Project configuration dictionary
        
    Returns:
        Configured logger instance
    """
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Setup logger
    logger = logging.getLogger('health_xai')
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if log_config.get('console_handler', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_config.get('file_handler', True):
        log_file = log_config.get('log_file', 'logs/health_xai.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_results(data: Any, 
                filepath: str, 
                format_type: str = 'auto') -> None:
    """
    Save data to file with automatic format detection.
    
    Args:
        data: Data to save (DataFrame, dict, model, etc.)
        filepath: Output file path
        format_type: File format ('auto', 'csv', 'json', 'pickle', 'joblib')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format from file extension
    if format_type == 'auto':
        format_type = filepath.suffix[1:].lower()
    
    try:
        if format_type in ['csv']:
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                pd.DataFrame(data).to_csv(filepath, index=False)
                
        elif format_type in ['json']:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format_type in ['pkl', 'pickle']:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
        elif format_type in ['joblib']:
            joblib.dump(data, filepath)
            
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        logging.info(f"Results saved to: {filepath}")
        
    except Exception as e:
        logging.error(f"Error saving results to {filepath}: {e}")
        raise


def load_data(filepath: str, **kwargs) -> Union[pd.DataFrame, Any]:
    """
    Load data from file with automatic format detection.
    
    Args:
        filepath: Path to data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_ext = filepath.suffix.lower()
    
    try:
        if file_ext == '.csv':
            return pd.read_csv(filepath, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(filepath, **kwargs)
        elif file_ext == '.parquet':
            return pd.read_parquet(filepath, **kwargs)
        elif file_ext == '.json':
            return pd.read_json(filepath, **kwargs)
        elif file_ext in ['.pkl', '.pickle']:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif file_ext == '.joblib':
            return joblib.load(filepath)
        else:
            # Try to read as CSV by default
            return pd.read_csv(filepath, **kwargs)
            
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise


def create_timestamp() -> str:
    """Create timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_proba is not None:
        if y_proba.ndim > 1:
            # Multi-class case - use probability for positive class
            y_proba_pos = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
        else:
            y_proba_pos = y_proba
            
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            metrics['average_precision'] = average_precision_score(y_true, y_proba_pos)
        except ValueError as e:
            logging.warning(f"Could not calculate probabilistic metrics: {e}")
    
    return metrics


def print_classification_summary(y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               y_proba: Optional[np.ndarray] = None,
                               class_names: Optional[List[str]] = None) -> None:
    """
    Print comprehensive classification performance summary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_proba: Predicted probabilities (optional)
        class_names: Names of classes (optional)
    """
    print("\\n" + "="*50)
    print("CLASSIFICATION PERFORMANCE SUMMARY")
    print("="*50)
    
    # Basic metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    print("\\nOverall Metrics:")
    print("-" * 20)
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title():<20}: {value:.4f}")
    
    # Detailed classification report
    print("\\nDetailed Classification Report:")
    print("-" * 35)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    print("\\nConfusion Matrix:")
    print("-" * 17)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


def validate_data_splits(train_data: pd.DataFrame,
                        val_data: pd.DataFrame, 
                        test_data: pd.DataFrame,
                        target_col: str) -> Dict[str, Any]:
    """
    Validate data splits for consistency and quality.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset  
        target_col: Target variable column name
        
    Returns:
        Validation results dictionary
    """
    results = {
        'validation_passed': True,
        'warnings': [],
        'errors': []
    }
    
    # Check for overlapping indices
    train_idx = set(train_data.index)
    val_idx = set(val_data.index)
    test_idx = set(test_data.index)
    
    if train_idx & val_idx:
        results['errors'].append("Training and validation sets have overlapping indices")
        results['validation_passed'] = False
        
    if train_idx & test_idx:
        results['errors'].append("Training and test sets have overlapping indices")
        results['validation_passed'] = False
        
    if val_idx & test_idx:
        results['errors'].append("Validation and test sets have overlapping indices")
        results['validation_passed'] = False
    
    # Check target distribution
    train_dist = train_data[target_col].value_counts(normalize=True)
    val_dist = val_data[target_col].value_counts(normalize=True)
    test_dist = test_data[target_col].value_counts(normalize=True)
    
    # Check for significant distribution differences (using 10% threshold)
    for class_val in train_dist.index:
        train_prop = train_dist.get(class_val, 0)
        val_prop = val_dist.get(class_val, 0)
        test_prop = test_dist.get(class_val, 0)
        
        if abs(train_prop - val_prop) > 0.1:
            results['warnings'].append(
                f"Large difference in class {class_val} distribution between train ({train_prop:.3f}) and validation ({val_prop:.3f})"
            )
            
        if abs(train_prop - test_prop) > 0.1:
            results['warnings'].append(
                f"Large difference in class {class_val} distribution between train ({train_prop:.3f}) and test ({test_prop:.3f})"
            )
    
    # Check dataset sizes
    total_size = len(train_data) + len(val_data) + len(test_data)
    train_ratio = len(train_data) / total_size
    val_ratio = len(val_data) / total_size
    test_ratio = len(test_data) / total_size
    
    if train_ratio < 0.6:
        results['warnings'].append(f"Training set might be too small ({train_ratio:.2%} of total data)")
    
    if val_ratio < 0.1:
        results['warnings'].append(f"Validation set might be too small ({val_ratio:.2%} of total data)")
        
    if test_ratio < 0.1:
        results['warnings'].append(f"Test set might be too small ({test_ratio:.2%} of total data)")
    
    return results


def memory_usage_info() -> Dict[str, str]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage details
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': f"{memory_info.rss / 1024 / 1024:.2f} MB",
        'vms': f"{memory_info.vms / 1024 / 1024:.2f} MB",
        'percent': f"{process.memory_percent():.2f}%",
        'available': f"{psutil.virtual_memory().available / 1024 / 1024:.2f} MB"
    }


def diagnose_model_fit(train_scores: Union[List, np.ndarray], 
                      val_scores: Union[List, np.ndarray], 
                      model_name: str = "Model",
                      threshold_gap: float = 0.05,
                      threshold_low_performance: float = 0.7) -> Dict[str, Any]:
    """
    Diagnose if a model is overfitting, underfitting, or well-fitted based on training and validation scores.
    
    Args:
        train_scores: Training scores (higher is better, e.g., accuracy, ROC-AUC)
        val_scores: Validation scores 
        model_name: Name of the model for reporting
        threshold_gap: Maximum acceptable gap between train and validation scores
        threshold_low_performance: Minimum acceptable performance threshold
        
    Returns:
        Dictionary containing diagnosis, recommendations, and metrics
    """
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)
    
    # Calculate key metrics
    avg_train = np.mean(train_scores)
    avg_val = np.mean(val_scores)
    gap = avg_train - avg_val
    
    # Calculate trends
    train_trend = np.polyfit(range(len(train_scores)), train_scores, 1)[0] if len(train_scores) > 1 else 0
    val_trend = np.polyfit(range(len(val_scores)), val_scores, 1)[0] if len(val_scores) > 1 else 0
    
    # Stability metrics
    train_std = np.std(train_scores)
    val_std = np.std(val_scores)
    
    # Diagnosis logic
    diagnosis = "WELL_FITTED"
    severity = "LOW"
    recommendations = []
    
    # Check for overfitting
    if gap > threshold_gap:
        diagnosis = "OVERFITTING"
        if gap > threshold_gap * 2:
            severity = "HIGH"
        elif gap > threshold_gap * 1.5:
            severity = "MEDIUM"
        else:
            severity = "LOW"
            
        recommendations.extend([
            "Reduce model complexity (fewer parameters, simpler architecture)",
            "Increase regularization (L1/L2, dropout, early stopping)",
            "Add more training data or use data augmentation",
            "Use cross-validation for more robust evaluation",
            "Consider ensemble methods to reduce variance"
        ])
    
    # Check for underfitting
    elif avg_train < threshold_low_performance or avg_val < threshold_low_performance:
        diagnosis = "UNDERFITTING"
        if max(avg_train, avg_val) < threshold_low_performance * 0.8:
            severity = "HIGH"
        elif max(avg_train, avg_val) < threshold_low_performance * 0.9:
            severity = "MEDIUM"
        else:
            severity = "LOW"
            
        recommendations.extend([
            "Increase model complexity (more parameters, deeper architecture)",
            "Reduce regularization strength",
            "Add more relevant features or feature engineering",
            "Train for more epochs/iterations",
            "Check for data quality issues or class imbalance"
        ])
    
    # Check for high variance (unstable training)
    elif train_std > 0.1 or val_std > 0.1:
        diagnosis = "HIGH_VARIANCE"
        severity = "MEDIUM"
        recommendations.extend([
            "Use more stable training procedures",
            "Implement learning rate scheduling",
            "Increase batch size or use gradient accumulation",
            "Add batch normalization or layer normalization",
            "Use ensemble methods to reduce variance"
        ])
    
    # Check for convergence issues
    elif abs(train_trend) < 0.001 and abs(val_trend) < 0.001 and avg_val < threshold_low_performance:
        diagnosis = "CONVERGENCE_ISSUES"
        severity = "MEDIUM"
        recommendations.extend([
            "Adjust learning rate or use adaptive optimizers",
            "Check gradient flow and avoid vanishing gradients",
            "Improve data preprocessing and normalization",
            "Use different initialization strategies",
            "Consider different model architectures"
        ])
    
    return {
        'model_name': model_name,
        'diagnosis': diagnosis,
        'severity': severity,
        'metrics': {
            'avg_train_score': round(avg_train, 4),
            'avg_val_score': round(avg_val, 4),
            'performance_gap': round(gap, 4),
            'train_stability': round(train_std, 4),
            'val_stability': round(val_std, 4),
            'train_trend': round(train_trend, 6),
            'val_trend': round(val_trend, 6)
        },
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    }


def plot_learning_curves(train_scores: Union[List, np.ndarray], 
                        val_scores: Union[List, np.ndarray],
                        model_name: str = "Model",
                        save_path: Optional[str] = None,
                        show_diagnosis: bool = True) -> None:
    """
    Create comprehensive learning curve plots with diagnosis overlay.
    
    Args:
        train_scores: Training scores over epochs/iterations
        val_scores: Validation scores over epochs/iterations
        model_name: Name of the model for plot title
        save_path: Path to save the plot (optional)
        show_diagnosis: Whether to include diagnosis information on plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(train_scores) + 1)
    
    # Main learning curve
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2, alpha=0.8)
    ax1.fill_between(epochs, train_scores, alpha=0.1, color='blue')
    ax1.fill_between(epochs, val_scores, alpha=0.1, color='red')
    ax1.set_xlabel('Epoch/Iteration')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{model_name} - Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance gap over time
    ax2 = axes[0, 1]
    gap = np.array(train_scores) - np.array(val_scores)
    ax2.plot(epochs, gap, 'g-', linewidth=2, alpha=0.8)
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    ax2.fill_between(epochs, gap, alpha=0.2, color='green')
    ax2.set_xlabel('Epoch/Iteration')
    ax2.set_ylabel('Performance Gap (Train - Val)')
    ax2.set_title('Overfitting Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Score distribution
    ax3 = axes[1, 0]
    ax3.hist(train_scores, bins=15, alpha=0.6, label='Training', color='blue', density=True)
    ax3.hist(val_scores, bins=15, alpha=0.6, label='Validation', color='red', density=True)
    ax3.axvline(np.mean(train_scores), color='blue', linestyle='--', alpha=0.8, label='Train Mean')
    ax3.axvline(np.mean(val_scores), color='red', linestyle='--', alpha=0.8, label='Val Mean')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Stability analysis
    ax4 = axes[1, 1]
    window_size = min(5, len(train_scores) // 3)
    if window_size >= 2:
        train_rolling_std = pd.Series(train_scores).rolling(window=window_size).std()
        val_rolling_std = pd.Series(val_scores).rolling(window=window_size).std()
        
        ax4.plot(epochs, train_rolling_std, 'b-', label=f'Training Std (window={window_size})', alpha=0.8)
        ax4.plot(epochs, val_rolling_std, 'r-', label=f'Validation Std (window={window_size})', alpha=0.8)
        ax4.set_xlabel('Epoch/Iteration')
        ax4.set_ylabel('Rolling Standard Deviation')
        ax4.set_title('Training Stability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor stability analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # Add diagnosis text if requested
    if show_diagnosis:
        diagnosis = diagnose_model_fit(train_scores, val_scores, model_name)
        
        # Create diagnosis text
        diag_text = f"Diagnosis: {diagnosis['diagnosis']} ({diagnosis['severity']})\n"
        diag_text += f"Avg Train: {diagnosis['metrics']['avg_train_score']:.3f}\n"
        diag_text += f"Avg Val: {diagnosis['metrics']['avg_val_score']:.3f}\n"
        diag_text += f"Gap: {diagnosis['metrics']['performance_gap']:.3f}"
        
        # Color based on diagnosis
        color_map = {
            'OVERFITTING': 'red',
            'UNDERFITTING': 'orange', 
            'HIGH_VARIANCE': 'purple',
            'CONVERGENCE_ISSUES': 'brown',
            'WELL_FITTED': 'green'
        }
        text_color = color_map.get(diagnosis['diagnosis'], 'black')
        
        fig.text(0.02, 0.98, diag_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                color=text_color, weight='bold')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve plot saved to: {save_path}")
    
    # Close the plot instead of showing it interactively
    plt.close()


def save_model_diagnosis_report(diagnosis: Dict[str, Any], 
                               save_path: str,
                               include_recommendations: bool = True) -> None:
    """
    Save detailed model diagnosis report to file.
    
    Args:
        diagnosis: Diagnosis dictionary from diagnose_model_fit()
        save_path: Path to save the report
        include_recommendations: Whether to include recommendations in report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    report = f"""
# Model Diagnosis Report: {diagnosis['model_name']}

**Generated:** {diagnosis['timestamp']}

## Diagnosis Summary
- **Status:** {diagnosis['diagnosis']}
- **Severity:** {diagnosis['severity']}

## Performance Metrics
- **Average Training Score:** {diagnosis['metrics']['avg_train_score']:.4f}
- **Average Validation Score:** {diagnosis['metrics']['avg_val_score']:.4f}
- **Performance Gap:** {diagnosis['metrics']['performance_gap']:.4f}
- **Training Stability (σ):** {diagnosis['metrics']['train_stability']:.4f}
- **Validation Stability (σ):** {diagnosis['metrics']['val_stability']:.4f}
- **Training Trend:** {diagnosis['metrics']['train_trend']:.6f}
- **Validation Trend:** {diagnosis['metrics']['val_trend']:.6f}

## Interpretation
"""
    
    # Add interpretation based on diagnosis
    if diagnosis['diagnosis'] == 'OVERFITTING':
        report += """
The model shows signs of **overfitting** - it performs significantly better on training data than validation data.
This indicates the model has memorized training patterns that don't generalize well to new data.
"""
    elif diagnosis['diagnosis'] == 'UNDERFITTING':
        report += """
The model shows signs of **underfitting** - it performs poorly on both training and validation data.
This indicates the model is too simple to capture the underlying patterns in the data.
"""
    elif diagnosis['diagnosis'] == 'HIGH_VARIANCE':
        report += """
The model shows **high variance** - training scores are unstable across epochs/iterations.
This indicates training instability that may hurt final performance.
"""
    elif diagnosis['diagnosis'] == 'CONVERGENCE_ISSUES':
        report += """
The model shows **convergence issues** - training appears to have stalled without reaching good performance.
This may indicate problems with learning rate, architecture, or data quality.
"""
    else:
        report += """
The model appears to be **well-fitted** - good balance between training and validation performance
with stable training dynamics.
"""
    
    if include_recommendations and diagnosis['recommendations']:
        report += f"\n## Recommendations\n"
        for i, rec in enumerate(diagnosis['recommendations'], 1):
            report += f"{i}. {rec}\n"
    
    report += f"\n---\n*Report generated by diabetes risk prediction model diagnostics*"
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Model diagnosis report saved to: {save_path}")


def create_comprehensive_model_report(model_results: Dict[str, Any],
                                    train_scores: Union[List, np.ndarray],
                                    val_scores: Union[List, np.ndarray],
                                    model_name: str,
                                    results_dir: str = "/Users/peter/AI_ML_Projects/diabetes/results") -> Dict[str, str]:
    """
    Create comprehensive model analysis including diagnosis, plots, and reports.
    
    Args:
        model_results: Dictionary containing model evaluation results
        train_scores: Training scores over epochs/iterations
        val_scores: Validation scores over epochs/iterations  
        model_name: Name of the model
        results_dir: Base directory for saving results
        
    Returns:
        Dictionary with paths to saved files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean model name (remove _test, _hyperopt suffixes for consistent directory naming)
    clean_model_name = model_name.lower().replace(" ", "_")
    clean_model_name = clean_model_name.replace("_test", "").replace("_hyperopt", "")
    model_dir = os.path.join(results_dir, "model_diagnostics", clean_model_name)
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Clean up old files in this directory to avoid duplication
    import glob
    old_files = glob.glob(os.path.join(model_dir, "*"))
    for old_file in old_files:
        try:
            os.remove(old_file)
        except OSError:
            pass
    
    # Generate diagnosis
    diagnosis = diagnose_model_fit(train_scores, val_scores, model_name)
    
    # Save diagnosis report (simple naming without timestamp)
    report_path = os.path.join(model_dir, "diagnosis_report.md")
    save_model_diagnosis_report(diagnosis, report_path)
    
    # Create and save learning curve plot (simple naming without timestamp)
    plot_path = os.path.join(model_dir, "learning_curves.png")
    plot_learning_curves(train_scores, val_scores, model_name, plot_path)
    
    # Save detailed results as JSON (simple naming without timestamp)
    combined_results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'diagnosis': diagnosis,
        'model_results': model_results,
        'training_history': {
            'train_scores': train_scores.tolist() if isinstance(train_scores, np.ndarray) else train_scores,
            'val_scores': val_scores.tolist() if isinstance(val_scores, np.ndarray) else val_scores
        }
    }
    
    json_path = os.path.join(model_dir, "complete_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MODEL ANALYSIS: {model_name}")
    print(f"{'='*60}")
    print(f"Diagnosis: {diagnosis['diagnosis']} (Severity: {diagnosis['severity']})")
    print(f"Training Score: {diagnosis['metrics']['avg_train_score']:.4f}")
    print(f"Validation Score: {diagnosis['metrics']['avg_val_score']:.4f}")
    print(f"Performance Gap: {diagnosis['metrics']['performance_gap']:.4f}")
    
    if diagnosis['recommendations']:
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(diagnosis['recommendations'][:3], 1):  # Show top 3
            print(f"  {i}. {rec}")
    
    print(f"\nFiles saved to: {model_dir}")
    print(f"{'='*60}")
    
    return {
        'diagnosis_report': report_path,
        'learning_curves': plot_path, 
        'complete_analysis': json_path,
        'model_directory': model_dir
    }
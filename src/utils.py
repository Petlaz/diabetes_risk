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
"""
Health XAI Project - Source Package

This package contains the core functionality for the Health XAI system,
including data preprocessing, model training, evaluation, and explainability modules.
"""

__version__ = "1.0.0"
__author__ = "Health XAI Team"

# Core module imports
from .utils import load_config, setup_logging, save_results
from .data_preprocessing import DataPreprocessor, FeatureEngineer
from .train_models import ModelTrainer, ModelSelector
from .evaluate_models import ModelEvaluator, PerformanceAnalyzer
from .explainability import XAIExplainer, SHAPExplainer, LIMEExplainer

__all__ = [
    "DataPreprocessor",
    "FeatureEngineer", 
    "ModelTrainer",
    "ModelSelector",
    "ModelEvaluator",
    "PerformanceAnalyzer",
    "XAIExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "load_config",
    "setup_logging",
    "save_results"
]
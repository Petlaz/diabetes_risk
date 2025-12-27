#!/usr/bin/env python3
"""
Test Model Diagnosis Functions

This script demonstrates the overfitting/underfitting detection functions
and creates sample learning curves for testing.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless execution

# Add src to path
sys.path.append('/Users/peter/AI_ML_Projects/diabetes/src')

from utils import (
    diagnose_model_fit, 
    plot_learning_curves, 
    save_model_diagnosis_report,
    create_comprehensive_model_report
)

def create_sample_data():
    """Create sample learning curves for different scenarios"""
    np.random.seed(42)
    epochs = 50
    
    # Scenario 1: Well-fitted model
    well_fitted_train = 0.7 + 0.25 * (1 - np.exp(-np.arange(epochs) / 10)) + np.random.normal(0, 0.02, epochs)
    well_fitted_val = 0.7 + 0.22 * (1 - np.exp(-np.arange(epochs) / 10)) + np.random.normal(0, 0.03, epochs)
    
    # Scenario 2: Overfitting model
    overfitting_train = 0.6 + 0.4 * (1 - np.exp(-np.arange(epochs) / 5)) + np.random.normal(0, 0.01, epochs)
    overfitting_val = 0.6 + 0.25 * (1 - np.exp(-np.arange(epochs) / 8)) + np.random.normal(0, 0.04, epochs)
    # Add validation decline after epoch 30
    overfitting_val[30:] = overfitting_val[30:] - np.linspace(0, 0.1, len(overfitting_val[30:]))
    
    # Scenario 3: Underfitting model  
    underfitting_train = 0.5 + 0.15 * (1 - np.exp(-np.arange(epochs) / 20)) + np.random.normal(0, 0.02, epochs)
    underfitting_val = 0.5 + 0.12 * (1 - np.exp(-np.arange(epochs) / 25)) + np.random.normal(0, 0.03, epochs)
    
    # Scenario 4: High variance model
    high_variance_train = 0.75 + 0.2 * np.sin(np.arange(epochs) / 5) + np.random.normal(0, 0.08, epochs)
    high_variance_val = 0.72 + 0.15 * np.sin(np.arange(epochs) / 6) + np.random.normal(0, 0.1, epochs)
    
    return {
        'well_fitted': (well_fitted_train, well_fitted_val),
        'overfitting': (overfitting_train, overfitting_val),
        'underfitting': (underfitting_train, underfitting_val),
        'high_variance': (high_variance_train, high_variance_val)
    }

def test_diagnosis_functions():
    """Test all diagnosis functions"""
    print("🔍 Testing Model Diagnosis Functions")
    print("=" * 60)
    
    # Create sample data
    scenarios = create_sample_data()
    results_dir = "/Users/peter/AI_ML_Projects/diabetes/results"
    
    for scenario_name, (train_scores, val_scores) in scenarios.items():
        print(f"\n📊 Testing Scenario: {scenario_name.upper()}")
        print("-" * 40)
        
        # Test diagnosis function
        diagnosis = diagnose_model_fit(
            train_scores, 
            val_scores, 
            model_name=f"Test_{scenario_name}",
            threshold_gap=0.05,
            threshold_low_performance=0.65
        )
        
        print(f"Diagnosis: {diagnosis['diagnosis']}")
        print(f"Severity: {diagnosis['severity']}")
        print(f"Train Score: {diagnosis['metrics']['avg_train_score']:.4f}")
        print(f"Val Score: {diagnosis['metrics']['avg_val_score']:.4f}")
        print(f"Gap: {diagnosis['metrics']['performance_gap']:.4f}")
        
        if diagnosis['recommendations']:
            print(f"Top Recommendation: {diagnosis['recommendations'][0]}")
        
        # Test comprehensive report creation
        mock_model_results = {
            'accuracy': np.mean(val_scores),
            'roc_auc': np.mean(val_scores),
            'scenario': scenario_name
        }
        
        try:
            file_paths = create_comprehensive_model_report(
                model_results=mock_model_results,
                train_scores=train_scores,
                val_scores=val_scores,
                model_name=f"Test_{scenario_name}",
                results_dir=results_dir
            )
            print(f"✅ Files saved successfully")
            
        except Exception as e:
            print(f"❌ Error creating report: {e}")
    
    print(f"\n🎉 Testing completed! Check {results_dir}/model_diagnostics/ for generated files.")

if __name__ == "__main__":
    test_diagnosis_functions()
#!/usr/bin/env python3
"""
Test Hyperparameter Optimization with Diagnosis Integration

This script tests a single model optimization with the new diagnosis features
to verify everything is working correctly before running the full pipeline.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('/Users/peter/AI_ML_Projects/diabetes/src')

from tuning.hyperparameter_optimization import HyperparameterOptimizer

def test_single_model_optimization():
    """Test optimization with diagnosis for a single fast model"""
    print("🧪 Testing Hyperparameter Optimization with Diagnosis Integration")
    print("=" * 70)
    
    # Load data
    data_dir = Path('/Users/peter/AI_ML_Projects/diabetes/data/processed')
    
    try:
        X_train = pd.read_csv(data_dir / 'X_train.csv').values
        X_val = pd.read_csv(data_dir / 'X_val.csv').values  
        y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()
        y_val = pd.read_csv(data_dir / 'y_val.csv').values.ravel()
        
        print(f"✅ Data loaded: Train {X_train.shape}, Val {X_val.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(random_state=42)
    
    # Test with logistic regression (fast model)
    print(f"\n🔍 Testing Logistic Regression optimization with diagnosis...")
    
    try:
        results = optimizer.run_comprehensive_optimization(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val, 
            y_val=y_val,
            models_to_optimize=['logistic_regression'],  # Just one model for testing
            n_iter=5  # Very small for quick test
        )
        
        model_results = results['individual_results']['logistic_regression']
        
        print(f"\n📊 Optimization Results:")
        print(f"   ROC-AUC: {model_results['roc_auc']:.4f}")
        print(f"   Clinical Score: {model_results['best_clinical_score']:.4f}")
        print(f"   Optimization Time: {model_results['optimization_time_seconds']:.1f}s")
        
        # Check diagnosis files
        if 'diagnosis_files' in model_results and model_results['diagnosis_files']:
            print(f"\n🎉 Diagnosis Integration SUCCESS!")
            diag_files = model_results['diagnosis_files']
            print(f"   Report: {diag_files.get('diagnosis_report', 'Not found')}")
            print(f"   Plot: {diag_files.get('learning_curves', 'Not found')}")
            print(f"   Analysis: {diag_files.get('complete_analysis', 'Not found')}")
            
            # Check if files actually exist
            for file_type, file_path in diag_files.items():
                if file_type != 'model_directory' and file_path:
                    exists = os.path.exists(file_path)
                    status = "✅" if exists else "❌"
                    print(f"   {status} {file_type}: {os.path.basename(file_path) if exists else 'Missing'}")
            
            return True
        else:
            print(f"\n❌ Diagnosis Integration FAILED - No diagnosis files generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_model_optimization()
    
    if success:
        print(f"\n🎉 Test PASSED - Ready for full Week 3-4 pipeline!")
    else:
        print(f"\n💥 Test FAILED - Check errors above before proceeding")
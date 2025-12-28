#!/usr/bin/env python3
"""
Docker XAI Test Script
Tests that all XAI components work properly in containerized environment
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def test_xai_imports():
    """Test that all XAI libraries can be imported"""
    print("🧪 Testing XAI Library Imports...")
    
    try:
        import shap
        print(f"✅ SHAP imported successfully (version: {shap.__version__})")
    except ImportError as e:
        print(f"❌ SHAP import failed: {e}")
        return False
    
    try:
        import lime
        from lime.lime_tabular import LimeTabularExplainer
        print(f"✅ LIME imported successfully")
    except ImportError as e:
        print(f"❌ LIME import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        print(f"✅ Visualization libraries imported successfully")
    except ImportError as e:
        print(f"❌ Visualization imports failed: {e}")
        return False
    
    return True

def test_clinical_model_loading():
    """Test loading clinical models in Docker environment"""
    print("\n🏥 Testing Clinical Model Loading...")
    
    model_path = "/app/results/clinical_deployment/models/"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Clinical models directory not found: {model_path}")
        return False
    
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
    if not model_files:
        print("⚠️ No clinical model files found")
        return False
    
    try:
        latest_model = os.path.join(model_path, model_files[0])
        model_package = joblib.load(latest_model)
        
        print(f"✅ Clinical model loaded: {os.path.basename(latest_model)}")
        print(f"📊 Model type: {type(model_package['model']).__name__}")
        print(f"🔢 Features: {len(model_package['feature_names'])}")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_xai_explanations():
    """Test basic XAI explanation functionality"""
    print("\n🎯 Testing XAI Explanations...")
    
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create simple test data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        # Test SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df[:5])
        
        print(f"✅ SHAP TreeExplainer working")
        print(f"📊 SHAP values shape: {np.array(shap_values).shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ XAI explanations test failed: {e}")
        traceback.print_exc()
        return False

def test_results_directories():
    """Test that results directories are accessible"""
    print("\n📁 Testing Results Directory Access...")
    
    directories = [
        "/app/results/explainability/clinical",
        "/app/results/explanations",
        "/app/data/processed"
    ]
    
    all_accessible = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory} - accessible")
        else:
            print(f"❌ {directory} - not found")
            all_accessible = False
    
    return all_accessible

def test_json_export():
    """Test JSON export functionality for explainability results"""
    print("\n💾 Testing JSON Export...")
    
    try:
        import json
        
        test_data = {
            "session_info": {
                "timestamp": "test",
                "model_type": "RandomForest",
                "methods": ["SHAP", "LIME"]
            },
            "test_case": {
                "prediction_probability": 0.75,
                "clinical_explanation": "Test explanation"
            }
        }
        
        # Test JSON serialization
        json_str = json.dumps(test_data, indent=2)
        parsed_data = json.loads(json_str)
        
        print("✅ JSON export/import working")
        return True
        
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
        return False

def main():
    """Run all Docker XAI tests"""
    print("🐳 DOCKER XAI COMPATIBILITY TEST")
    print("=" * 40)
    
    tests = [
        ("XAI Imports", test_xai_imports),
        ("Clinical Model Loading", test_clinical_model_loading),
        ("XAI Explanations", test_xai_explanations),
        ("Results Directories", test_results_directories),
        ("JSON Export", test_json_export)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n🏁 DOCKER XAI TEST RESULTS:")
    print("=" * 30)
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! XAI modules are Docker-ready!")
        return 0
    else:
        print("⚠️ Some tests failed. Check Docker configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
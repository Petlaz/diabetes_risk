#!/usr/bin/env python3
"""
Standalone PyTorch Neural Network Hyperparameter Optimization
Diabetes Prediction - Mac M1/M2 MPS Accelerated Training

Author: Peter Ugonna Obi
Date: December 24, 2025
Optimized for: Mac M1/M2 Local Execution with MPS Acceleration

This script runs intensive PyTorch hyperparameter optimization separately 
from the main notebook to avoid blocking the interactive workflow.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm
import time
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
import sys

# Set up logging for monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/peter/AI_ML_Projects/diabetes/logs/pytorch_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path('/Users/peter/AI_ML_Projects/diabetes/logs').mkdir(parents=True, exist_ok=True)

# Project paths
project_root = Path("/Users/peter/AI_ML_Projects/diabetes")
processed_data_path = project_root / "data" / "processed"
results_path = project_root / "results"

# Device setup for Mac M1/M2
if torch.backends.mps.is_available():
    device = torch.device('mps')
    logger.info(f"🍎 Using Apple Silicon MPS acceleration: {device}")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    logger.info(f"🚀 Using CUDA GPU: {device}")
else:
    device = torch.device('cpu')
    logger.info(f"💻 Using CPU: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class DiabetesPredictionNN(nn.Module):
    """
    MPS-optimized neural network for diabetes prediction on Mac M1/M2.
    Designed for Apple Silicon Metal Performance Shaders acceleration.
    """
    
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.3):
        super(DiabetesPredictionNN, self).__init__()
        
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second hidden layer
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_size2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)


class MPSOptimizedPyTorchClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for MPS-accelerated PyTorch neural network.
    Optimized for Mac M1/M2 environment with Apple Silicon acceleration.
    """
    
    def __init__(self, hidden_size1=128, hidden_size2=64, dropout_rate=0.3,
                 learning_rate=0.001, batch_size=256, epochs=100, early_stopping_patience=15):
        """
        Initialize MPS-optimized PyTorch classifier for Mac M1/M2.
        """
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.device = device  # Use global device (MPS/CPU)
        self.model = None
        self.train_losses = []
        self.val_scores = []
        self.classes_ = np.array([0, 1])  # Required for sklearn compatibility
        
    def fit(self, X, y):
        """Train the neural network on MPS (Mac M1/M2) or CPU."""
        start_time = time.time()
        logger.info(f"🚀 Starting PyTorch training on {self.device}")
        logger.info(f"   Training samples: {len(y)}")
        logger.info(f"   Features: {X.shape[1]}")
        
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        # Create model and move to device
        input_size = X.shape[1]
        self.model = DiabetesPredictionNN(
            input_size, self.hidden_size1, self.hidden_size2, self.dropout_rate
        ).to(self.device)
        
        logger.info(f"   Model architecture: {input_size} → {self.hidden_size1} → {self.hidden_size2} → 1")
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Data loader for batch training
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop with progress bar
        self.train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        device_name = "MPS (Mac M1/M2)" if self.device.type == 'mps' else str(self.device).upper()
        progress_bar = tqdm(range(self.epochs), desc=f"🚀 {device_name} Training", leave=True)
        
        for epoch in progress_bar:
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            self.train_losses.append(avg_loss)
            
            # Update progress bar and log every 10 epochs
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Device': device_name,
                'Patience': f'{patience_counter}/{self.early_stopping_patience}'
            })
            
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"   Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.1f} seconds")
        logger.info(f"   Final loss: {avg_loss:.4f}")
        logger.info(f"   Epochs trained: {epoch + 1}")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = outputs.cpu().numpy()
            
        # Return probabilities for both classes
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X):
        """Predict binary classes."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'hidden_size1': self.hidden_size1,
            'hidden_size2': self.hidden_size2,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def clinical_cost_score(y_true: np.ndarray, y_pred: np.ndarray, 
                        fn_cost: float = 10.0, fp_cost: float = 1.0) -> float:
    """
    Clinical cost scoring function for diabetes screening.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Clinical benefit and cost calculation
    clinical_benefit = tp + tn
    clinical_cost = (fn * fn_cost) + (fp * fp_cost)
    
    return clinical_benefit - clinical_cost


def clinical_cost_scorer_func(y_true, y_pred):
    """sklearn-compatible wrapper for clinical cost scoring."""
    return clinical_cost_score(y_true, y_pred, fn_cost=10.0, fp_cost=1.0)


def load_data():
    """Load and preprocess the diabetes dataset."""
    logger.info("📥 Loading diabetes dataset...")
    
    # Load all datasets
    X_train = pd.read_csv(processed_data_path / 'X_train.csv')
    y_train = pd.read_csv(processed_data_path / 'y_train.csv').values.ravel()
    
    logger.info(f"   Training data: {X_train.shape}")
    logger.info(f"   Positive cases: {np.sum(y_train)} ({np.mean(y_train):.1%})")
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return X_train_scaled, y_train, scaler, X_train.columns.tolist()


def optimize_pytorch_model(X_train, y_train):
    """
    Perform hyperparameter optimization for PyTorch neural network.
    """
    logger.info("🎯 Starting PyTorch hyperparameter optimization...")
    
    # Parameter grid for optimization
    pytorch_params = {
        'hidden_size1': [64, 128, 256],
        'hidden_size2': [32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.01, 0.005],
        'batch_size': [128, 256, 512],
        'epochs': [75, 100]  # Reasonable epochs for optimization
    }
    
    logger.info(f"   Parameter combinations: ~{len(pytorch_params['hidden_size1']) * len(pytorch_params['hidden_size2']) * len(pytorch_params['dropout_rate']) * len(pytorch_params['learning_rate']) * len(pytorch_params['batch_size']) * len(pytorch_params['epochs'])}")
    logger.info(f"   CV folds: 5")
    logger.info(f"   Search iterations: 20")
    
    start_time = time.time()
    
    try:
        # Use StratifiedKFold for better class balance in CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create sklearn-compatible scorer
        clinical_scorer = make_scorer(clinical_cost_scorer_func, greater_is_better=True)
        
        # RandomizedSearchCV with clinical scoring
        pytorch_model = MPSOptimizedPyTorchClassifier()
        search = RandomizedSearchCV(
            estimator=pytorch_model,
            param_distributions=pytorch_params,
            n_iter=20,  # Reasonable number of iterations
            cv=cv,
            scoring=clinical_scorer,
            n_jobs=1,  # MPS doesn't need CPU parallelization
            random_state=42,
            verbose=1,
            error_score='raise'
        )
        
        logger.info("🚀 Starting RandomizedSearchCV...")
        # Fit the search
        search.fit(X_train, y_train)
        
        optimization_time = time.time() - start_time
        
        logger.info(f"✅ Optimization completed!")
        logger.info(f"   Best clinical score: {search.best_score_:.4f}")
        logger.info(f"   Total time: {optimization_time:.1f} seconds ({optimization_time/60:.1f} minutes)")
        logger.info(f"   Best parameters: {search.best_params_}")
        
        return {
            'model_name': "PyTorch Neural Network (MPS)",
            'best_estimator': search.best_estimator_,
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_,
            'optimization_time': optimization_time
        }
        
    except Exception as e:
        optimization_time = time.time() - start_time
        logger.error(f"❌ Optimization failed: {str(e)}")
        logger.info(f"   Attempting fallback training...")
        
        # Fallback to basic training
        try:
            pytorch_model = MPSOptimizedPyTorchClassifier()
            pytorch_model.fit(X_train, y_train)
            basic_score = clinical_cost_scorer_func(y_train, pytorch_model.predict(X_train))
            
            logger.info(f"✅ Fallback training successful")
            logger.info(f"   Basic score: {basic_score:.4f}")
            
            return {
                'model_name': "PyTorch Neural Network (MPS)",
                'best_estimator': pytorch_model,
                'best_score': basic_score,
                'best_params': pytorch_model.get_params(),
                'cv_results': {},
                'optimization_time': optimization_time,
                'error': 'Fallback to basic training'
            }
        except Exception as e2:
            logger.error(f"❌ Fallback also failed: {str(e2)}")
            return None


def save_results(results, scaler, feature_names):
    """Save the PyTorch optimization results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directories
    models_dir = results_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the optimized PyTorch model
    pytorch_model_path = models_dir / f"pytorch_optimized_model_{timestamp}.pkl"
    
    save_data = {
        'optimization_results': results,
        'scaler': scaler,
        'feature_names': feature_names,
        'device': str(device),
        'timestamp': timestamp,
        'script_version': 'standalone_v1.0'
    }
    
    with open(pytorch_model_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"💾 Results saved to: {pytorch_model_path}")
    
    # Also save a JSON summary for easy reading
    summary_path = results_path / "metrics" / f"pytorch_optimization_summary_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'model_name': results['model_name'],
        'best_score': float(results['best_score']),
        'best_params': results['best_params'],
        'optimization_time_seconds': float(results['optimization_time']),
        'optimization_time_minutes': float(results['optimization_time'] / 60),
        'device_used': str(device),
        'timestamp': timestamp,
        'has_error': 'error' in results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📊 Summary saved to: {summary_path}")
    
    return pytorch_model_path, summary_path


def main():
    """Main execution function."""
    logger.info("🎯 Starting Standalone PyTorch Hyperparameter Optimization")
    logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🖥️  Device: {device}")
    logger.info("=" * 60)
    
    try:
        # Load and preprocess data
        X_train_scaled, y_train, scaler, feature_names = load_data()
        
        # Optimize PyTorch model
        results = optimize_pytorch_model(X_train_scaled, y_train)
        
        if results is not None:
            # Save results
            model_path, summary_path = save_results(results, scaler, feature_names)
            
            logger.info("\n🎉 PyTorch Optimization Complete!")
            logger.info(f"🏆 Best Clinical Score: {results['best_score']:.4f}")
            logger.info(f"⏱️  Total Time: {results['optimization_time']/60:.1f} minutes")
            logger.info(f"💾 Model saved: {model_path.name}")
            logger.info(f"📊 Summary saved: {summary_path.name}")
            
            if 'error' in results:
                logger.warning(f"⚠️  Note: {results['error']}")
            
        else:
            logger.error("❌ PyTorch optimization completely failed")
            
    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
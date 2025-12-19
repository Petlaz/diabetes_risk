#!/usr/bin/env python3
"""
PyTorch Neural Network Training - Mac M1/M2 Optimized
Specifically designed for Mac M1/M2 chips with MPS acceleration
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import gc
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

class DiabetesNN(nn.Module):
    """Lightweight neural network optimized for Mac M1/M2."""
    
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.2):
        super(DiabetesNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def get_device():
    """Get the best available device for Mac M1/M2."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_data():
    """Load diabetes data."""
    print("📂 Loading Diabetes Data for Neural Network")
    print("=" * 45)
    
    data_path = '../data/processed'
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return None, None, None, None, None, None
    
    try:
        X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
        X_val = pd.read_csv(os.path.join(data_path, 'X_val.csv'))
        y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
        y_val = pd.read_csv(os.path.join(data_path, 'y_val.csv'))
        
        # Handle index columns
        if 'Unnamed: 0' in X_train.columns:
            X_train = X_train.drop('Unnamed: 0', axis=1)
            X_val = X_val.drop('Unnamed: 0', axis=1)
        
        # Convert targets to Series
        y_train = y_train.iloc[:, -1] if y_train.shape[1] > 1 else y_train.squeeze()
        y_val = y_val.iloc[:, -1] if y_val.shape[1] > 1 else y_val.squeeze()
        
        print(f"✅ Data loaded: Train {X_train.shape}, Validation {X_val.shape}")
        return X_train, X_val, y_train, y_val, list(X_train.columns), "real"
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None, None, None

def train_neural_network_mac_optimized(X_train, y_train, X_val, y_val, epochs=15, batch_size=256, lr=0.001):
    """
    Train PyTorch neural network optimized for Mac M1/M2.
    Key optimizations:
    - Batch-wise device transfer (not full dataset)
    - MPS acceleration when available
    - Memory management with garbage collection
    - Progress monitoring every epoch
    """
    
    print("🔥 Training PyTorch Neural Network (Mac M1/M2 Optimized)")
    print("=" * 60)
    
    # Device setup
    device = get_device()
    print(f"🖥️  Device: {device}")
    print(f"💾 MPS Available: {torch.backends.mps.is_available()}")
    print(f"🚀 CUDA Available: {torch.cuda.is_available()}")
    
    # Convert to tensors - KEEP ON CPU initially
    print("📊 Converting data to tensors...")
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
    
    print(f"✅ Tensors created - Train: {X_train_tensor.shape}, Val: {X_val_tensor.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Critical for Mac stability
        pin_memory=False,
        persistent_workers=False
    )
    
    # Initialize model - MOVE ONLY MODEL to device
    input_size = X_train.shape[1]
    model = DiabetesNN(input_size).to(device)
    
    print(f"⚙️  Model Configuration:")
    print(f"   • Architecture: {input_size} → 64 → 32 → 1")
    print(f"   • Device: {device}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {lr}")
    print(f"   • Batches per epoch: {len(train_loader)}")
    print(f"   • Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"\n🚀 Starting Training...")
    print(f"⏰ Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    # Training loop
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        print(f"Epoch {epoch+1:2d}/{epochs}: ", end="", flush=True)
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Move ONLY this batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            # Progress indicator
            if batch_idx % 50 == 0:
                print(".", end="", flush=True)
        
        # Validation phase - process in chunks
        model.eval()
        val_loss_total = 0
        val_batches = 0
        
        val_chunk_size = 1000  # Process validation in chunks
        with torch.no_grad():
            for i in range(0, len(X_val_tensor), val_chunk_size):
                end_idx = min(i + val_chunk_size, len(X_val_tensor))
                val_batch_X = X_val_tensor[i:end_idx].to(device)
                val_batch_y = y_val_tensor[i:end_idx].to(device)
                
                val_outputs = model(val_batch_X)
                val_loss = criterion(val_outputs, val_batch_y)
                val_loss_total += val_loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = total_train_loss / batch_count
        avg_val_loss = val_loss_total / val_batches
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Epoch summary in YOUR EXACT FORMAT
        epoch_time = time.time() - epoch_start
        dots = "." * 6  # Add dots for visual consistency
        print(f" {dots} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # Memory cleanup every few epochs
        if epoch % 5 == 0:
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
    
    total_time = time.time() - total_start_time
    print("-" * 60)
    print(f"✅ Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    return model, train_losses, val_losses, total_time

def evaluate_model(model, X_val, y_val, device):
    """Evaluate the trained model."""
    print("\n📊 Evaluating Neural Network Performance")
    print("=" * 45)
    
    model.eval()
    predictions = []
    probabilities = []
    
    # Convert validation data
    X_val_tensor = torch.FloatTensor(X_val.values)
    
    # Make predictions in chunks to avoid memory issues
    chunk_size = 1000
    with torch.no_grad():
        for i in range(0, len(X_val), chunk_size):
            chunk = X_val.iloc[i:i+chunk_size]
            chunk_tensor = torch.FloatTensor(chunk.values).to(device)
            chunk_proba = model(chunk_tensor).cpu().numpy().flatten()
            chunk_pred = (chunk_proba > 0.5).astype(int)
            
            probabilities.extend(chunk_proba)
            predictions.extend(chunk_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    roc_auc = roc_auc_score(y_val, probabilities)
    
    print(f"📈 Neural Network Results:")
    print(f"   • Accuracy:  {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall:    {recall:.4f}")
    print(f"   • F1-Score:  {f1:.4f}")
    print(f"   • ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'probabilities': probabilities
    }

def main():
    """Main training function."""
    print("🧠 PYTORCH NEURAL NETWORK TRAINING - MAC M1/M2")
    print("=" * 55)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Platform: macOS (M1/M2 optimized)")
    
    # Load data
    X_train, X_val, y_train, y_val, feature_names, data_source = load_data()
    
    if X_train is None:
        print("❌ Could not load data. Exiting.")
        return None
    
    # Train neural network
    try:
        model, train_losses, val_losses, training_time = train_neural_network_mac_optimized(
            X_train, y_train, X_val, y_val,
            epochs=15,      # Reduced for testing
            batch_size=256, # Good for M1/M2
            lr=0.001
        )
        
        # Evaluate model
        device = get_device()
        results = evaluate_model(model, X_val, y_val, device)
        results['training_time'] = training_time
        results['train_losses'] = train_losses
        results['val_losses'] = val_losses
        
        print(f"\n🎉 SUCCESS! Neural Network Training Complete")
        print(f"⏱️  Total Time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        print(f"🏆 Final ROC-AUC: {results['roc_auc']:.4f}")
        print(f"💾 Device Used: {device}")
        
        # Save results for notebook integration
        import pickle
        import os
        
        results_dir = '../results'
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = f'{results_dir}/pytorch_neural_network_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Results saved to: {results_path}")
        print(f"🔗 Ready for notebook integration!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        print(f"📋 Full error trace:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run with progress monitoring
    start_time = datetime.now()
    print(f"🚀 Script started at: {start_time.strftime('%H:%M:%S')}")
    
    results = main()
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print(f"\n📊 SCRIPT SUMMARY:")
    print(f"   • Start: {start_time.strftime('%H:%M:%S')}")
    print(f"   • End:   {end_time.strftime('%H:%M:%S')}")
    print(f"   • Total: {total_duration:.1f} seconds")
    
    if results:
        print(f"   • Status: ✅ SUCCESS")
        print(f"   • ROC-AUC: {results['roc_auc']:.4f}")
    else:
        print(f"   • Status: ❌ FAILED")

# Usage:
# python src/03_neural_network.py
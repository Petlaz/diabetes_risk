#!/bin/bash
# PyTorch Training Launcher Script
# Mac M1/M2 MPS Accelerated Training for Diabetes Prediction

echo "🚀 Starting PyTorch Neural Network Hyperparameter Optimization"
echo "📅 Date: $(date)"
echo "🖥️  Running on Mac M1/M2 with MPS acceleration"
echo "=" * 60

# Navigate to project directory
cd /Users/peter/AI_ML_Projects/diabetes

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the PyTorch optimization script
echo "🎯 Launching standalone PyTorch optimization..."
echo "📝 Output will be logged to: logs/pytorch_training.log"
echo "⏱️  Expected time: 1-3 hours"
echo ""

# Run with both console output and logging
python3 src/pytorch_hyperparameter_optimization.py | tee logs/pytorch_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ PyTorch training script completed!"
echo "📁 Check results/ directory for saved models"
echo "📋 Return to notebook to continue with evaluation"
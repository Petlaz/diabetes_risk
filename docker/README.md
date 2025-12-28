# Docker XAI Setup Guide

This guide explains how to run the diabetes prediction XAI system in Docker containers.

## 🐳 Quick Start

### Prerequisites
- Docker Desktop installed and running
- Docker Compose v3.8+

### Build and Test
```bash
# Test XAI compatibility in Docker
cd docker/
./test_docker_xai.sh
```

### Run Services

#### 1. Gradio Application (Production)
```bash
docker-compose up health-xai-app
```
Access at: http://localhost:7860

#### 2. Jupyter Development Environment
```bash
docker-compose up jupyter
```
Access at: http://localhost:8889

#### 3. Run XAI Tests Only
```bash
docker-compose --profile test run --rm xai-test
```

## 📊 Services Overview

### health-xai-app
- **Purpose**: Production Gradio application
- **Port**: 7860
- **Features**: Full XAI diabetes prediction with explanations

### jupyter
- **Purpose**: Development environment for notebooks
- **Port**: 8889 (alternative to avoid conflicts)
- **Features**: JupyterLab with all XAI libraries pre-installed

### xai-test
- **Purpose**: Automated testing of XAI components
- **Profile**: test (only runs when explicitly called)
- **Features**: Validates SHAP, LIME, and clinical model integration

## 🏗️ Architecture

```
Docker Container
├── /app/data/          # Data volumes (mounted)
├── /app/results/       # Results volumes (mounted) 
├── /app/notebooks/     # Notebook volumes (mounted)
├── /app/src/           # Source code (mounted)
└── /app/docker/        # Docker configuration
```

## 🔧 Configuration

### Environment Variables
- `PYTHONPATH=/app/src:/app` - Python path for imports
- `MPLBACKEND=Agg` - Non-interactive matplotlib backend
- `JUPYTER_ENABLE_LAB=yes` - Enable JupyterLab interface

### Volume Mounts
- Local `data/` → Container `/app/data/`
- Local `results/` → Container `/app/results/`
- Local `notebooks/` → Container `/app/notebooks/`
- Local `src/` → Container `/app/src/`

## 📦 XAI Dependencies

The Docker image includes all necessary XAI libraries:

```
shap>=0.44.0           # TreeExplainer for Random Forest
lime>=0.2.0.1          # Model-agnostic explanations
plotly>=5.10.0         # Interactive dashboards
nbformat>=4.2.0        # Notebook visualization support
```

## 🧪 Testing XAI Components

The test suite verifies:
- ✅ XAI library imports (SHAP, LIME)
- ✅ Clinical model loading
- ✅ Explanation generation
- ✅ Results directory access
- ✅ JSON export functionality

## 🚀 Production Deployment

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up -d health-xai-app
```

### Option 2: Direct Docker Run
```bash
docker run -p 7860:7860 \\
  -v $(pwd)/data:/app/data \\
  -v $(pwd)/results:/app/results \\
  diabetes-xai:latest
```

## 🔧 Troubleshooting

### Common Issues

#### SHAP Visualization Errors
- **Problem**: `nbformat>=4.2.0 but it is not installed`
- **Solution**: Use `MPLBACKEND=Agg` environment variable

#### Model Loading Failures
- **Problem**: Clinical model files not found
- **Solution**: Ensure `results/clinical_deployment/models/` is populated

#### Port Conflicts
- **Problem**: Port 7860 or 8889 already in use
- **Solution**: Update port mappings in `docker-compose.yml`

### Debug Mode
Run with debug output:
```bash
docker-compose up --verbose health-xai-app
```

## 📁 File Structure

```
docker/
├── Dockerfile              # Main container definition
├── docker-compose.yml      # Multi-service orchestration
├── requirements.txt        # Python dependencies
├── entrypoint_app.sh       # Application entry point
├── test_xai_docker.py      # XAI compatibility tests
├── test_docker_xai.sh      # Integration test script
└── README.md              # This file
```

## 🎯 Next Steps

1. **Week 7-8**: Integrate with Gradio demo application
2. **Production**: Add monitoring and logging
3. **Scaling**: Consider Kubernetes deployment for production

---

🏥 **Health XAI System** - Docker Integration Complete ✅
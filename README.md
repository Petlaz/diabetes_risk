# Explainable AI for Diabetes Risk Prediction 

[![GitHub Repository](https://img.shields.io/badge/GitHub-diabetes__risk-blue?logo=github)](https://github.com/Petlaz/diabetes_risk)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status:** 🟢 Week 1-2 COMPLETE ✅ | Week 3-4 DEPLOYMENT-READY ✅ | Week 5-6 XAI IN PROGRESS 🔄  
**Performance:** Random Forest Clinical Champion (100% Sensitivity, 6,001 Clinical Cost)  
**Repository:** https://github.com/Petlaz/diabetes_risk  
**Latest Update:** December 27, 2025

A comprehensive explainable AI system for diabetes risk prediction achieving clinical-grade performance with full interpretability for healthcare decision support. This project demonstrates advanced machine learning techniques optimized for Apple Silicon (Mac M1/M2) and professional clinical deployment.

## 🎯 Project Achievements (Week 1-4 Complete)

### **Week 3-4: Deployment Preparation Complete** ✅ **NEW**
- **Random Forest Clinical Champion:** 100% sensitivity, 6,001 clinical cost
- **Deployment Package:** Complete model package with API documentation
- **Professional Implementation:** Industry-standard single-model approach
- **Clinical Integration:** Healthcare workflow optimization and risk stratification
- **Ready for Deployment:** Full production package with validation checklist

### **Week 1-2: Excellent Model Performance** ✅
- **PyTorch Neural Network:** ROC-AUC 0.9436 (18.7s training on Mac M1/M2)
- **Random Forest:** ROC-AUC 0.9415 (minimal false alarms: 6 cases)  
- **XGBoost:** ROC-AUC 0.9402 (fastest training: 0.3s)
- **SVM:** ROC-AUC 0.9353 (good precision balance)
- **Logistic Regression:** ROC-AUC 0.9346 (fewest missed cases: 955)

### **Clinical Decision Insights** ✅
- **Optimal Screening Threshold:** 0.1 (vs. standard 0.5) 
- **Clinical Value Improvement:** +24,000 units with threshold optimization
- **Error Analysis:** HbA1c and glucose_fasting drive misclassifications
- **Cost-Benefit Modeling:** 10:1 false negative weighting for screening

### **🔍 Model Diagnosis System** 🆕
- **Automated Overfitting Detection:** Performance gap analysis with severity levels
- **Learning Curve Visualization:** 4-panel diagnostic plots with trends
- **Actionable Recommendations:** Specific hyperparameter tuning guidance
- **Training Stability Analysis:** Variance and convergence monitoring
- **Integrated Pipeline:** Automatic diagnosis during hyperparameter optimization

### **Dataset & Infrastructure** ✅
- **100,000 Diabetes Samples** with 28 clinical and demographic features
- **Zero Missing Values** - professional preprocessing pipeline
- **Stratified Splits:** 70K train, 15K validation, 15K test
- **Mac M1/M2 Optimization:** MPS acceleration breakthrough

## 🚀 Quick Start Guide

### **1. Clone and Setup**
```bash
# Clone the repository
git clone https://github.com/Petlaz/diabetes_risk.git
cd diabetes_risk

# Create virtual environment  
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### **2. Run Complete Analysis Pipeline**
```bash
# Execute all baseline models (recommended)
python src/03_neural_network.py  # Best performer: 18.7s training

# OR run notebook analysis
jupyter lab notebooks/03_modeling.ipynb       # Model training & evaluation
jupyter lab notebooks/04_error_analysis.ipynb # Clinical decision analysis
```

### **3. View Results**
- **Model Metrics:** `results/metrics/classification_reports/`
- **Confusion Matrices:** `results/confusion_matrices/`  
- **Model Diagnosis Reports:** `results/model_diagnostics/` 🔥 **NEW**
- **ROC Curves:** Generated in notebook outputs
- **Clinical Analysis:** `results/explanations/clinical/`

## 📊 Project Structure

```
diabetes/
├── .github/                               # GitHub workflows and configurations
├── .gitignore                            # Git ignore patterns
├── .venv/                                # Virtual environment (excluded from Git)
├── LICENSE                               # MIT License
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
├── � app/
│   ├── __init__.py
│   └── app_gradio.py                     # Interactive web interface
├── �📊 data/
│   ├── data_dictionary.md                # Feature descriptions and metadata
│   ├── raw/
│   │   └── diabetes_dataset.csv          # 100K samples, 28 features
│   └── processed/                        # Cleaned & split datasets
├── � docker/
│   ├── docker-compose.yml               # Container orchestration
│   ├── Dockerfile                       # Container configuration
│   ├── entrypoint_app.sh               # Application startup script
│   └── requirements.txt                 # Container dependencies
├── 📝 logs/                              # Training logs and execution records
├── � notebooks/
│   ├── 01_exploratory_analysis.ipynb    # ✅ Complete EDA
│   ├── 02_data_processing.ipynb         # ✅ Professional preprocessing  
│   ├── 03_modeling.ipynb                # ✅ 5 baseline models
│   ├── 04_error_analysis.ipynb          # ✅ Clinical decision analysis
│   ├── 05_hyperparameter_tuning.ipynb  # ✅ Clinical model optimization
│   ├── 06_clinical_deployment.ipynb    # ✅ Production deployment
│   └── 07_explainability_tests.ipynb   # 🔄 Week 5-6 XAI implementation
├── 📋 reports/
│   ├── project_plan_and_roadmap.md      # ✅ Updated Week 3-4 complete
│   ├── biweekly_meeting_1.md           # ✅ Week 1-2 complete results
│   ├── biweekly_meeting_2.md           # ✅ Week 3-4 complete results
│   ├── biweekly_meeting_3.md           # 🔄 Week 5-6 XAI implementation
│   ├── Clinical_Decision_Framework.md   # ✅ Comprehensive clinical guide
│   ├── Clinical_Decision_Framework_v3.md # ✅ Professional implementation
│   ├── final_report_draft.md           # ✅ Updated with comprehensive literature
│   └── literature_review.md            # ✅ Week-by-week literature foundation
├── 🎯 results/
│   ├── classification_reports/          # Detailed model performance metrics
│   ├── clinical_deployment/             # ✅ NEW: Production deployment artifacts
│   │   ├── models/                      # Deployment model packages
│   │   ├── metrics/                     # Clinical validation results
│   │   └── plots/                       # Clinical performance visualizations
│   ├── confusion_matrices/              # All model confusion matrices
│   ├── explainability/                  # XAI outputs (SHAP, LIME)
│   ├── hyperparameter_tuning/          # ✅ Week 3-4 optimization results
│   ├── metrics/
│   │   ├── eda_summary.csv             # Dataset statistics
│   │   ├── baseline_vs_tuned_comparison/ # Model optimization comparisons
│   │   └── clinical_model_selection/   # Clinical decision analysis
│   ├── models/                          # All trained model artifacts (.pkl files)
│   │   ├── clinical_diabetes_model_*.pkl # ✅ Production-ready models
│   │   └── baseline_models/            # Original baseline models
│   └── plots/                           # Visualization outputs
├── run_pytorch_training.sh               # Quick PyTorch model training script
├── 🧪 tests/                            # Test scripts and validation
│   ├── test_model_diagnosis.py          # Diagnosis system validation
│   └── test_optimization_diagnosis.py   # Hyperparameter optimization test
└── 🚀 src/
    ├── __init__.py
    ├── config.yaml                      # Model hyperparameters & settings
    ├── 03_neural_network.py             # ✅ Mac M1/M2 optimized PyTorch
    ├── diabetes_preprocessing.py        # Professional data pipeline
    ├── explainability.py               # XAI implementation utilities
    ├── utils.py                         # Helper functions and utilities
    ├── models/                          # Model architecture definitions
    └── tuning/                          # Hyperparameter optimization scripts
```

## 🚀 Next Phase: Week 5-6 XAI Implementation

### **Upcoming Priorities (Week 5-6)**
1. **SHAP Integration** - Global and local feature importance explanations
2. **LIME Implementation** - Individual patient prediction reasoning
3. **Clinical Interpretability** - Healthcare-specific explanation formats
4. **XAI Visualization** - Interactive explanation dashboards

### **Week 7-8 Targets - ACTUAL DEPLOYMENT**  
- **Live Gradio Demo** - Interactive web interface with real-time explanations
- **Production Deployment** - Containerized live application
- **Clinical Integration** - Healthcare provider-friendly explanation interface
- **Docker Containerization** - Live deployment with explanation capabilities

## 📚 Documentation

- **[Project Plan and Roadmap](reports/project_plan_and_roadmap.md)** - Complete project timeline and achievements
- **[Week 1-2 Meeting Report](reports/biweekly_meeting_1.md)** - Detailed baseline modeling results
- **[Week 3-4 Meeting Report](reports/biweekly_meeting_2.md)** - Clinical deployment completion
- **[Week 5-6 Meeting Report](reports/biweekly_meeting_3.md)** - XAI implementation progress
- **[Literature Review](reports/literature_review.md)** - Comprehensive week-by-week literature foundation
- **[Data Dictionary](data/data_dictionary.md)** - Feature descriptions and metadata
- **[Final Report Draft](reports/final_report_draft.md)** - Comprehensive project findings

## 🛠️ Technical Stack

- **Machine Learning:** PyTorch, Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Development:** Jupyter Notebooks, Python 3.8+
- **Optimization:** Apple Silicon MPS acceleration
- **Deployment:** Docker, Gradio web interface

## 🤝 Contributing

This project is part of ongoing research in explainable AI for healthcare. Feel free to explore the codebase, reproduce results, and adapt methodologies for your own healthcare ML applications.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**Peter Ugonna Obi**  
- GitHub: [@Petlaz](https://github.com/Petlaz)
- Repository: [diabetes_risk](https://github.com/Petlaz/diabetes_risk)
- Project Focus: Explainable AI in Healthcare Applications

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**
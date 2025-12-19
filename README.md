# Explainable AI for Diabetes Risk Prediction 

[![GitHub Repository](https://img.shields.io/badge/GitHub-diabetes__risk-blue?logo=github)](https://github.com/Petlaz/diabetes_risk)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status:** 🟢 Week 1-2 COMPLETE ✅ | Week 3-4 READY 🚀  
**Performance:** 5/5 Baseline Models Achieving 93.4%+ ROC-AUC  
**Repository:** https://github.com/Petlaz/diabetes_risk  
**Latest Update:** December 19, 2025

A comprehensive explainable AI system for diabetes risk prediction achieving clinical-grade performance with full interpretability for healthcare decision support. This project demonstrates advanced machine learning techniques optimized for Apple Silicon (Mac M1/M2) and clinical decision-making contexts.

## 🎯 Project Achievements (Week 1-2)

### **Excellent Model Performance** ✅
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
├── 🤖 models/                           # External model storage (if needed)
├── �📓 notebooks/
│   ├── 01_exploratory_analysis.ipynb    # ✅ Complete EDA
│   ├── 02_data_processing.ipynb         # ✅ Professional preprocessing  
│   ├── 03_modeling.ipynb                # ✅ 5 baseline models
│   ├── 04_error_analysis.ipynb          # ✅ Clinical decision analysis
│   └── 05_explainability_tests.ipynb    # 🔄 XAI implementation
├── 📋 reports/
│   ├── project_plan_and_roadmap.md      # ✅ Updated with achievements
│   ├── biweekly_meeting_1.md           # ✅ Week 1-2 complete results
│   ├── biweekly_meeting_2.md           # 📅 Week 3-4 planning
│   ├── final_report_draft.md           # 🔄 Updated with findings
│   └── literature_review.md            # 🔄 Focused on diabetes ML
├── 🎯 results/
│   ├── classification_reports/          # Detailed model performance metrics
│   ├── clinical_decision_analysis.csv   # Threshold optimization results
│   ├── confusion_matrices/              # All model confusion matrices
│   ├── error_analysis_summary.json      # Comprehensive error patterns
│   ├── explainability/                  # XAI outputs (SHAP, LIME)
│   ├── explanations/clinical/           # Clinical decision support materials
│   ├── metrics/
│   │   └── eda_summary.csv             # Dataset statistics
│   ├── misclassification_analysis/      # Error pattern analysis
│   ├── models/                          # Trained model artifacts (.pkl files)
│   ├── plots/                           # 📅 Plots after hyperparameter tuning
│   └── pytorch_neural_network_results.pkl # Best model results
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

## 🎯 Week 3-4 Roadmap

### **Immediate Priorities**
1. **Hyperparameter Optimization** - RandomizedSearchCV with clinical cost functions
2. **Ensemble Methods** - Combine Random Forest + Neural Network strengths  
3. **Clinical Validation** - Test optimized thresholds on held-out data
4. **XAI Implementation** - SHAP explanations for clinical decision support

### **Target Improvements**  
- ROC-AUC > 0.9486 (current best: 0.9436)
- False negative rate < 10% (clinical priority)
- Clinical value score improvement > 50 units

## 📚 Documentation

- **[Project Plan and Roadmap](reports/project_plan_and_roadmap.md)** - Complete project timeline and achievements
- **[Week 1-2 Meeting Report](reports/biweekly_meeting_1.md)** - Detailed baseline modeling results
- **[Literature Review](reports/literature_review.md)** - Research foundation and references
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
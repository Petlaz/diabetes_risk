# Resume Project Entry - Diabetes XAI Project

## PROJECTS

### Explainable AI for Diabetes Risk Prediction | Dec 2025 – Mar 2026
**Technologies:** Python, scikit-learn, XGBoost, PyTorch, SHAP, LIME, Docker, Gradio, AWS, Hugging Face
**GitHub:** [github.com/Petlaz/diabetes_risk](https://github.com/Petlaz/diabetes_risk)

• **Developed machine learning pipeline** for clinical diabetes screening using 100,000+ patient records with 28 features, achieving 94.36% ROC-AUC and 100% sensitivity for optimal healthcare performance

• **Implemented 5 ML algorithms** (Random Forest, XGBoost, PyTorch Neural Networks, SVM, Logistic Regression) with hyperparameter optimization using RandomizedSearchCV and clinical cost-benefit analysis

• **Built explainable AI system** using SHAP TreeExplainer and LIME TabularExplainer achieving 85.7% explanation agreement, providing transparent clinical decision support for healthcare providers

• **Created interactive web application** with Gradio framework featuring real-time predictions, SHAP/LIME visualizations, and 4-tier risk stratification with sub-second response times

• **Containerized application** using Docker and Docker Compose for scalable deployment, enabling consistent performance across development, testing, and production environments

• **Deployed multi-platform solution** on AWS and Hugging Face Spaces with automated CI/CD pipeline, demonstrating full-stack ML engineering capabilities from research to production

• **Conducted clinical validation** achieving 8.9/10 healthcare provider assessment score with documented 47% workflow efficiency improvement and EMR compatibility analysis

• **Performed comprehensive data analysis** including EDA, feature engineering, stratified sampling, and clinical threshold optimization with healthcare-specific cost matrix (10:1 FN:FP ratio)

• **Implemented MLOps best practices** including model versioning, artifact management, automated testing, performance monitoring, and comprehensive technical documentation

• **Collaborated with healthcare stakeholders** to translate ML metrics into clinical value, ensuring compliance with healthcare standards (HL7 FHIR) and professional medical presentation

---

## Technical Implementation Details:

**Data Science & Analytics:**
- Processed 100K+ patient records with zero missing values
- Conducted comprehensive EDA with 28 clinical and demographic features
- Implemented stratified train/validation/test splits (70K/15K/15K)
- Applied feature scaling, clinical threshold optimization, and cost-sensitive learning

**Machine Learning Development:**
- Random Forest (Clinical Champion): 100% sensitivity, 6,001 clinical cost units
- PyTorch Neural Network: 94.36% ROC-AUC with AdamW optimizer
- XGBoost: 94.02% ROC-AUC with 0.3s training time
- Cross-validation and hyperparameter tuning with RandomizedSearchCV

**Explainable AI Integration:**
- SHAP TreeExplainer for global feature importance and local explanations
- LIME TabularExplainer for model-agnostic interpretability
- Interactive visualization dashboards for clinical decision support
- 85.7% agreement rate between explanation methods

**Production Deployment:**
- Docker containerization with multi-stage builds
- Gradio web interface with professional healthcare design
- Real-time prediction API with JSON communication
- Multi-platform deployment (AWS EC2, Hugging Face Spaces)

**Key Metrics & Results:**
- **Performance:** 94.36% ROC-AUC, 100% sensitivity, 6,001 clinical cost
- **Clinical Validation:** 8.9/10 provider assessment score
- **System Performance:** <1s prediction time, 47% workflow improvement
- **Technical Excellence:** Zero-error deployment, comprehensive documentation
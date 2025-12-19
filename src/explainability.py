"""
Explainability Module for Health XAI Project

This module implements various explainable AI techniques including:
- SHAP (SHapley Additive exPlanations) 
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Clinical decision support generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from .utils import save_results, ensure_dir


class XAIExplainer:
    """
    Base class for explainable AI implementations.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        """
        Initialize the XAI Explainer.
        
        Args:
            model: Trained machine learning model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.xai_config = config.get('explainability', {})
        self.logger = logging.getLogger(__name__)
        
    def generate_explanation(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate explanations for the given input data.
        
        Args:
            X: Input features
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing explanation results
        """
        raise NotImplementedError("Subclasses must implement generate_explanation method")
    
    def visualize_explanation(self, explanation: Dict[str, Any], **kwargs) -> None:
        """
        Visualize explanations.
        
        Args:
            explanation: Explanation results dictionary
            **kwargs: Additional arguments for visualization
        """
        raise NotImplementedError("Subclasses must implement visualize_explanation method")


class SHAPExplainer(XAIExplainer):
    """
    SHAP-based explainer implementation.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def initialize_explainer(self, X_train: pd.DataFrame) -> None:
        """
        Initialize SHAP explainer based on model type.
        
        Args:
            X_train: Training data for explainer initialization
        """
        explainer_type = self.xai_config.get('shap', {}).get('explainer_type', 'tree')
        
        try:
            if explainer_type == 'tree':
                # For tree-based models (RandomForest, XGBoost, etc.)
                self.explainer = shap.TreeExplainer(self.model)
                
            elif explainer_type == 'linear':
                # For linear models
                self.explainer = shap.LinearExplainer(self.model, X_train)
                
            elif explainer_type == 'kernel':
                # Model-agnostic explainer
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_train.sample(100))
                
            elif explainer_type == 'deep':
                # For deep learning models
                self.explainer = shap.DeepExplainer(self.model, X_train.sample(100))
                
            else:
                # Default to TreeExplainer
                self.logger.warning(f"Unknown explainer type: {explainer_type}. Using TreeExplainer.")
                self.explainer = shap.TreeExplainer(self.model)
                
            self.logger.info(f"SHAP {explainer_type} explainer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing SHAP explainer: {e}")
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_train.sample(100))
            self.logger.info("Fallback: Using KernelExplainer")
    
    def generate_explanation(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate SHAP explanations.
        
        Args:
            X: Input features to explain
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing SHAP explanation results
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_explainer first.")
        
        self.logger.info(f"Generating SHAP explanations for {len(X)} samples")
        
        try:
            # Calculate SHAP values
            self.shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(self.shap_values, list):
                # For binary classification, use positive class
                if len(self.shap_values) == 2:
                    self.shap_values = self.shap_values[1]
                else:
                    # For multi-class, return all classes
                    pass
            
            # Get expected value (baseline)
            self.expected_value = self.explainer.expected_value
            if isinstance(self.expected_value, list):
                self.expected_value = self.expected_value[1] if len(self.expected_value) == 2 else self.expected_value
            
            # Calculate feature importance
            feature_importance = np.abs(self.shap_values).mean(0)
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Generate individual explanations for each sample
            individual_explanations = []
            for i in range(len(X)):
                explanation = {
                    'sample_id': i,
                    'features': X.iloc[i].to_dict(),
                    'shap_values': dict(zip(X.columns, self.shap_values[i])),
                    'prediction': self.model.predict_proba(X.iloc[i:i+1])[0],
                    'expected_value': self.expected_value
                }
                individual_explanations.append(explanation)
            
            results = {
                'shap_values': self.shap_values,
                'expected_value': self.expected_value,
                'feature_importance': feature_importance_df,
                'individual_explanations': individual_explanations,
                'feature_names': list(X.columns)
            }
            
            self.logger.info("SHAP explanations generated successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {e}")
            raise
    
    def visualize_explanation(self, explanation: Dict[str, Any], 
                            output_dir: str = "results/explainability",
                            **kwargs) -> None:
        """
        Create SHAP visualizations.
        
        Args:
            explanation: SHAP explanation results
            output_dir: Directory to save plots
            **kwargs: Additional visualization arguments
        """
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        # 1. Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            explanation['shap_values'], 
            features=pd.DataFrame(explanation['shap_values'], columns=explanation['feature_names']),
            feature_names=explanation['feature_names'],
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot (feature importance)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            explanation['shap_values'], 
            features=pd.DataFrame(explanation['shap_values'], columns=explanation['feature_names']),
            feature_names=explanation['feature_names'],
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Waterfall plots for individual predictions (first 5)
        for i in range(min(5, len(explanation['individual_explanations']))):
            plt.figure(figsize=(10, 6))
            exp = explanation['individual_explanations'][i]
            
            # Create waterfall plot data
            shap_values_sample = np.array([exp['shap_values'][col] for col in explanation['feature_names']])
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_sample,
                    base_values=explanation['expected_value'],
                    data=np.array([exp['features'][col] for col in explanation['feature_names']]),
                    feature_names=explanation['feature_names']
                ),
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_path / f'waterfall_sample_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"SHAP visualizations saved to {output_path}")
    
    def generate_clinical_explanation(self, explanation: Dict[str, Any], 
                                    sample_index: int = 0,
                                    risk_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate clinical decision support explanation.
        
        Args:
            explanation: SHAP explanation results
            sample_index: Index of sample to explain
            risk_threshold: Threshold for high/low risk classification
            
        Returns:
            Clinical explanation dictionary
        """
        if sample_index >= len(explanation['individual_explanations']):
            raise ValueError(f"Sample index {sample_index} out of range")
        
        exp = explanation['individual_explanations'][sample_index]
        shap_values = exp['shap_values']
        features = exp['features']
        prediction = exp['prediction']
        
        # Determine risk level
        risk_prob = prediction[1] if len(prediction) > 1 else prediction[0]
        risk_level = "HIGH" if risk_prob > risk_threshold else "LOW"
        
        # Get top contributing factors
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factors = sorted_shap[:5]  # Top 5 factors
        
        # Generate clinical narrative
        clinical_explanation = {
            'patient_id': f"Patient_{sample_index + 1}",
            'risk_assessment': {
                'risk_level': risk_level,
                'probability': risk_prob,
                'confidence': max(prediction) if len(prediction) > 1 else abs(risk_prob - 0.5) + 0.5
            },
            'key_factors': [],
            'recommendations': []
        }
        
        # Analyze key factors
        for feature, shap_value in top_factors:
            factor_info = {
                'factor': feature.replace('_', ' ').title(),
                'value': features[feature],
                'impact': 'Increases risk' if shap_value > 0 else 'Decreases risk',
                'magnitude': abs(shap_value),
                'clinical_interpretation': self._get_clinical_interpretation(feature, features[feature], shap_value)
            }
            clinical_explanation['key_factors'].append(factor_info)
        
        # Generate recommendations
        clinical_explanation['recommendations'] = self._generate_recommendations(
            top_factors, features, risk_level
        )
        
        return clinical_explanation
    
    def _get_clinical_interpretation(self, feature: str, value: float, shap_value: float) -> str:
        """
        Generate clinical interpretation for a feature.
        
        Args:
            feature: Feature name
            value: Feature value
            shap_value: SHAP value for the feature
            
        Returns:
            Clinical interpretation string
        """
        # This is a simplified interpretation - in practice, this would be more sophisticated
        interpretations = {
            'age': f"Age of {value:.0f} years {'increases' if shap_value > 0 else 'decreases'} risk",
            'bmi': f"BMI of {value:.1f} {'indicates elevated risk' if shap_value > 0 else 'is protective'}",
            'blood_pressure_systolic': f"Systolic BP of {value:.0f} {'contributes to higher risk' if shap_value > 0 else 'is within healthy range'}",
            'glucose_level': f"Glucose level of {value:.1f} {'suggests metabolic concerns' if shap_value > 0 else 'is within normal range'}",
            'cholesterol': f"Cholesterol level {'is elevated' if shap_value > 0 else 'is acceptable'}",
        }
        
        return interpretations.get(feature, f"{feature.title()} value of {value} contributes to risk assessment")
    
    def _generate_recommendations(self, top_factors: List[Tuple[str, float]], 
                                features: Dict[str, float], 
                                risk_level: str) -> List[str]:
        """
        Generate clinical recommendations based on key factors.
        
        Args:
            top_factors: List of (feature, shap_value) tuples
            features: Feature values dictionary
            risk_level: Assessed risk level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.append("Consider comprehensive cardiovascular risk assessment")
            recommendations.append("Schedule follow-up appointment within 2-4 weeks")
            
        # Feature-specific recommendations
        for feature, shap_value in top_factors[:3]:  # Top 3 factors
            if shap_value > 0:  # Risk-increasing factors
                if 'bmi' in feature.lower() and features.get(feature, 0) > 25:
                    recommendations.append("Discuss weight management and lifestyle modifications")
                elif 'blood_pressure' in feature.lower():
                    recommendations.append("Monitor blood pressure regularly and consider antihypertensive therapy")
                elif 'glucose' in feature.lower():
                    recommendations.append("Evaluate for diabetes risk and consider glucose monitoring")
                elif 'cholesterol' in feature.lower():
                    recommendations.append("Consider lipid management and dietary counseling")
                elif 'smoking' in feature.lower():
                    recommendations.append("Provide smoking cessation counseling and support")
        
        if not recommendations:
            recommendations.append("Continue current preventive care and regular check-ups")
        
        return recommendations


class LIMEExplainer(XAIExplainer):
    """
    LIME-based explainer implementation.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        self.explainer = None
        
    def initialize_explainer(self, X_train: pd.DataFrame) -> None:
        """
        Initialize LIME explainer.
        
        Args:
            X_train: Training data for explainer initialization
        """
        lime_config = self.xai_config.get('lime', {})
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=['Low Risk', 'High Risk'],
            mode='classification',
            kernel_width=lime_config.get('kernel_width', 0.75),
            verbose=False
        )
        
        self.logger.info("LIME explainer initialized successfully")
    
    def generate_explanation(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate LIME explanations.
        
        Args:
            X: Input features to explain
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing LIME explanation results
        """
        if self.explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_explainer first.")
        
        lime_config = self.xai_config.get('lime', {})
        n_features = lime_config.get('n_features', 10)
        n_samples = lime_config.get('n_samples', 5000)
        
        self.logger.info(f"Generating LIME explanations for {len(X)} samples")
        
        explanations = []
        
        for i in range(len(X)):
            try:
                # Generate explanation for individual instance
                exp = self.explainer.explain_instance(
                    X.iloc[i].values,
                    self.model.predict_proba,
                    num_features=n_features,
                    num_samples=n_samples
                )
                
                # Extract explanation data
                explanation_data = {
                    'sample_id': i,
                    'features': X.iloc[i].to_dict(),
                    'prediction': self.model.predict_proba(X.iloc[i:i+1])[0],
                    'lime_explanation': exp.as_list(),
                    'lime_score': exp.score,
                    'intercept': exp.intercept[1] if len(exp.intercept) > 1 else exp.intercept[0]
                }
                
                explanations.append(explanation_data)
                
            except Exception as e:
                self.logger.error(f"Error generating LIME explanation for sample {i}: {e}")
                continue
        
        results = {
            'explanations': explanations,
            'feature_names': list(X.columns)
        }
        
        self.logger.info(f"LIME explanations generated for {len(explanations)} samples")
        return results
    
    def visualize_explanation(self, explanation: Dict[str, Any], 
                            output_dir: str = "results/explainability",
                            **kwargs) -> None:
        """
        Create LIME visualizations.
        
        Args:
            explanation: LIME explanation results
            output_dir: Directory to save plots
            **kwargs: Additional visualization arguments
        """
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        # Create visualizations for first few samples
        for i, exp in enumerate(explanation['explanations'][:5]):
            # Create horizontal bar plot for feature importance
            lime_exp = exp['lime_explanation']
            features = [item[0] for item in lime_exp]
            values = [item[1] for item in lime_exp]
            
            plt.figure(figsize=(10, 6))
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            plt.barh(features, values, color=colors, alpha=0.7)
            plt.xlabel('LIME Explanation Score')
            plt.title(f'LIME Explanation - Sample {i+1}')
            plt.grid(axis='x', alpha=0.3)
            
            # Add prediction probability as text
            pred_prob = exp['prediction'][1] if len(exp['prediction']) > 1 else exp['prediction'][0]
            plt.text(0.02, 0.98, f'Risk Probability: {pred_prob:.3f}', 
                    transform=plt.gca().transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path / f'lime_explanation_sample_{i+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"LIME visualizations saved to {output_path}")


def generate_shap_explanation(model: Any, X: pd.DataFrame, feature_names: List[str]) -> str:
    """
    Convenience function to generate SHAP explanation text.
    
    Args:
        model: Trained model
        X: Input features
        feature_names: List of feature names
        
    Returns:
        Formatted explanation string
    """
    try:
        # Initialize SHAP explainer (simplified for demo)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle binary classification case
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Get feature importance
        feature_importance = dict(zip(feature_names, np.abs(shap_values[0])))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Generate explanation text
        explanation = "**SHAP Feature Impact Analysis:**\\n\\n"
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            impact = "increases" if shap_values[0][feature_names.index(feature)] > 0 else "decreases"
            explanation += f"{i+1}. **{feature}**: {impact} risk (importance: {importance:.3f})\\n"
        
        return explanation
        
    except Exception as e:
        return f"Error generating SHAP explanation: {str(e)}"
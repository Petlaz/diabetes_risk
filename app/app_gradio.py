"""
Health XAI Gradio Application

Interactive web interface for health risk predictions with explainable AI features.
Provides both prediction capabilities and model explanations through SHAP visualizations.
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from explainability import generate_shap_explanation
    from utils import load_model, preprocess_input
except ImportError:
    # Fallback for development
    def generate_shap_explanation(*args, **kwargs):
        return "SHAP explanation would appear here"
    
    def load_model(path):
        return None
    
    def preprocess_input(data):
        return data

class HealthXAIApp:
    """Main application class for the Health XAI Gradio interface."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained models and preprocessing artifacts."""
        try:
            # Load model artifacts from results directory
            results_path = Path(__file__).parent.parent / "results" / "models"
            
            # This will be populated when models are trained
            # self.model = load_model(results_path / "best_model.joblib")
            # self.scaler = load_model(results_path / "standard_scaler.joblib")
            
            # Load feature names if available
            data_path = Path(__file__).parent.parent / "data" / "processed"
            if (data_path / "feature_names.csv").exists():
                feature_df = pd.read_csv(data_path / "feature_names.csv")
                self.feature_names = feature_df['feature'].tolist()
            else:
                # Default feature names for demo
                self.feature_names = [
                    "age", "gender", "bmi", "blood_pressure", "glucose_level",
                    "cholesterol", "smoking", "family_history", "exercise_frequency"
                ]
                
        except Exception as e:
            print(f"Warning: Could not load model artifacts: {e}")
            print("Running in demo mode with placeholder functionality.")
    
    def predict_risk(self, *inputs):
        """
        Make health risk prediction based on input features.
        
        Args:
            *inputs: Variable number of input features
            
        Returns:
            tuple: (risk_level, confidence, explanation)
        """
        try:
            # Convert inputs to DataFrame
            input_data = pd.DataFrame([inputs], columns=self.feature_names[:len(inputs)])
            
            if self.model is not None:
                # Real prediction
                if self.scaler:
                    input_scaled = self.scaler.transform(input_data)
                else:
                    input_scaled = input_data
                
                prediction_proba = self.model.predict_proba(input_scaled)[0]
                prediction = self.model.predict(input_scaled)[0]
                
                confidence = max(prediction_proba)
                risk_level = "High Risk" if prediction == 1 else "Low Risk"
                
                # Generate explanation
                explanation = generate_shap_explanation(self.model, input_scaled, self.feature_names)
                
            else:
                # Demo prediction for development
                # Simple heuristic based on inputs for demonstration
                risk_score = sum(inputs) / len(inputs) if inputs else 0.5
                confidence = min(0.95, max(0.6, risk_score))
                risk_level = "High Risk" if risk_score > 0.6 else "Low Risk"
                explanation = self._generate_demo_explanation(inputs)
            
            return risk_level, f"{confidence:.2%}", explanation
            
        except Exception as e:
            return "Error", "N/A", f"Prediction error: {str(e)}"
    
    def _generate_demo_explanation(self, inputs):
        """Generate a demo explanation for development purposes."""
        if not inputs:
            return "No input data provided for explanation."
        
        explanation = "**Risk Factor Analysis (Demo Mode)**\\n\\n"
        
        for i, (feature, value) in enumerate(zip(self.feature_names[:len(inputs)], inputs)):
            impact = "High" if value > 0.7 else "Medium" if value > 0.4 else "Low"
            explanation += f"• **{feature.replace('_', ' ').title()}**: {value:.2f} ({impact} impact)\\n"
        
        explanation += "\\n*This is a demonstration. Actual SHAP explanations will be provided when models are trained.*"
        return explanation
    
    def create_interface(self):
        """Create and configure the Gradio interface."""
        
        # Define input components based on feature names
        inputs = []
        for feature in self.feature_names[:9]:  # Limit to first 9 for demo
            if feature in ['gender', 'smoking', 'family_history']:
                inputs.append(gr.Dropdown(
                    choices=[0, 1], 
                    label=feature.replace('_', ' ').title(),
                    value=0
                ))
            else:
                inputs.append(gr.Slider(
                    minimum=0, 
                    maximum=1, 
                    value=0.5,
                    label=feature.replace('_', ' ').title()
                ))
        
        # Define outputs
        outputs = [
            gr.Textbox(label="Risk Assessment", interactive=False),
            gr.Textbox(label="Confidence", interactive=False),
            gr.Markdown(label="Explanation")
        ]
        
        # Create interface
        interface = gr.Interface(
            fn=self.predict_risk,
            inputs=inputs,
            outputs=outputs,
            title="Health XAI - Risk Assessment System",
            description="""
            This system provides health risk predictions with explainable AI.
            
            **Instructions:**
            1. Adjust the input values using the sliders and dropdowns
            2. Click 'Submit' to get a risk assessment
            3. Review the explanation to understand the prediction
            
            **Note:** This is a demonstration system. In production, it would use trained models on real health data.
            """,
            examples=[
                [0.3, 0, 0.4, 0.5, 0.3, 0.4, 0, 0, 0.6],  # Low risk example
                [0.7, 1, 0.8, 0.9, 0.8, 0.7, 1, 1, 0.2],  # High risk example
            ],
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                font-family: 'Arial', sans-serif;
            }
            .title {
                text-align: center;
                color: #2c3e50;
            }
            """
        )
        
        return interface

def main():
    """Main function to launch the Gradio app."""
    app = HealthXAIApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        share=False,  # Set to True for public sharing
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()
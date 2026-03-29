import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.predict import DropoutPredictor

# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def set_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .risk-low {
            color: #28a745;
            font-weight: bold;
        }
        .risk-medium {
            color: #ffc107;
            font-weight: bold;
        }
        .risk-high {
            color: #dc3545;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .stSlider > div > div > div > div {
            background-color: #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        self.predictor = DropoutPredictor()
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model and preprocessing objects."""
        try:
            self.model_loaded = self.predictor.load_model()
            return self.model_loaded
        except FileNotFoundError:
            st.error("Error: Model files not found. Please run training first.")
            return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def render_header(self):
        """Render the dashboard header."""
        st.markdown('<h1 class="main-header">🎓 Student Dropout Prediction System</h1>', 
                    unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            **Explainable AI-based Early Warning System for Student Dropout Prediction**
            
            This system uses machine learning to predict student dropout risk and provides 
            explanations for predictions using SHAP (SHapley Additive exPlanations).
            """)
    
    def render_input_section(self):
        """Render the input section with sliders."""
        st.header("📊 Student Information")
        st.markdown("Adjust the sliders to input student data:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            attendance = st.slider(
                "Attendance Rate (%)",
                min_value=0,
                max_value=100,
                value=75,
                step=1,
                key="attendance_slider",
                help="Student's attendance percentage"
            )
            
            assignment_delay = st.slider(
                "Average Assignment Delay (days)",
                min_value=0,
                max_value=30,
                value=2,
                step=1,
                key="assignment_delay_slider",
                help="Average delay in assignment submission"
            )
            
            participation = st.slider(
                "Class Participation Score",
                min_value=0,
                max_value=10,
                value=6,
                step=1,
                key="participation_slider",
                help="Student's participation in class activities (0-10)"
            )
        
        with col2:
            study_hours = st.slider(
                "Weekly Study Hours",
                min_value=0,
                max_value=40,
                value=15,
                step=1,
                key="study_hours_slider",
                help="Number of hours spent studying per week"
            )
            
            stress_level = st.slider(
                "Stress Level",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key="stress_level_slider",
                help="Self-reported stress level (1=Low, 10=High)"
            )
        
        # Create input dictionary
        input_data = {
            'attendance': attendance,
            'assignment_delay': assignment_delay,
            'participation': participation,
            'study_hours': study_hours,
            'stress_level': stress_level
        }
        
        return input_data
    
    def render_prediction_results(self, input_data):
        """Render prediction results."""
        st.header("🔮 Prediction Results")
        
        # Get prediction
        result = self.predictor.get_prediction_details(input_data)
        
        if result is None:
            st.error("Unable to make prediction. Please check if the model is loaded correctly.")
            return
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Prediction",
                result['prediction'] if result else 'Error',
                delta=None
            )
        
        with col2:
            st.metric(
                "Risk Score",
                f"{result['risk_score']:.1f}/100" if result else 'Error',
                delta=None
            )
        
        with col3:
            risk_class = f"risk-{result['risk_level'].lower().split()[0]}" if result else 'risk-error'
            st.markdown(f'<div class="metric-card {risk_class}">Risk Level: {result["risk_level"] if result else "Error"}</div>', 
                       unsafe_allow_html=True)
        
        with col4:
            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}" if result else 'Error',
                delta=None
            )
        
        # Probability visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Probabilities")
            
            # Create gauge chart for risk score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result['risk_score'] if result else 0,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Dropout Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Class Probabilities")
            
            # Create bar chart for probabilities
            prob_data = pd.DataFrame({
                'Class': ['No Dropout', 'Dropout'],
                'Probability': [result['no_dropout_probability'] if result else 0, result['dropout_probability'] if result else 0]
            })
            
            fig = px.bar(
                prob_data,
                x='Class',
                y='Probability',
                color='Class',
                color_discrete_map={'No Dropout': 'green', 'Dropout': 'red'},
                text_auto='.2%'
            )
            
            fig.update_layout(
                yaxis_title="Probability",
                xaxis_title="",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_shap_explanation(self, input_data):
        """Render SHAP explanation."""
        st.header("🧠 Explainable AI - Feature Importance")
        
        try:
            # Load SHAP explainer
            if not hasattr(self, 'shap_explainer') or self.shap_explainer is None:
                self.create_shap_explainer()
            
            # Check if explainer was created successfully
            if self.shap_explainer is None:
                st.error("SHAP explainer could not be created. Please check the logs.")
                return
            
            # Preprocess input
            processed_input = self.predictor.preprocess_input(input_data)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(processed_input)
            
            # Handle binary classification - extract the positive class (dropout = 1)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            elif len(shap_values.shape) == 3:  # (samples, features, classes)
                shap_values = shap_values[0, :, 1]  # First sample, all features, positive class
            elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:  # (features, classes)
                shap_values = shap_values[:, 1]  # All features, positive class
            
            # Ensure shap_values is 1D
            if len(shap_values.shape) > 1:
                shap_values = shap_values.flatten()
            
            # Create feature contribution dataframe
            contributions = pd.DataFrame({
                'Feature': self.predictor.feature_columns,
                'SHAP Value': shap_values,
                'Feature Value': processed_input.iloc[0].values,
                'Impact': ['Increases Risk' if float(val) > 0 else 'Decreases Risk' for val in shap_values]
            })
            
            contributions['Absolute SHAP'] = contributions['SHAP Value'].abs()
            contributions = contributions.sort_values('Absolute SHAP', ascending=False)
            
            # Display top contributors
            st.subheader("Top Feature Contributors")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create waterfall-like bar chart
                top_features = contributions.head(10)
                
                fig = px.bar(
                    top_features,
                    x='SHAP Value',
                    y='Feature',
                    orientation='h',
                    color='Impact',
                    color_discrete_map={'Increases Risk': 'red', 'Decreases Risk': 'green'},
                    title="Feature Contributions to Dropout Risk",
                    text_auto='.3f'
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    xaxis_title="SHAP Value (Impact on Prediction)",
                    yaxis_title=""
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display feature details table
                st.subheader("Feature Details")
                
                display_df = contributions[['Feature', 'Feature Value', 'SHAP Value', 'Impact']].head(10)
                display_df = display_df.rename(columns={
                    'Feature Value': 'Value',
                    'SHAP Value': 'Contribution'
                })
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Feature importance summary
            st.subheader("Global Feature Importance")
            
            # Get the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            if os.path.exists(os.path.join(project_root, 'models/shap_feature_importance.csv')):
                global_importance = pd.read_csv(os.path.join(project_root, 'models/shap_feature_importance.csv'))
                
                fig = px.bar(
                    global_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Overall Feature Importance (Based on SHAP)",
                    text_auto='.3f'
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    xaxis_title="Mean |SHAP Value|",
                    yaxis_title=""
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            st.info("SHAP explanation requires the model to be trained with SHAP analysis.")
    
    def create_shap_explainer(self):
        """Create SHAP explainer."""
        try:
            # Get the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            # Load a sample of training data for background
            df = pd.read_csv(os.path.join(project_root, 'data/student_data.csv'))
            X_background = df.drop('dropout_risk', axis=1).head(100)
            X_background_processed = self.predictor.preprocess_input(X_background)
            
            # Create explainer using Random Forest from ensemble
            rf_model = None
            for estimator_tuple in self.predictor.model.estimators_:
                if isinstance(estimator_tuple, tuple):
                    name, estimator = estimator_tuple
                    if hasattr(estimator, 'feature_importances_'):  # Tree-based model
                        rf_model = estimator
                        break
                else:
                    # Direct estimator object
                    estimator = estimator_tuple
                    if hasattr(estimator, 'feature_importances_'):  # Tree-based model
                        rf_model = estimator
                        break
            
            if rf_model is None:
                # Fallback to first estimator
                first_estimator = self.predictor.model.estimators_[0]
                if isinstance(first_estimator, tuple):
                    rf_model = first_estimator[1]
                else:
                    rf_model = first_estimator
            
            self.shap_explainer = shap.TreeExplainer(rf_model)
            st.success("SHAP explainer created successfully!")
            
        except Exception as e:
            st.error(f"Error creating SHAP explainer: {str(e)}")
            self.shap_explainer = None
    
    def render_model_info(self):
        """Render model information."""
        st.header("📈 Model Information")
        
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            
            if os.path.exists(os.path.join(project_root, 'models/model_comparison.csv')):
                model_comparison = pd.read_csv(os.path.join(project_root, 'models/model_comparison.csv'))
                model_comparison = model_comparison.set_index('Unnamed: 0')
                
                # Display performance metrics
                st.dataframe(
                    model_comparison.round(4),
                    use_container_width=True
                )
                
                # Highlight best model
                best_model = model_comparison['f1_score'].idxmax()
                st.success(f"Best performing model: **{best_model}** (F1 Score: {model_comparison.loc[best_model, 'f1_score']:.4f})")
            else:
                st.info("Model performance data not available. Please run training first.")
        
        with col2:
            st.subheader("Dataset Statistics")
            
            if os.path.exists(os.path.join(project_root, 'data/student_data.csv')):
                df = pd.read_csv(os.path.join(project_root, 'data/student_data.csv'))
                
                # Basic statistics
                st.write(f"**Total Students:** {len(df)}")
                st.write(f"**Dropout Rate:** {df['dropout_risk'].mean():.2%}")
                
                # Feature statistics
                feature_stats = df.describe().round(2)
                st.dataframe(
                    feature_stats,
                    use_container_width=True
                )
            else:
                st.info("Dataset not available. Please generate data first.")
    
    def run(self):
        """Run the dashboard application."""
        set_custom_css()
        
        # Load model
        if not self.load_model():
            st.error("Failed to load model. Please ensure the model is trained first.")
            st.stop()
        
        # Render header
        self.render_header()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🧠 Explainability", "📊 Model Info"])
        
        # Get input data once at the top
        input_data = self.render_input_section()
        
        with tab1:
            # Predict button
            if st.button("🔮 Predict Dropout Risk", type="primary", use_container_width=True, key="predict_button"):
                # Render prediction results
                self.render_prediction_results(input_data)
        
        with tab2:
            # SHAP explanation
            if st.button("🧠 Generate SHAP Explanation", use_container_width=True, key="shap_button"):
                self.render_shap_explanation(input_data)
        
        with tab3:
            # Model information
            self.render_model_info()

def main():
    """Main function to run the Streamlit app."""
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    main()

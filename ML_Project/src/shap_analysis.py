import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

class SHAPAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.explainer = None
        
    def load_model_and_preprocessors(self):
        """Load the trained model and preprocessing objects."""
        print("Loading trained model and preprocessing objects...")
        
        # Load the best performing model (Voting Classifier)
        self.model = joblib.load('models/voting_classifier.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.imputer = joblib.load('models/imputer.pkl')
        self.feature_columns = joblib.load('models/feature_columns.pkl')
        
        print("Model and preprocessing objects loaded successfully!")
    
    def load_data(self):
        """Load the dataset for analysis."""
        print("Loading dataset...")
        df = pd.read_csv('data/student_data.csv')
        return df
    
    def preprocess_sample(self, sample):
        """Preprocess a single sample for SHAP analysis."""
        # Convert to DataFrame if needed
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        elif isinstance(sample, list):
            sample = pd.DataFrame([sample], columns=self.feature_columns)
        
        # Handle missing values
        sample_imputed = self.imputer.transform(sample)
        sample_imputed = pd.DataFrame(sample_imputed, columns=self.feature_columns)
        
        # Scale features
        sample_scaled = self.scaler.transform(sample_imputed)
        sample_scaled = pd.DataFrame(sample_scaled, columns=self.feature_columns)
        
        return sample_scaled
    
    def create_explainer(self, X_background):
        """Create SHAP explainer."""
        print("Creating SHAP explainer...")
        
        # For Voting Classifier, we'll use the first estimator (Random Forest) for SHAP
        # as SHAP doesn't directly support VotingClassifier
        if hasattr(self.model, 'estimators_'):
            # Use Random Forest as the representative model
            rf_model = None
            for estimator_tuple in self.model.estimators_:
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
                first_estimator = self.model.estimators_[0]
                if isinstance(first_estimator, tuple):
                    rf_model = first_estimator[1]
                else:
                    rf_model = first_estimator
            
            self.explainer = shap.TreeExplainer(rf_model)
            print(f"Using {type(rf_model).__name__} for SHAP analysis")
        else:
            # Check if it's a tree-based model
            if hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for non-tree models
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
        
        return self.explainer
    
    def generate_summary_plot(self, X_test):
        """Generate and save SHAP summary plot."""
        print("\n=== GENERATING SHAP SUMMARY PLOT ===")
        
        # Create explainer
        explainer = self.create_explainer(X_test)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test)
        
        # Handle binary classification (shap_values might be a list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (dropout = 1)
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('plots/shap_summary_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig('plots/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("SHAP summary plots saved to plots/")
        
        # Calculate feature importance
        if len(shap_values.shape) > 1:
            # For multi-class SHAP values, take the mean across samples and classes
            if len(shap_values.shape) == 3:  # (samples, features, classes)
                importance_values = np.abs(shap_values).mean(axis=0).mean(axis=1)
            else:  # (samples, features)
                importance_values = np.abs(shap_values).mean(axis=0)
        else:
            importance_values = np.abs(shap_values)
        
        # Ensure importance_values is 1D
        if len(importance_values.shape) > 1:
            importance_values = importance_values.flatten()
        
        # Debug print to check shapes
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"Importance values shape: {importance_values.shape}")
        print(f"Feature columns length: {len(self.feature_columns)}")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns[:len(importance_values)],
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (SHAP values):")
        print(feature_importance)
        
        return shap_values, feature_importance
    
    def explain_single_prediction(self, sample_data):
        """Explain a single prediction using SHAP."""
        print("\n=== SINGLE PREDICTION EXPLANATION ===")
        
        # Preprocess the sample
        sample_processed = self.preprocess_sample(sample_data)
        
        # Make prediction
        prediction = self.model.predict(sample_processed)[0]
        prediction_proba = self.model.predict_proba(sample_processed)[0]
        
        print(f"Prediction: {'Dropout' if prediction == 1 else 'No Dropout'}")
        print(f"Prediction Probability: {prediction_proba}")
        
        # Calculate SHAP values for this sample
        if self.explainer is None:
            self.create_explainer(sample_processed)
        
        shap_values = self.explainer.shap_values(sample_processed)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Create force plot
        plt.figure(figsize=(12, 6))
        
        # Handle different SHAP versions - skip force plot for now due to version issues
        try:
            # Try new SHAP API
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[1]
            else:
                base_value = self.explainer.expected_value
            
            # Use shap.plots.force instead of shap.force_plot for newer versions
            import shap as shap_module
            if hasattr(shap_module, 'plots'):
                shap_module.plots.force(base_value, shap_values[0])
                plt.title('SHAP Force Plot - Single Prediction')
                plt.tight_layout()
                plt.savefig('plots/shap_force_plot.png', dpi=300, bbox_inches='tight')
            else:
                # Skip force plot if not supported
                print("Force plot skipped due to SHAP version compatibility")
                
        except Exception as e:
            print(f"Force plot skipped due to error: {str(e)}")
        
        plt.close()
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        
        # Handle different SHAP versions and shapes
        try:
            # Check if shap_values is 2D (features, classes)
            if len(shap_values.shape) == 2:
                # Use the positive class (dropout = 1)
                waterfall_values = shap_values[:, 1] if shap_values.shape[1] > 1 else shap_values[:, 0]
            else:
                waterfall_values = shap_values[0]
            
            # Get base value
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[1]
            else:
                base_value = self.explainer.expected_value
            
            shap.waterfall_plot(
                shap.Explanation(values=waterfall_values, 
                                base_values=base_value,
                                data=sample_processed.iloc[0],
                                feature_names=self.feature_columns),
                max_display=10,
                show=False
            )
        except Exception as e:
            print(f"Waterfall plot skipped due to error: {str(e)}")
        
        plt.title('SHAP Waterfall Plot - Feature Contributions')
        plt.tight_layout()
        plt.savefig('plots/shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print feature contributions
        print("\nFeature Contributions:")
        
        # Handle shap_values shape for contributions
        if len(shap_values.shape) == 2:
            # Use the positive class for binary classification
            contrib_values = shap_values[:, 1] if shap_values.shape[1] > 1 else shap_values[:, 0]
        else:
            contrib_values = shap_values[0]
        
        # Ensure contrib_values is 1D
        if len(contrib_values.shape) > 1:
            contrib_values = contrib_values.flatten()
        
        # Get feature values from processed sample
        feature_values = sample_processed.iloc[0].values
        if len(feature_values.shape) > 1:
            feature_values = feature_values.flatten()
        
        # Ensure all arrays have same length
        min_length = min(len(self.feature_columns), len(contrib_values), len(feature_values))
        
        contributions = pd.DataFrame({
            'feature': self.feature_columns[:min_length],
            'shap_value': contrib_values[:min_length],
            'feature_value': feature_values[:min_length]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        print(contributions)
        
        return shap_values[0], contributions
    
    def create_risk_scoring_system(self, prediction_proba):
        """Convert prediction probability to risk score (0-100)."""
        risk_score = prediction_proba[1] * 100  # Probability of dropout * 100
        
        if risk_score <= 30:
            risk_level = "Low Risk"
        elif risk_score <= 70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        return risk_score, risk_level
    
    def complete_analysis(self):
        """Run complete SHAP analysis."""
        print("=== COMPLETE SHAP ANALYSIS ===")
        
        # Load model and data
        self.load_model_and_preprocessors()
        df = self.load_data()
        
        # Preprocess data
        X = df.drop('dropout_risk', axis=1)
        y = df['dropout_risk']
        
        X_processed = self.preprocess_sample(X)
        
        # Generate summary plot
        shap_values, feature_importance = self.generate_summary_plot(X_processed)
        
        # Explain a few sample predictions
        print("\n=== SAMPLE PREDICTION EXPLANATIONS ===")
        
        # High risk student
        high_risk_sample = {
            'attendance': 40,
            'assignment_delay': 10,
            'participation': 2,
            'study_hours': 5,
            'stress_level': 9
        }
        
        print("\n--- High Risk Student Example ---")
        shap_vals, contributions = self.explain_single_prediction(high_risk_sample)
        
        sample_processed = self.preprocess_sample(high_risk_sample)
        pred_proba = self.model.predict_proba(sample_processed)[0]
        risk_score, risk_level = self.create_risk_scoring_system(pred_proba)
        
        print(f"Risk Score: {risk_score:.2f}/100")
        print(f"Risk Level: {risk_level}")
        
        # Low risk student
        low_risk_sample = {
            'attendance': 95,
            'assignment_delay': 0,
            'participation': 9,
            'study_hours': 25,
            'stress_level': 2
        }
        
        print("\n--- Low Risk Student Example ---")
        shap_vals, contributions = self.explain_single_prediction(low_risk_sample)
        
        sample_processed = self.preprocess_sample(low_risk_sample)
        pred_proba = self.model.predict_proba(sample_processed)[0]
        risk_score, risk_level = self.create_risk_scoring_system(pred_proba)
        
        print(f"Risk Score: {risk_score:.2f}/100")
        print(f"Risk Level: {risk_level}")
        
        # Save feature importance
        feature_importance.to_csv('models/shap_feature_importance.csv', index=False)
        
        print("\nSHAP Analysis Complete!")
        return feature_importance

def main():
    # Initialize analyzer
    analyzer = SHAPAnalyzer()
    
    # Run complete analysis
    feature_importance = analyzer.complete_analysis()
    
    print("\n=== ANALYSIS SUMMARY ===")
    print("Generated files:")
    print("- ../plots/shap_summary.png")
    print("- ../plots/shap_summary_bar.png")
    print("- ../plots/shap_force_plot.png")
    print("- ../plots/shap_waterfall.png")
    print("- ../models/shap_feature_importance.csv")

if __name__ == "__main__":
    main()

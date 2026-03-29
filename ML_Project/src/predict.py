import pandas as pd
import numpy as np
import joblib
import os

class DropoutPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
    def load_model(self):
        """Load the trained model and preprocessing objects."""
        try:
            # Get the project root directory
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            models_dir = os.path.join(project_root, 'models')
            
            self.model = joblib.load(os.path.join(models_dir, 'voting_classifier.pkl'))
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            self.imputer = joblib.load(os.path.join(models_dir, 'imputer.pkl'))
            self.feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
            
            print("Model and preprocessing objects loaded successfully!")
            return True
        except FileNotFoundError:
            print("Error: Model files not found. Please run training first.")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction."""
        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame([input_data], columns=self.feature_columns)
        else:
            input_df = input_data.copy()
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value
        
        # Reorder columns to match training data
        input_df = input_df[self.feature_columns]
        
        # Handle missing values
        input_imputed = self.imputer.transform(input_df)
        input_imputed = pd.DataFrame(input_imputed, columns=self.feature_columns)
        
        # Scale features
        input_scaled = self.scaler.transform(input_imputed)
        input_scaled = pd.DataFrame(input_scaled, columns=self.feature_columns)
        
        return input_scaled
    
    def predict(self, input_data):
        """Make prediction on input data."""
        if not self.load_model():
            return None, None
        
        # Preprocess input
        processed_input = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_input)[0]
        prediction_proba = self.model.predict_proba(processed_input)[0]
        
        return prediction, prediction_proba
    
    def calculate_risk_score(self, prediction_proba):
        """Calculate risk score and level."""
        dropout_probability = prediction_proba[1]
        risk_score = dropout_probability * 100
        
        if risk_score <= 30:
            risk_level = "Low Risk"
            risk_color = "green"
        elif risk_score <= 70:
            risk_level = "Medium Risk"
            risk_color = "orange"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        return risk_score, risk_level, risk_color
    
    def get_prediction_details(self, input_data):
        """Get detailed prediction with risk assessment."""
        prediction, prediction_proba = self.predict(input_data)
        
        if prediction is None:
            return None
        
        risk_score, risk_level, risk_color = self.calculate_risk_score(prediction_proba)
        
        result = {
            'prediction': 'Dropout' if prediction == 1 else 'No Dropout',
            'dropout_probability': prediction_proba[1],
            'no_dropout_probability': prediction_proba[0],
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': max(prediction_proba)
        }
        
        return result

def test_predictions():
    """Test the prediction system with sample data."""
    print("=== TESTING PREDICTION SYSTEM ===")
    
    predictor = DropoutPredictor()
    
    # Test cases
    test_cases = [
        {
            'name': 'High Risk Student',
            'data': {
                'attendance': 35,
                'assignment_delay': 12,
                'participation': 1,
                'study_hours': 3,
                'stress_level': 9
            }
        },
        {
            'name': 'Medium Risk Student',
            'data': {
                'attendance': 65,
                'assignment_delay': 5,
                'participation': 5,
                'study_hours': 12,
                'stress_level': 6
            }
        },
        {
            'name': 'Low Risk Student',
            'data': {
                'attendance': 92,
                'assignment_delay': 1,
                'participation': 8,
                'study_hours': 22,
                'stress_level': 3
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        result = predictor.get_prediction_details(test_case['data'])
        
        if result:
            print(f"Input Data: {test_case['data']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Dropout Probability: {result['dropout_probability']:.4f}")
            print(f"No Dropout Probability: {result['no_dropout_probability']:.4f}")
            print(f"Risk Score: {result['risk_score']:.2f}/100")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print("Prediction failed!")

def main():
    """Main function for testing predictions."""
    test_predictions()

if __name__ == "__main__":
    main()

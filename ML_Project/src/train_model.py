import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import os

class StudentDropoutPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        
    def preprocess_data(self, df):
        """Preprocess the data for training."""
        print("=== DATA PREPROCESSING ===")
        
        # Separate features and target
        X = df.drop('dropout_risk', axis=1)
        y = df['dropout_risk']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Handle missing values
        print("Handling missing values...")
        X_imputed = self.imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=self.feature_columns)
        
        # Feature scaling
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_imputed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        print(f"Features: {self.feature_columns}")
        print(f"Shape after preprocessing: {X_scaled.shape}")
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        print(f"\n=== DATA SPLITTING ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Training dropout rate: {y_train.mean():.3f}")
        print(f"Test dropout rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_individual_models(self, X_train, y_train):
        """Train individual models."""
        print("\n=== TRAINING INDIVIDUAL MODELS ===")
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Support Vector Machine
        print("Training Support Vector Machine...")
        svm = SVC(random_state=42, probability=True)
        svm.fit(X_train, y_train)
        self.models['svm'] = svm
        
        print("Individual models trained successfully!")
    
    def train_ensemble_model(self, X_train, y_train):
        """Train ensemble Voting Classifier."""
        print("\n=== TRAINING ENSEMBLE MODEL ===")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('rf', self.models['random_forest']),
                ('svm', self.models['svm'])
            ],
            voting='soft'  # Use probability voting
        )
        
        print("Training Voting Classifier...")
        voting_clf.fit(X_train, y_train)
        self.models['voting_classifier'] = voting_clf
        
        print("Ensemble model trained successfully!")
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\n--- {model_name.upper()} PERFORMANCE ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        return metrics, y_pred_proba
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models and compare performance."""
        print("\n=== MODEL EVALUATION ===")
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            if name != 'voting_classifier':
                metrics, _ = self.evaluate_model(model, X_test, y_test, name)
                results[name] = metrics
        
        # Evaluate ensemble model
        ensemble_metrics, _ = self.evaluate_model(
            self.models['voting_classifier'], X_test, y_test, 'voting_classifier'
        )
        results['voting_classifier'] = ensemble_metrics
        
        # Create comparison table
        print("\n=== MODEL COMPARISON ===")
        comparison_df = pd.DataFrame(results).T
        print(comparison_df.round(4))
        
        # Find best model
        best_f1_model = comparison_df['f1_score'].idxmax()
        print(f"\nBest model based on F1 Score: {best_f1_model}")
        
        return results, comparison_df
    
    def save_models(self):
        """Save all trained models."""
        print("\n=== SAVING MODELS ===")
        
        os.makedirs('../models', exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = f'../models/{name}.pkl'
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, '../models/scaler.pkl')
        joblib.dump(self.imputer, '../models/imputer.pkl')
        joblib.dump(self.feature_columns, '../models/feature_columns.pkl')
        
        print("All models and preprocessing objects saved successfully!")
    
    def train_pipeline(self, df):
        """Complete training pipeline."""
        print("=== STUDENT DROPOUT PREDICTION TRAINING PIPELINE ===")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train models
        self.train_individual_models(X_train, y_train)
        self.train_ensemble_model(X_train, y_train)
        
        # Evaluate models
        results, comparison = self.evaluate_all_models(X_test, y_test)
        
        # Save models
        self.save_models()
        
        # Save comparison results
        comparison.to_csv('../models/model_comparison.csv')
        
        print("\n=== TRAINING COMPLETE ===")
        return results, comparison

def main():
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('../data/student_data.csv')
    
    # Initialize predictor
    predictor = StudentDropoutPredictor()
    
    # Run training pipeline
    results, comparison = predictor.train_pipeline(df)
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()

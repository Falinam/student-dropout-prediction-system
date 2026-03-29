import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def generate_student_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic student dataset for dropout prediction.
    
    Features:
    - attendance: Student attendance percentage (0-100)
    - assignment_delay: Average delay in assignment submission (days)
    - participation: Class participation score (0-10)
    - study_hours: Weekly study hours (0-40)
    - stress_level: Self-reported stress level (1-10)
    - dropout_risk: Binary target (0=No Dropout, 1=Dropout)
    """
    
    np.random.seed(random_state)
    
    # Generate correlated features
    print("Generating synthetic student dataset...")
    
    # Create realistic student data with correlations
    data = {
        'attendance': np.random.normal(75, 15, n_samples).clip(0, 100),
        'assignment_delay': np.random.exponential(2, n_samples).clip(0, 30),
        'participation': np.random.normal(6, 2, n_samples).clip(0, 10),
        'study_hours': np.random.normal(15, 8, n_samples).clip(0, 40),
        'stress_level': np.random.normal(5, 2, n_samples).clip(1, 10)
    }
    
    df = pd.DataFrame(data)
    
    # Create dropout risk based on feature combinations
    # Higher risk for: low attendance, high assignment delay, low participation, low study hours, high stress
    risk_score = (
        (100 - df['attendance']) * 0.3 +
        df['assignment_delay'] * 2 +
        (10 - df['participation']) * 4 +
        (40 - df['study_hours']) * 0.5 +
        df['stress_level'] * 3
    )
    
    # Add some noise and create binary target
    risk_score += np.random.normal(0, 10, n_samples)
    dropout_threshold = np.percentile(risk_score, 70)  # Top 30% at risk
    df['dropout_risk'] = (risk_score > dropout_threshold).astype(int)
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'study_hours'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
    df.loc[missing_indices, 'stress_level'] = np.nan
    
    print(f"Dataset generated with {n_samples} samples")
    print(f"Dropout distribution: {df['dropout_risk'].value_counts().to_dict()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def save_dataset(df, filepath):
    """Save the dataset to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")

def main():
    # Generate dataset
    df = generate_student_dataset(n_samples=1000, random_state=42)
    
    # Save dataset
    save_dataset(df, '../data/student_data.csv')
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    print("\nFeature correlations with dropout risk:")
    correlations = df.corr()['dropout_risk'].sort_values(ascending=False)
    print(correlations)

if __name__ == "__main__":
    main()

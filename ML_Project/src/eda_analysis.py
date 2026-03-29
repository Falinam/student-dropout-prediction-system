import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath):
    """Load the student dataset."""
    return pd.read_csv(filepath)

def perform_eda(df):
    """Perform exploratory data analysis."""
    print("=== EXPLORATORY DATA ANALYSIS ===\n")
    
    # Basic info
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
    
    # Target distribution
    print("\nTarget Variable Distribution:")
    target_counts = df['dropout_risk'].value_counts()
    print(target_counts)
    print(f"Dropout Rate: {target_counts[1]/len(df)*100:.2f}%")
    
    # Feature statistics by dropout status
    print("\nFeature Statistics by Dropout Status:")
    print(df.groupby('dropout_risk').agg({
        'attendance': ['mean', 'std'],
        'assignment_delay': ['mean', 'std'],
        'participation': ['mean', 'std'],
        'study_hours': ['mean', 'std'],
        'stress_level': ['mean', 'std']
    }))
    
    return df

def create_visualizations(df):
    """Create and save visualization plots."""
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create plots directory
    os.makedirs('../plots', exist_ok=True)
    
    # 1. Target distribution
    plt.figure(figsize=(8, 6))
    df['dropout_risk'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Dropout Risk Distribution')
    plt.xlabel('Dropout Risk (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../plots/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature distributions by dropout status
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    features = ['attendance', 'assignment_delay', 'participation', 'study_hours', 'stress_level']
    colors = ['skyblue', 'salmon']
    
    for i, feature in enumerate(features):
        if i < len(axes):
            for dropout_status in [0, 1]:
                subset = df[df['dropout_risk'] == dropout_status]
                axes[i].hist(subset[feature], alpha=0.7, label=f'Dropout={dropout_status}', 
                           color=colors[dropout_status], bins=20)
            axes[i].set_title(f'{feature.replace("_", " ").title()}')
            axes[i].set_xlabel(feature.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('../plots/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('../plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            df.boxplot(column=feature, by='dropout_risk', ax=axes[i])
            axes[i].set_title(f'{feature.replace("_", " ").title()} by Dropout Status')
            axes[i].set_xlabel('Dropout Risk')
            axes[i].set_ylabel(feature.replace("_", " ").title())
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('../plots/box_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to ../plots/")

def main():
    # Load data
    df = load_data('../data/student_data.csv')
    
    # Perform EDA
    df = perform_eda(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\nEDA Analysis Complete!")

if __name__ == "__main__":
    main()

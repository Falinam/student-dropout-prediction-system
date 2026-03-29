#!/usr/bin/env python3
"""
Main execution script for Student Dropout Prediction System

This script runs the complete pipeline:
1. Generate synthetic dataset
2. Perform exploratory data analysis
3. Train multiple ML models
4. Create ensemble model
5. Generate SHAP explanations
6. Launch Streamlit dashboard

Author: ML Project Team
Date: March 2026
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print project banner."""
    print("=" * 80)
    print("🎓 EXPLAINABLE ML-BASED EARLY WARNING SYSTEM FOR STUDENT DROPOUT PREDICTION")
    print("=" * 80)
    print("Final Year Machine Learning Mini-Project")
    print("Features: Multiple Models, Ensemble Learning, SHAP Explainability, Dashboard")
    print("=" * 80)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('shap', 'shap'),
        ('plotly', 'plotly'),
        ('streamlit', 'streamlit'),
        ('joblib', 'joblib')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n🔄 {description}")
    print("-" * 50)
    
    try:
        # Change to src directory for script execution
        if script_path.startswith('src/'):
            original_cwd = os.getcwd()
            os.chdir('src')
            script_path = script_path.replace('src/', '')
        
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        # Change back to original directory
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"✅ {description} - COMPLETED")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ['data', 'models', 'src', 'dashboard', 'plots']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/")
        else:
            print(f"✅ {directory}/ already exists")

def verify_files():
    """Verify that necessary files exist."""
    print("\n🔍 Verifying generated files...")
    
    required_files = [
        'data/student_data.csv',
        'models/voting_classifier.pkl',
        'models/scaler.pkl',
        'models/imputer.pkl',
        'models/feature_columns.pkl',
        'models/model_comparison.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files are present!")
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n🚀 Launching Streamlit Dashboard...")
    print("-" * 50)
    
    try:
        # Launch Streamlit in a new process
        dashboard_process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 
            'dashboard/app.py', '--server.headless', 'true'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        print("✅ Dashboard is starting...")
        print("📊 Dashboard will be available at: http://localhost:8501")
        print("🔄 The dashboard is running in the background...")
        print("💡 Press Ctrl+C to stop the dashboard")
        
        # Keep the script running
        try:
            dashboard_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            dashboard_process.terminate()
            dashboard_process.wait()
            print("✅ Dashboard stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error launching dashboard: {str(e)}")
        return False
    
    return True

def main():
    """Main execution function."""
    print_banner()
    
    # Step 0: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Generate dataset
    if not run_script('src/generate_dataset.py', 'Generating Student Dataset'):
        print("\n❌ Dataset generation failed. Please check the error above.")
        return
    
    # Step 3: Perform EDA
    if not run_script('src/eda_analysis.py', 'Performing Exploratory Data Analysis'):
        print("\n❌ EDA failed. Please check the error above.")
        return
    
    # Step 4: Train models
    if not run_script('src/train_model.py', 'Training Machine Learning Models'):
        print("\n❌ Model training failed. Please check the error above.")
        return
    
    # Step 5: SHAP analysis
    if not run_script('src/shap_analysis.py', 'Performing SHAP Explainability Analysis'):
        print("\n❌ SHAP analysis failed. Please check the error above.")
        return
    
    # Step 6: Verify files
    if not verify_files():
        print("\n❌ Some required files are missing. Please check the errors above.")
        return
    
    # Step 7: Display success message
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📊 Generated Files:")
    print("  📁 data/student_data.csv - Synthetic dataset")
    print("  📁 models/ - Trained models and preprocessing objects")
    print("  📁 plots/ - Visualization plots and SHAP explanations")
    print("\n📈 Model Performance:")
    print("  🤖 Multiple ML models trained (Logistic Regression, Random Forest, SVM)")
    print("  🎯 Ensemble Voting Classifier created")
    print("  📊 Performance metrics calculated and saved")
    print("  🧠 SHAP explanations generated")
    
    # Step 8: Ask user about dashboard
    print("\n" + "=" * 80)
    choice = input("🚀 Would you like to launch the interactive dashboard? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes', '']:
        launch_dashboard()
    else:
        print("\n💡 To launch the dashboard later, run:")
        print("   streamlit run dashboard/app.py")
        print("\n📚 Project is ready for use!")
    
    print("\n" + "=" * 80)
    print("🎓 Student Dropout Prediction System - Ready!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

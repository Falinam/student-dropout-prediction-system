# Explainable Machine Learning Based Early Warning System for Student Dropout Prediction

## 🎓 Project Overview

This project implements an explainable machine learning system for predicting student dropout risk. The system uses ensemble learning techniques combined with SHAP (SHapley Additive exPlanations) to provide both accurate predictions and interpretable explanations for those predictions.

### Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, Support Vector Machine
- **Ensemble Learning**: Voting Classifier for improved performance
- **Explainable AI**: SHAP-based feature importance and prediction explanations
- **Risk Scoring System**: Converts predictions to 0-100 risk scores with categorical levels
- **Interactive Dashboard**: Streamlit-based web interface for real-time predictions
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

## 📁 Project Structure

```
ML_Project/
│
├── data/
│   └── student_data.csv          # Generated synthetic dataset
│
├── models/
│   ├── voting_classifier.pkl     # Trained ensemble model
│   ├── logistic_regression.pkl   # Logistic Regression model
│   ├── random_forest.pkl         # Random Forest model
│   ├── svm.pkl                   # SVM model
│   ├── scaler.pkl                # Feature scaler
│   ├── imputer.pkl               # Missing value imputer
│   ├── feature_columns.pkl       # Feature column names
│   ├── model_comparison.csv      # Model performance comparison
│   └── shap_feature_importance.csv # SHAP feature importance
│
├── src/
│   ├── generate_dataset.py       # Dataset generation
│   ├── train_model.py           # Model training pipeline
│   ├── predict.py               # Prediction functionality
│   ├── shap_analysis.py         # SHAP explainability analysis
│   └── eda_analysis.py          # Exploratory Data Analysis
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard
│
├── plots/                       # Generated visualizations
│   ├── target_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── box_plots.png
│   ├── shap_summary.png
│   ├── shap_summary_bar.png
│   ├── shap_force_plot.png
│   └── shap_waterfall.png
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── main.py                      # Main execution script
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd ML_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**
   ```bash
   python main.py
   ```

## 🏃‍♂️ How to Run

### Option 1: Complete Pipeline (Recommended)

Run the entire project from start to finish:

```bash
python main.py
```

This will:
1. Generate synthetic student dataset
2. Perform exploratory data analysis
3. Train multiple ML models
4. Create ensemble model
5. Generate SHAP explanations
6. Launch the Streamlit dashboard

### Option 2: Step-by-Step Execution

1. **Generate Dataset**
   ```bash
   python src/generate_dataset.py
   ```

2. **Exploratory Data Analysis**
   ```bash
   python src/eda_analysis.py
   ```

3. **Train Models**
   ```bash
   python src/train_model.py
   ```

4. **SHAP Analysis**
   ```bash
   python src/shap_analysis.py
   ```

5. **Launch Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

## 📊 Features & Functionality

### Dataset Features

- **attendance**: Student attendance percentage (0-100)
- **assignment_delay**: Average delay in assignment submission (days)
- **participation**: Class participation score (0-10)
- **study_hours**: Weekly study hours (0-40)
- **stress_level**: Self-reported stress level (1-10)
- **dropout_risk**: Binary target (0=No Dropout, 1=Dropout)

### Risk Scoring System

- **Low Risk (0-30)**: Green indicator
- **Medium Risk (30-70)**: Orange indicator
- **High Risk (70-100)**: Red indicator

### Dashboard Features

1. **Prediction Tab**:
   - Interactive sliders for student data input
   - Real-time dropout risk prediction
   - Risk score visualization with gauge chart
   - Probability distribution display

2. **Explainability Tab**:
   - SHAP feature importance visualization
   - Individual feature contributions
   - Waterfall plots showing prediction drivers
   - Global feature importance rankings

3. **Model Info Tab**:
   - Model performance comparison
   - Dataset statistics
   - Feature distributions

## 📈 Model Performance

The system uses an ensemble Voting Classifier that combines:
- Logistic Regression
- Random Forest (100 estimators)
- Support Vector Machine (with probability estimates)

Performance metrics are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 🧠 Explainable AI (XAI)

The project implements SHAP (SHapley Additive exPlanations) for model interpretability:

- **Global Explanations**: Overall feature importance across all predictions
- **Local Explanations**: Individual prediction explanations
- **Visualizations**: Summary plots, force plots, waterfall plots
- **Feature Contributions**: Shows how each feature impacts predictions

## 🎯 Sample Output

### Prediction Example:
```
Input Data:
- Attendance: 65%
- Assignment Delay: 5 days
- Participation: 5/10
- Study Hours: 12/week
- Stress Level: 6/10

Results:
- Prediction: Dropout Risk
- Risk Score: 67.3/100
- Risk Level: Medium Risk
- Confidence: 82.4%
- Dropout Probability: 67.3%
```

### SHAP Explanation:
- **attendance**: Low attendance increases risk (+0.45)
- **stress_level**: High stress increases risk (+0.32)
- **study_hours**: Moderate study hours decreases risk (-0.18)
- **participation**: Low participation increases risk (+0.25)
- **assignment_delay**: Moderate delay increases risk (+0.15)

## 🔧 Dependencies

### Core Libraries:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms

### Visualization:
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualization
- `plotly`: Interactive plots for dashboard

### Explainability:
- `shap`: SHAP explanations for model interpretability

### Dashboard:
- `streamlit`: Web application framework

### Utilities:
- `joblib`: Model serialization

## 📝 Academic Use

This project is designed as a final-year machine learning mini-project with:

- **Moderate Complexity**: Not too basic, not overly complex
- **Educational Value**: Covers multiple ML concepts
- **Practical Application**: Real-world problem (student retention)
- **Modern Techniques**: Ensemble learning, XAI, interactive dashboard
- **Clean Code**: Modular, well-documented, reproducible

## 🤝 Contributing

This is an academic project. For improvements or suggestions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please feel free to use and modify for academic projects.

## 🐛 Troubleshooting

### Common Issues:

1. **Model Loading Error**:
   - Ensure you've run the training pipeline first
   - Check if model files exist in the `models/` directory

2. **SHAP Analysis Error**:
   - Requires trained models
   - May need additional memory for large datasets

3. **Dashboard Not Loading**:
   - Check Streamlit installation: `pip install streamlit`
   - Ensure all dependencies are installed

4. **Memory Issues**:
   - Reduce dataset size in `generate_dataset.py`
   - Close other applications to free memory

### Performance Tips:

- Use smaller datasets for testing (n_samples=500)
- Disable SHAP visualizations for faster execution
- Use CPU instead of GPU for this project

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure proper file structure
4. Run the complete pipeline first

---

**Project Status**: ✅ Complete and Tested
**Last Updated**: March 2026
**Python Version**: 3.8+

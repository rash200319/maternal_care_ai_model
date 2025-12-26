# Maternal Care AI — Preeclampsia Risk Prediction 🏥

A machine learning system for predicting preeclampsia risk using maternal clinical data, built with scikit-learn's Random Forest classifier. This tool is designed for educational and research purposes only — **not for clinical use**.

## 🌟 Features

- **Comprehensive Risk Assessment**: Predicts preeclampsia risk based on 21+ clinical parameters
- **Data Preprocessing**: Handles missing values, feature scaling, and data normalization
- **Model Explainability**: Feature importance analysis to understand risk factors
- **Easy Integration**: Simple API for making predictions on new patient data
- **Visual Analytics**: Built-in visualization of model performance and feature importance
---

## 📊 Model Performance

The current model achieves the following performance metrics:
- Accuracy: [To be filled after model training]
- ROC-AUC: [To be filled after model training]
- Precision/Recall: [To be filled after model training]

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/maternal_care_ai_model.git
   cd maternal_care_ai_model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

Run the training script:
```bash
python work.py
```

This will:
1. Load and preprocess the data
2. Train the Random Forest classifier
3. Save the model and scaler as `.pkl` files
4. Generate performance metrics and visualizations

## � Data Dictionary

| Feature       | Description                          | Type    | Range/Values        |
|---------------|--------------------------------------|---------|---------------------|
| age           | Maternal age                         | Numeric | 18-45 years         |
| gest_age      | Gestational age                      | Numeric | Weeks               |
| height        | Height in cm                         | Numeric | 140-200 cm          |
| weight        | Weight in kg                         | Numeric | 40-120 kg           |
| bmi           | Body Mass Index                      | Numeric | 18-40 kg/m²         |
| sysbp         | Systolic Blood Pressure              | Numeric | 90-200 mmHg         |
| diabp         | Diastolic Blood Pressure             | Numeric | 60-120 mmHg         |
| hb            | Hemoglobin level                     | Numeric | g/dL                |
| ...           | ... (complete with all features)     | ...     | ...                 |

## 🤖 Making Predictions

### Programmatic Usage

```python
from joblib import load
import pandas as pd

# Load the trained model and scaler
model = load('models/preeclampsia_model.pkl')
scaler = load('models/scaler.pkl')
feature_columns = load('models/feature_columns.pkl')

def predict_preeclampsia_risk(patient_data):
    """Predict preeclampsia risk for a new patient."""
    # Convert to DataFrame and ensure correct column order
    patient_df = pd.DataFrame([patient_data])[feature_columns]
    
    # Scale the features
    scaled_features = scaler.transform(patient_df)
    
    # Get prediction probability
    risk_probability = model.predict_proba(scaled_features)[0][1]
    
    return {
        'risk_probability': float(risk_probability),
        'risk_category': 'High' if risk_probability > 0.5 else 'Low'
    }

# Example usage
patient = {
    'age': 28,
    'gest_age': 24,
    'height': 165,
    'weight': 70,
    'bmi': 25.7,
    'sysbp': 120,
    'diabp': 80,
    'hb': 12.5,
    'pcv': 36,
    'tsh': 2.1,
    'platelet': 250,
    'creatinine': 0.8,
    'plgfsflt': 50,
    'seng': 10,
    'cysc': 0.9,
    'pp_13': 5,
    'glycerides': 120,
    'htn': 0,
    'diabetes': 0,
    'fam_htn': 0,
    'sp_art': 0
}
result = predict_preeclampsia_risk(patient)
print(f"Preeclampsia Risk: {result['risk_probability']*100:.1f}% ({result['risk_category']})")
```

## 📊 Model Interpretation

The model provides feature importance scores to help understand which factors contribute most to the risk prediction:

```python
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Preeclampsia Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

## 🔧 Notes / Troubleshooting

- The scripts expect numeric columns; some binary/categorical columns (e.g., `htn`, `diabetes`, `fam_htn`, `sp_art`) are coerced to numeric before imputing missing values.
- If you see errors related to `median()` or `fillna()`, ensure columns are numeric (the repository contains a fix to coerce binaries and use `median(numeric_only=True)`).
- This model is for experimentation. Do not use in production without clinical validation and regulatory approvals.

## 🧪 Notebooks

Open `work.ipynb` and `work2.ipynb` to interactively explore preprocessing, feature importance plots, and other experiments.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or feedback, please open an issue or contact the repository maintainer.

## 🙏 Acknowledgments

- Data provided by [Source Name]
- Built with scikit-learn, pandas, and other open-source libraries
- Special thanks to contributors and researchers in maternal health

---

<div align="center">
  Made with love for better maternal healthcare
</div>

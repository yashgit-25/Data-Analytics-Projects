# Breast Cancer Classification with Logistic Regression

A machine learning pipeline for breast cancer diagnosis using Logistic Regression with GridSearchCV hyperparameter optimization.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [Performance](#performance)
- [Visualizations](#visualizations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project classifies breast tumors as:
- **Benign (B)** - Non-cancerous (Label: 0)
- **Malignant (M)** - Cancerous (Label: 1)

Uses 30+ cell nuclei features from 569 samples for binary classification.

## Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Setup
1. Clone or download the repository
2. Ensure `daa.txt` dataset is in the project directory
3. Run the script

## Usage

### Quick Start
```bash
python breast_cancer_classification.py
```

### File Structure
```
project/
├── breast_cancer_classification.py
├── daa.txt
└── README.md
```

## Dataset

**Format:** CSV file with columns `id,diagnosis,radius_mean,texture_mean,...`

**Features include:**
- Cell nuclei measurements (radius, texture, perimeter, area)
- Statistical measures (mean, standard error, worst values)
- 30+ numerical features total

## Model Pipeline

### 1. Data Preprocessing
- Handle missing values with median imputation
- Remove ID columns automatically
- Encode labels: M→1 (Malignant), B→0 (Benign)
- Validate and clean infinite values

### 2. Feature Engineering
- **StandardScaler:** Normalize features to zero mean, unit variance
- **SelectKBest:** Choose top 20 most significant features
- **Train-test split:** 80-20 with stratification

### 3. Model Training
**GridSearchCV Parameters:**
```python
{
    'C': [0.001, 0.01, 0.1, 1, 10, 100],    # Regularization strength
    'penalty': ['l1', 'l2'],                # Regularization type
    'solver': ['liblinear'],                # Optimization algorithm
    'max_iter': [1000, 2000]                # Maximum iterations
}
```
- **Cross-validation:** 5-fold CV
- **Scoring:** Accuracy optimization

### 4. Evaluation Metrics
- Accuracy score
- AUC-ROC curve
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Feature importance analysis

## Performance

### Expected Results
- **Accuracy:** >95%
- **AUC-ROC:** >0.98
- **Sensitivity:** >95%
- **Specificity:** >95%

### Output Example
```
Best cross-validation score: 0.9649
Test Accuracy: 0.9649
Test AUC-ROC: 0.9978
```

## Visualizations

The script generates four key plots:

1. **Confusion Matrix Heatmap** - Classification accuracy breakdown
2. **ROC Curve** - True positive vs false positive rates
3. **Feature Importance Plot** - Top 10 most predictive features
4. **Cross-Validation Scores** - Model stability across folds

### Data Format Requirements
- CSV format with header row
- First column: Sample ID (auto-detected and removed)
- Second column: Diagnosis (M/B values)
- Remaining columns: Numerical features only

### Debug Tips
- Check data types: All features must be numeric
- Verify target distribution: Should have both M and B labels  
- Ensure no duplicate samples in dataset

## License

This project is for educational and research purposes only.

---

**⚠️ Medical Disclaimer:** This tool is for research and educational purposes only. Always consult healthcare professionals for medical diagnosis and treatment decisions.

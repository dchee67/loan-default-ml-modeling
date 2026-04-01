# Loan Default Prediction — Machine Learning Model Comparison

A end-to-end machine learning project predicting consumer loan default and loss severity using a progression of models from logistic regression through deep learning. Built across four assignments as part of a graduate machine learning course, this project covers the full ML lifecycle: data preparation, feature engineering, model development, evaluation, and communication of results.

---

## Overview

This project tackles two prediction problems on a consumer loan dataset:

1. **Binary Classification** — Will this loan default? (`TARGET_BAD_FLAG`)
2. **Regression** — If the loan defaults, how much will be lost? (`TARGET_LOSS_AMT`)

Five model families are built and benchmarked against each other, culminating in a TensorFlow neural network comparison across activation functions and architectures.

---

## Dataset

- **Source:** HMEQ (Home Equity) loan dataset — a widely used benchmark dataset for credit risk modeling
- **Records:** 5,960 consumer loans
- **Features:** 13 variables including loan amount, debt-to-income ratio, delinquency history, job type, and reason for loan
- **Targets:** Binary default flag + continuous loss amount (for defaulted loans only)

---

## Project Structure

```
loan-default-ml-modeling/
├── Assignment_1_ML.ipynb       # Data preparation, EDA, feature engineering
├── Chee_Assignment_2.ipynb     # Decision trees (classification + regression)
├── Chee_Assignment_3.ipynb     # Random forest & gradient boosting
├── Chee_Assignment_4.ipynb     # TensorFlow deep learning + full model comparison
└── README.md
```

---

## Progression by Assignment

### Assignment 1 — Data Preparation & EDA
- Exploratory data analysis: distribution plots, missing value assessment, class balance
- Missing value imputation strategy for continuous and categorical variables
- Feature encoding for categorical variables (job type, loan reason)
- Train/test split (80/20) with consistent random state

### Assignment 2 — Decision Trees
- Decision tree classifier for default prediction (max_depth=4)
- Decision tree regressor for loss-given-default
- Variable importance extraction from tree structure
- Accuracy and RMSE benchmarking on train and test sets

### Assignment 3 — Ensemble Methods
- **Random Forest** classifier and regressor
- **Gradient Boosting** classifier and regressor
- Feature importance visualization
- ROC curve comparison across logistic regression, random forest, and gradient boosting
- Model selection rationale based on AUC and generalization

### Assignment 4 — TensorFlow Deep Learning
- Sequential Feature Selection (SFS) using logistic regression as base estimator, optimized on ROC-AUC
- Five TensorFlow neural network architectures compared:

| Model | Architecture | Test AUC | Test Accuracy |
|-------|-------------|----------|---------------|
| TF_1_hidden_ReLU | 1 hidden layer, ReLU | 0.908 | 88.65% |
| TF_1_hidden_Tanh | 1 hidden layer, Tanh | 0.909 | 88.65% |
| TF_2_hidden_ReLU_Dropout | 2 hidden layers, ReLU + Dropout(0.3) | **0.917** | **89.37%** |

- Full model comparison: logistic regression, gradient boosting, and all TF architectures on a single ROC curve
- **Best model:** TF_2_hidden_ReLU_Dropout — Test AUC 0.917, Test Accuracy 89.4%
- **Recommended model for LGD regression:** Gradient Boosting (Test RMSE ≈ 2,961 vs TF best of ≈ 13,062)

---

## Key Results

### Classification (Loan Default)
| Model | Test AUC | Test Accuracy |
|-------|----------|---------------|
| Logistic Regression | 0.899 | — |
| Gradient Boosting | 0.914 | — |
| TF 1-hidden ReLU | 0.908 | 88.65% |
| TF 1-hidden Tanh | 0.909 | 88.65% |
| **TF 2-hidden ReLU + Dropout** | **0.917** | **89.37%** |

### Regression (Loss Given Default)
| Model | Test RMSE |
|-------|-----------|
| Linear Regression | 3,835 |
| **Gradient Boosting** | **2,961** |
| TF ReLU Dropout | 13,062 |

---

## How to Run

1. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow mlxtend
```

2. Run notebooks in order:
```
Assignment_1_ML.ipynb → Assignment_2 → Assignment_3 → Assignment_4
```

> Note: The HMEQ dataset is a publicly available benchmark. A copy is available at [various academic sources](https://www.kaggle.com/datasets/ajay1735/hmeq-data) or can be requested from the course materials.

---

## Key Technical Features

- **Feature engineering:** imputation, one-hot encoding, standardization pipeline
- **Sequential Feature Selection (SFS):** forward selection with 5-fold CV optimized on ROC-AUC
- **Model evaluation:** accuracy, ROC/AUC curves, RMSE, train vs test comparison to assess overfitting
- **Deep learning:** TensorFlow/Keras Sequential API with ReLU, Tanh, and Dropout regularization
- **Model communication:** written comparison reports with clear rationale for technical and non-technical audiences

---

## Built With

- Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn, mlxtend
- Northwestern University — MSDS 422: Machine Learning

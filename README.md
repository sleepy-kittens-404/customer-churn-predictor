# Customer Churn Predictor

A machine learning web application that predicts whether a telecom customer is likely to churn, explains *why* using SHAP explainability, and visualizes churn risk through an interactive gauge — all wrapped in a deployable Streamlit interface.

**Live Demo:** [https://customer-churn-predictor-ml.streamlit.app/]

---

## What This Project Does

Customer churn — when a customer stops using a service — is one of the most costly problems in the telecom industry. Acquiring a new customer costs 5–7x more than retaining an existing one. This project builds an end-to-end ML pipeline that helps businesses identify at-risk customers before they leave, and understand exactly which factors are driving that risk.

Given a customer's contract type, tenure, payment method, and service details, the model outputs:
- A churn prediction (will churn / will not churn)
- The probability of churn
- A SHAP-based explanation of the top factors influencing that specific prediction
- A visual churn risk gauge

---

## Demo

![App Screenshot](c:\Users\Muhammad Sami\Pictures\Screenshots\Screenshot 2026-06-06 231224.png)

> Enter customer details in the sidebar → Click Predict → Get prediction, explanation, and risk gauge

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | pandas, numpy, scikit-learn |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Modeling | XGBoost, scikit-learn |
| Explainability | SHAP |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
customer-churn-predictor/
│
├── eda_and_preprocessing.ipynb   # Exploratory data analysis, feature engineering, preprocessing
├── model_training.ipynb          # Model training, hyperparameter tuning, evaluation
├── model_explainability.ipynb    # SHAP global and local explainability
├── app.py                        # Streamlit web application
│
├── best_model.pkl                # Saved XGBoost model
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ML Pipeline

### 1. Exploratory Data Analysis
- Investigated class imbalance (churn vs no-churn distribution)
- Identified and fixed data type issues (`TotalCharges` stored as string)
- Analyzed churn rates across contract types, tenure groups, and payment methods
- Generated correlation heatmaps and distribution plots

### 2. Preprocessing
- Applied `StandardScaler` to numerical features (tenure, MonthlyCharges, TotalCharges)
- Applied `OneHotEncoder` to categorical features
- Used `ColumnTransformer` to handle both in a single pipeline
- Performed proper train/test split before fitting — no data leakage

### 3. Handling Class Imbalance
Used **SMOTE (Synthetic Minority Oversampling Technique)** on the training set only to balance the churn classes. The test set was kept at its original distribution to reflect real-world evaluation conditions.

### 4. Model Training & Comparison
Trained and evaluated 9 models:
- Logistic Regression, Naive Bayes, KNN (baselines)
- SVM, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost

Models were compared using **F1-score and ROC-AUC** rather than accuracy, given the imbalanced nature of the original test distribution.

### 5. Hyperparameter Tuning
Used `GridSearchCV` with 5-fold cross-validation, scoring on F1, to find optimal parameters for SVM, Decision Tree, Random Forest, AdaBoost, Gradient Boost, and XGBoost.

### 6. Final Model: XGBoost
XGBoost with tuned hyperparameters achieved the best balance of precision and recall on the minority class (churners):

| Metric | Score |
|---|---|
| Accuracy | 76% |
| Churn Recall | 75% |
| Churn F1 | 0.65 |
| ROC-AUC | — |

> Prediction threshold lowered from 0.5 to 0.4 to prioritize recall — in churn prediction, missing an actual churner (false negative) is more costly than a false alarm.

### 7. SHAP Explainability
Applied SHAP `TreeExplainer` to understand model decisions:

**Key findings:**
- `Contract_Month-to-month` is the single strongest churn driver
- Low `tenure` strongly increases churn probability
- Absence of `OnlineSecurity` and `TechSupport` increases risk
- `Fiber optic` internet users churn more than DSL users
- `Electronic check` payment method correlates with higher churn

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/sleepy-kittens-404/customer-churn-predictor
cd customer-churn-predictor

# Install dependencies
pip install -r requirements.txt
# 1. Run the notebooks in order
jupyter notebook eda_and_preprocessing.ipynb   # generates processed CSVs
jupyter notebook model_training.ipynb           # generates best_model.pkl
jupyter notebook model_explainability.ipynb     # optional, for SHAP analysis

# 2. Then run the app
streamlit run app.py
```

---

## Dataset

**Telco Customer Churn** from Kaggle — 7,043 customers, 19 features including demographics, account information, and subscribed services.

[Dataset Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Key Learnings

- Preprocessing order matters: scalers and encoders must be fit on training data only
- Accuracy is misleading on imbalanced datasets — F1 and recall tell the real story
- SMOTE must be applied after the train/test split, never before
- Lowering the prediction threshold is a valid business decision, not a hack
- Model explainability (SHAP) bridges the gap between ML and real business decisions
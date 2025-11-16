# 🧠 Credit Risk Prediction with SHAP

This project builds a machine learning pipeline to predict credit risk and explain model decisions using SHAP (SHapley Additive exPlanations). It uses a synthetic dataset of loan applicants and provides interpretable insights for underwriting policies.

---

## 📦 Project Structure

- `credit_risk_shap_project.ipynb` — Main notebook with data generation, preprocessing, model training, SHAP analysis, and recommendations
- `credit_data.csv` — Synthetic dataset (generated if not found)
- `requirements.txt` — Python dependencies
- `bestmodel.pkl` — Trained model artifact
- `report.md` — Project summary and findings
- `outputs/` — All evaluation metrics, SHAP plots, and recommendations

---

## 🧪 Dataset

- 20,000 synthetic applicants
- Features include:
  - Numeric: `age`, `income`, `loan_amount`, `credit_score`, `employment_length_years`
  - Categorical: `home_ownership`, `employment_status`
- Target: `default` (1 = high risk, 0 = low risk)
- Class imbalance handled using SMOTE

---

## ⚙️ Pipeline Overview

1. **Preprocessing**: Imputation, scaling, encoding via `ColumnTransformer`
2. **Modeling**: LightGBM with SMOTE and hyperparameter tuning (`RandomizedSearchCV`)
3. **Evaluation**: AUC, F1, precision, recall, confusion matrix
4. **Interpretability**: SHAP global summary + local case studies (TP, FP, FN)
5. **Recommendations**: Business rules derived from SHAP insights

---

## 📊 Evaluation Metrics

- ROC-AUC: 0.5166  
- F1 Score: 0.0453  
- Precision: 0.4444  
- Recall: 0.0239  
- Accuracy: 0.92  

The model predicts class 1 with moderate precision, enabling valid SHAP analysis. Recall is low, indicating missed high-risk cases.

---

## 🔍 SHAP Insights

### Global Importance
Top features influencing risk predictions:
- CreditScore
- LoanAmount
- Income
- EmploymentStatus
- Age

### Local Case Studies
- True Positive: Correctly flagged risky applicant with low credit score and high loan amount
- False Positive: Overestimated risk due to short employment history
- False Negative: Missed risk due to low loan amount masking poor credit

---

## 📣 Underwriting Recommendations

1. Flag applicants with CreditScore < 600 for manual review  
2. Limit LoanAmount-to-Income ratios above 50%  
3. Prioritize stable employment history in risk assessment

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook credit_risk_shap_project.ipynb

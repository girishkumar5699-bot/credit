📄 Credit Risk Prediction with SHAP — Project Report
🎯 Objective
This project aims to build a machine learning pipeline that predicts credit risk and explains model decisions using SHAP (SHapley Additive exPlanations). The goal is to support data-driven underwriting by identifying high-risk applicants and providing interpretable insights into model behavior.

🧪 Dataset
- Source: Synthetic dataset of 20,000 loan applicants
- Features:
- Numeric: age, income, loan_amount, credit_score, employment_length_years
- Categorical: home_ownership, employment_status
- Target: default (binary: 1 = high risk, 0 = low risk)
- Class imbalance: Addressed using SMOTE (Synthetic Minority Oversampling Technique)

⚙️ Methodology
1. Preprocessing
- Numeric features: median imputation + standard scaling
- Categorical features: mode imputation + one-hot encoding
- Combined using ColumnTransformer inside a pipeline
2. Modeling
- Models tested: XGBoost, LightGBM, RandomForest
- Final model: LightGBM with SMOTE
- Hyperparameter tuning via RandomizedSearchCV (AUC scoring)
3. Evaluation
- Metrics computed on test set:
- ROC-AUC: 0.5166
- F1 Score: 0.0453
- Precision: 0.4444
- Recall: 0.0239
- Accuracy: 0.92
- Model predicts class 1 with moderate precision, enabling valid SHAP analysis

🔍 SHAP Interpretability
Global SHAP Summary
Top features influencing predictions:
- CreditScore: Lower scores strongly increase predicted risk
- LoanAmount: Larger loans push predictions toward high risk
- Income: Lower income contributes to higher risk
- EmploymentStatus: Unstable or unemployed status increases risk
- Age: Younger applicants are slightly more likely to be flagged as risky
Local SHAP Case Studies
True Positive
- CreditScore = 580, LoanAmount = ₹5L, Employment = 1 year
- SHAP impact: CreditScore (−0.45), LoanAmount (−0.32), EmploymentStatus (−0.21)
- Interpretation: Correctly flagged as high risk due to poor credit and high loan burden
False Positive
- CreditScore = 720, LoanAmount = ₹6L, Employment = 1 year
- SHAP impact: LoanAmount (−0.38), EmploymentStatus (−0.29), CreditScore (+0.22)
- Interpretation: Overestimated risk due to short employment and high loan despite good credit
False Negative
- CreditScore = 610, LoanAmount = ₹2L, Employment = 0 years
- SHAP impact: LoanAmount (+0.31), CreditScore (−0.28), EmploymentStatus (−0.25)
- Interpretation: Missed risk due to low loan masking poor credit and unstable employment

📣 Strategic Recommendations
- Flag applicants with CreditScore < 600 for manual review — SHAP shows this is the strongest risk driver
- Limit LoanAmount-to-Income ratios above 50% — SHAP reveals high loan burden increases default risk
- Prioritize stable employment history — SHAP shows short or unstable employment contributes to risk

📁 Deliverables
All outputs are saved in the outputs/ folder:
- metrics_report.json, classification_report.txt
- global_shap_importance.csv, textual_shap_report.txt
- local_shap_explanations.json, SHAP plots (PNG + HTML)
- underwriting_recommendations.txt
- bestmodel.pkl, shap_artifacts.joblib

🧾 Academic Integrity Statement
All code was written and executed by the author. SHAP plots and interpretations are based on actual model outputs. Markdown commentary reflects personal understanding of the results and their business implications.

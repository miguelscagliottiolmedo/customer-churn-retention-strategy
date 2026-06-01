# Customer Churn Prediction & Retention Strategy

Predicting bank customer churn and turning model output into a differentiated, value-based retention strategy.

## Context
Final project for the **Postgraduate Diploma in Data Science (FAMAF, Universidad Nacional de Córdoba)**. Completed as a team; this repository documents the full analysis, modeling, and strategy work.

## Problem
A bank wanted to identify customers at risk of churning (closing their accounts) and decide where to invest retention efforts. The challenge wasn't only *who* will churn, but *which* at-risk customers are worth retaining — balancing churn probability against customer value.

## Dataset
Public "Credit Card Customers" dataset: **10,127 records** and **38 variables** covering demographic, financial, and product-usage attributes. The target variable (`Attrition_Flag`) is imbalanced: ~16% churned, ~84% retained.

## Approach
- **Exploratory analysis**: distribution of the target, correlation analysis, and detection of multicollinearity. We tested models with and without multicollinearity treatment and found the full feature set gave the model better discriminative power for at-risk customers.
- **Modeling**: compared four classifiers, optimizing for **recall** (catching as many true churners as possible):

| Model | Recall | Notes |
|---|---|---|
| Logistic Regression | ~0.80 | Solid baseline, easy to interpret |
| AdaBoost | ~0.83 | Captures non-linear relationships |
| **XGBoost (selected)** | **~0.86** | Best recall, stable and robust to multicollinearity |
| MLPClassifier | ~0.82 | High train recall but overfitting risk |

- **Customer Value Index (CVI)**: built a normalized (0–1) index combining financial and behavioral signals — income & credit (`Income_Category`, `Credit_Limit`), transactional activity (`Total_Trans_Amt`, `Total_Trans_Ct`), relationship depth (`Total_Relationship_Count`, `Months_on_book`), and dynamism (`Avg_Utilization_Ratio`, transaction growth).
- **Decision matrix**: crossed CVI against predicted churn probability to create six strategic quadrants (Premium / Mid / Low value × high / low churn risk), each mapped to a concrete retention play — from personalized VIP retention to low-cost mass campaigns.

## Key findings
- XGBoost identified at-risk customers most reliably (recall ~0.86).
- Combining churn probability with customer value (rather than churn alone) lets the bank focus spend where it matters: prioritize **high-value customers at high risk**, and avoid over-investing in low-value churners.
- The output is an actionable segmentation, not just a prediction — each segment has a recommended action and suggested KPIs (churn reduction, transaction growth, engagement).

## Repository contents
- `notebooks/` — analysis and modeling notebooks (EDA, model comparison, CVI & segmentation)
- `report/` — full written report
- `presentation/` — final presentation deck

## Tech stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib / Seaborn

## Note
This is a team project from my data science diploma. Original deliverables are in Spanish; this summary is in English. The dataset is public.

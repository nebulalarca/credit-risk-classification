# Credit Risk Classification

Predicting the probability of a customer defaulting on a loan using machine learning.

## Dataset
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 customers, 10 features.

## Methods
- Exploratory Data Analysis (EDA)
- Missing value imputation
- Outlier removal
- SMOTE (imbalanced data)
- Logistic Regression, Random Forest, XGBoost
- Threshold Tuning (precision-recall trade-off)

## Results
| Model | Recall | Precision | F1 |
|---|---|---|---|
| Logistic Regression | 0.65 | 0.17 | 0.27 |
| Random Forest | 0.16 | 0.54 | 0.25 |
| XGBoost | 0.69 | 0.24 | 0.35 |

XGBoost with threshold=0.3 captures 82% of high-risk customers.

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib

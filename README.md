📊 Customer Credit Risk Analysis & Data Preprocessing Project
🚀 Overview

This project focuses on building a complete end-to-end data preprocessing pipeline for a Customer Credit Risk dataset.

The goal is to transform raw, unstructured data from multiple sources into a clean, feature-engineered dataset ready for machine learning models that predict loan default risk.

This project follows real-world data science workflows using industry-standard practices.

🎯 Objective
Perform data cleaning, preprocessing, and feature engineering
Combine data from multiple sources (CSV, JSON, SQL, API)
Handle missing values, outliers, and inconsistencies
Prepare a dataset suitable for machine learning modeling

As defined in the project brief, the final output is a fully processed dataset ready for ML prediction tasks

📂 Dataset Sources

This project integrates multiple data sources:

CSV Dataset
Customer financial data (income, loan, transactions)

JSON File

Customer demographic metadata
Example:
{
  "customer_id": "CUST1000",
  "age": 59,
  "gender": "Male",
  "region": "East"
}

SQL Database
Loan repayment history (missed payments, credit score)
External API (JSON)
Economic indicators (inflation, GDP, interest rate)
🧠 Problem Statement

Build a system that prepares data for predicting:

Loan Default

0 → No Default
1 → Default

This is a Binary Classification Problem

⚙️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
SciPy
SQLite
JSON Handling
🔄 Project Workflow
1. Data Loading
Load data from CSV, JSON, SQL, and API
Merge datasets using customer_id
2. Data Cleaning
Remove duplicates
Standardize column names
3. Missing Value Handling
Numerical: Median Imputation
Categorical: Most Frequent
Advanced Techniques:
KNN Imputer
MICE (Iterative Imputer)
4. Outlier Handling
Z-score filtering
Winsorization for extreme values
5. Feature Engineering
Date extraction (Year, Month)
Derived features like:
Debt-to-Income Ratio
6. Encoding
Ordinal Encoding → education level
Label Encoding → gender
One-Hot Encoding → region, loan purpose
7. Feature Scaling
StandardScaler applied to numerical data
8. Transformations
Log transformation
Power transformation (Yeo-Johnson)
9. Binning
K-Means clustering for transaction grouping
📈 Key Insights

From the analysis:

Credit score is the strongest predictor of risk
High debt-to-income ratio → higher default probability
Missed payments strongly indicate future default
Employment type impacts repayment capability
Economic indicators affect credit behavior
📊 Final Output
Cleaned and processed dataset
Ready for:
Machine Learning Models
Credit Risk Prediction

Final dataset shape (example):

Rows: 200
Columns: 25+

(as shown in project execution output)

🗂 Project Structure
📁 project/
│
├── data/
│   ├── customer_credit_risk_dataset.csv
│   ├── customer_metadata.json
│   ├── loan_repayment_history.sql
│   ├── external_economic_api.json
│
├── preprocessing_project.py
├── final_cleaned_credit_dataset.csv
├── README.md
▶️ How to Run
Clone the repository
git clone https://github.com/your-username/project-name.git
Install dependencies
pip install pandas numpy scikit-learn scipy
Run the project
python preprocessing_project.py
🧩 Key Concepts Used
Data Cleaning
Feature Engineering
Missing Value Imputation
Outlier Detection
Encoding Techniques
Feature Scaling
Data Integration

These align with standard data science workflow steps like CRISP-DM

📌 Future Improvements
Train ML models (Logistic Regression, Random Forest, XGBoost)
Build a prediction API
Deploy using Flask / FastAPI
Create a dashboard (Power BI / Streamlit)

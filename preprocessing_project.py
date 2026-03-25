# ==========================================================
# HOLISTIC DATA PREPARER – FULL END TO END PROJECT
# ==========================================================

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import warnings

warnings.filterwarnings("ignore")

# Sklearn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder,
    PowerTransformer, FunctionTransformer
)

from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats.mstats import winsorize


# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("Loading datasets...")

DATA_PATH = "data"

df_csv = pd.read_csv(os.path.join(DATA_PATH, "customer_credit_risk_dataset.csv"))
df_json = pd.read_json(os.path.join(DATA_PATH, "customer_metadata.json"))

# SQL Load
conn = sqlite3.connect(":memory:")
with open(os.path.join(DATA_PATH, "loan_repayment_history.sql"), "r") as f:
    conn.executescript(f.read())

df_sql = pd.read_sql("SELECT * FROM loan_repayment_history", conn)

# API Data
with open(os.path.join(DATA_PATH, "external_economic_api.json")) as f:
    api_data = json.load(f)

df_api = pd.DataFrame(api_data["indicators"])

print("Datasets loaded successfully!")


# ==========================================================
# 2. CLEAN COLUMN NAMES
# ==========================================================

df_csv.columns = df_csv.columns.str.strip()
df_json.columns = df_json.columns.str.strip()
df_sql.columns = df_sql.columns.str.strip()


# ==========================================================
# 3. MERGE DATASETS
# ==========================================================

print("Merging datasets...")

df = df_csv.copy()

if "customer_id" in df_json.columns:
    df = df.merge(df_json, on="customer_id", how="left")

if "customer_id" in df_sql.columns:
    df = df.merge(df_sql, on="customer_id", how="left")

print("Merged shape:", df.shape)


# ==========================================================
# 4. DATA UNDERSTANDING
# ==========================================================

print("\nDATA INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())


# ==========================================================
# 5. DATA CLEANING
# ==========================================================

df.drop_duplicates(inplace=True)
df.columns = df.columns.str.strip()


# ==========================================================
# 6. MISSING VALUE HANDLING
# ==========================================================

print("\nHandling missing values...")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Numerical
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# KNN
knn = KNNImputer(n_neighbors=5)
df[num_cols] = knn.fit_transform(df[num_cols])

# MICE
mice = IterativeImputer()
df[num_cols] = mice.fit_transform(df[num_cols])


# ==========================================================
# 7. OUTLIER HANDLING
# ==========================================================

print("\nHandling outliers...")

for col in num_cols:
    z = np.abs(stats.zscore(df[col]))
    df = df[z < 3]

# Winsorization
if "annual_income" in df.columns:
    df["annual_income"] = winsorize(df["annual_income"], limits=[0.05, 0.05])


# ==========================================================
# 8. FEATURE ENGINEERING
# ==========================================================

print("\nFeature engineering...")

# Date Handling
if "join_date" in df.columns:
    df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce")
    df["join_year"] = df["join_date"].dt.year
    df["join_month"] = df["join_date"].dt.month


# ==========================================================
# 9. ENCODING
# ==========================================================

print("\nEncoding...")

# Ordinal
if "education_level" in df.columns:
    ordinal = OrdinalEncoder()
    df[["education_level"]] = ordinal.fit_transform(df[["education_level"]])

# Label
if "gender" in df.columns:
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

# One-hot
cat_cols = df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=cat_cols)


# ==========================================================
# 10. NUMERICAL BINNING
# ==========================================================

if "transaction_count" in df.columns:
    kmeans = KMeans(n_clusters=3)
    df["txn_cluster"] = kmeans.fit_predict(df[["transaction_count"]])


# ==========================================================
# 11. FEATURE SCALING
# ==========================================================

print("\nScaling...")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# ==========================================================
# 12. TRANSFORMATIONS
# ==========================================================

print("\nTransformations...")

if "spending_ratio" in df.columns:
    log_transform = FunctionTransformer(np.log1p)
    df["log_spending_ratio"] = log_transform.fit_transform(df[["spending_ratio"]])

if "annual_income" in df.columns:
    pt = PowerTransformer(method="yeo-johnson")
    df[["annual_income"]] = pt.fit_transform(df[["annual_income"]])


# ==========================================================
# 13. FEATURE CONSTRUCTION
# ==========================================================

print("\nFeature construction...")

if "loan_amount" in df.columns and "annual_income" in df.columns:
    df["debt_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)

if "transaction_count" in df.columns:
    df["avg_monthly_txn"] = df["transaction_count"] / 6


# ==========================================================
# 14. FINAL SAVE
# ==========================================================

df.to_csv("final_cleaned_credit_dataset.csv", index=False)

print("\nProject Completed Successfully!")
print("Final Shape:", df.shape)
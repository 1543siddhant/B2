
# Data Cleaning and Preparation
# Problem Statement: Analyzing Customer Churn in a Telecommunications Company
# Dataset: "Telecom_Customer_Churn.csv"
# Description: The dataset contains information about customers of a telecommunications
# company and whether they have churned (i.e., discontinued their services). The dataset
# includes various attributes of the customers, such as their demographics, usage patterns, and
# account information. The goal is to perform data cleaning and preparation to gain insights
# into the factors that contribute to customer churn.
# Tasks to Perform:
# 1. Import the "Telecom_Customer_Churn.csv" dataset.
# 2. Explore the dataset to understand its structure and content.
# 3. Handle missing values in the dataset, deciding on an appropriate strategy.
# 4. Remove any duplicate records from the dataset.
# 5. Check for inconsistent data, such as inconsistent formatting or spelling variations,
# and standardize it.
# 6. Convert columns to the correct data types as needed.
# 7. Identify and handle outliers in the data.
# 8. Perform feature engineering, creating new features that may be relevant to
# predicting customer churn.
# 9. Normalize or scale the data if necessary
# 10. Split the dataset into training and testing sets for further analysis.
# 11. Export the cleaned dataset for future analysis or modeling.



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 1. Load ----------
dataset_path = r"D:\BE_Practicals\DMVcodes\DMV_Practical\3_tele_com.csv"
df = pd.read_csv(dataset_path, low_memory=False)

# ---------- 2. Quick explore ----------
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# ---------- 3. Basic cleanup of strings & small typo fixes ----------
# strip whitespace and normalize text columns to lower for consistent comparison
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype(str).str.strip()

# handle a known typo and unify capitalization
if 'InternetService' in df.columns:
    df['InternetService'] = df['InternetService'].replace({'Fiber opticalal': 'Fiber Optic'})

# ---------- 4. Handle missing values for specific cols ----------
# keep your original approach but safe: fill MultipleLines if present
if 'MultipleLines' in df.columns:
    df['MultipleLines'] = df['MultipleLines'].replace({'nan': np.nan}).fillna('Not known')

# ---------- 5. Safely convert numeric columns ----------
# use to_numeric to avoid errors from bad strings, then impute medians
for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')          # bad values -> NaN
        df[col] = df[col].fillna(df[col].median())               # numeric imputation (median)

# ---------- 6. Remove duplicates ----------
df = df.drop_duplicates()

# ---------- 7. Outlier detection & simple handling (IQR capping) ----------
def cap_iqr(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(lower=q1 - k*iqr, upper=q3 + k*iqr)

num_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges'] if c in df.columns]
for c in num_cols:
    # show how many would be flagged by z-score > 3
    z = np.abs((df[c] - df[c].mean()) / df[c].std())
    print(f"{c} outliers (z>3):", (z>3).sum())
    # cap extremes to reduce effect (instead of drop)
    df[c] = cap_iqr(df[c])

# ---------- 8. Feature engineering ----------
if 'Contract' in df.columns:
    df['Contract_Renewal'] = df['Contract'].apply(lambda x: 'Yes' if str(x) in ['One year','Two year','1 Year','2 Year'] else 'No')

# ---------- 9 & 10. Prepare target, split, and scale (no leakage) ----------
# map churn to 0/1 if possible
if 'Churn' in df.columns:
    y = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
else:
    raise KeyError("Target column 'Churn' not found")

X = df.drop(columns=['Churn'])

# split BEFORE scaling to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# scale numerical features: fit on train only
scaler = StandardScaler()
if num_cols:
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num  = scaler.transform(X_test[num_cols])
    # put scaled numbers back (keep other columns as-is)
    X_train.loc[:, num_cols] = X_train_num
    X_test.loc[:, num_cols]  = X_test_num

print("Train X shape:", X_train.shape, "Test X shape:", X_test.shape)

# ---------- 11. Export cleaned data ----------
cleaned_path = r"D:\BE_Practicals\DMVcodes\DMV_Practical\telecom_churn_cleaned.csv"
df.to_csv(cleaned_path, index=False)
print("Saved cleaned dataset to:", cleaned_path)

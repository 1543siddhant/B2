import os
import pandas as pd
import numpy as np
from scipy import stats

# ---------- 1. Load (adjust filename if needed) ----------
dataset_path = r"RealEstate_Prices.csv"
df = pd.read_csv(dataset_path, low_memory=False)
print("Initial shape:", df.shape)

# ---------- 1b. Clean column names (remove spaces & special chars) ----------
df.columns = (df.columns.str.strip()
                        .str.replace(r"[^\w\s]", "", regex=True)   # remove special chars
                        .str.replace(" ", "_"))

# ---------- 2. Quick inspect ----------
print(df.dtypes)
print("Missing values:\n", df.isnull().sum())

# ---------- 3. Safe numeric conversion & imputation ----------
for c in ['Price', 'SqFt', 'Bedrooms', 'YearBuilt', 'SaleYear']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')      # non-numeric -> NaN

# Numeric impute: Price -> median, SqFt -> median
if 'Price' in df.columns:
    df['Price'] = df['Price'].fillna(df['Price'].median())
if 'SqFt' in df.columns:
    df['SqFt'] = df['SqFt'].fillna(df['SqFt'].median())

# Categorical impute: Bedrooms (mode), others -> 'Unknown'
if 'Bedrooms' in df.columns:
    mode = df['Bedrooms'].mode()
    df['Bedrooms'] = df['Bedrooms'].fillna(mode[0] if not mode.empty else 0)

for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan})
    df[c] = df[c].fillna('Unknown')

# ---------- 4. Remove duplicates ----------
before = df.shape[0]
df = df.drop_duplicates()
print(f"Dropped {before - df.shape[0]} duplicate rows")

# ---------- 5. Optional merge (if extra file exists) ----------
neighborhoods_path = r"neighborhoods.csv"
if os.path.exists(neighborhoods_path):
    neigh = pd.read_csv(neighborhoods_path, low_memory=False)
    neigh.columns = neigh.columns.str.strip().str.replace(r"[^\w\s]", "", regex=True).str.replace(" ", "_")
    df = df.merge(neigh, on='Neighborhood', how='left')
    print("Merged neighborhoods -> new shape:", df.shape)
else:
    print("No neighborhoods.csv found — skipping merge")

# ---------- 6. Filter & subset examples ----------
# Example filter: only recent sales if SaleYear present
if 'SaleYear' in df.columns:
    df = df[df['SaleYear'] >= 2018]
# Example property filter: Houses only (if PropertyType exists)
if 'PropertyType' in df.columns:
    df = df[df['PropertyType'].str.contains('House', case=False, na=False)]

subset_df = df[['Price','SqFt','Bedrooms']].copy() if set(['Price','SqFt','Bedrooms']).issubset(df.columns) else df.head()
print("Subset sample:\n", subset_df.head())

# ---------- 7. Categorical encoding (minimal & safe) ----------
# Binary map example for 'Brick' (unknown -> -1)
if 'Brick' in df.columns:
    df['Brick'] = df['Brick'].map({'Yes':1,'No':0}).fillna(-1).astype(int)

# One-hot for low-cardinality categorical cols (<=10 unique values)
cat_cols = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() <= 10 and c not in ['Neighborhood']]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ---------- 8. Aggregate example: avg price by Neighborhood ----------
if 'Neighborhood' in df.columns and 'Price' in df.columns:
    grouped_data = df.groupby('Neighborhood')['Price'].agg(['mean','median','count']).reset_index().rename(columns={'mean':'AvgPrice','median':'MedianPrice','count':'N'})
    print("Top neighborhoods by AvgPrice:\n", grouped_data.sort_values('AvgPrice', ascending=False).head())

# ---------- 9. Outlier detection & handling ----------
if 'Price' in df.columns:
    # Report z-score-based outliers
    df['z_score'] = stats.zscore(df['Price'].astype(float))
    z_outliers_count = df['z_score'].abs().gt(3).sum()
    print("Z-score outliers (|z|>3):", z_outliers_count)

    # Safer: IQR capping
    q1, q3 = df['Price'].quantile([0.25,0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    print("IQR bounds:", lower, upper)
    df['Price'] = df['Price'].clip(lower, upper)

    # Recompute z_score after capping (optional)
    df['z_score'] = stats.zscore(df['Price'].astype(float))

# ---------- 10. Final tidy-ups and export ----------
# If you want integer prices, round then convert (safe after imputation & capping)
if 'Price' in df.columns:
    df['Price'] = df['Price'].round().astype(int)

out_path = "RealEstate_Prices_cleaned.csv"
df.to_csv(out_path, index=False)
print("Saved cleaned dataset to:", out_path)

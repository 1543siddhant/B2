import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load the main CSV (your original file)
# ---------------------------
df = pd.read_csv("1_sample.csv")
print("Initial rows:", len(df))
print(df.head())

# Basic info
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------
# Clean Sales column safely
# ---------------------------
# convert Sales to numeric; coerce bad values to NaN, then fill
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
# if you want integer sales
df['Sales'] = df['Sales'].astype(int)

# Channel cleaning: lowercase, fill missing, then encode
df['Channel'] = df['Channel'].fillna('Not known').str.strip().str.lower()
df['Channel'] = df['Channel'].replace(['offline', 'online'], [0, 1])
print("\nAfter cleaning Sales & Channel:")
print(df.head())

# ---------------------------
# Read file1 and file2, transform dates
# ---------------------------
df_file1 = pd.read_csv('1_file1.csv')
df_file2 = pd.read_csv('1_file2.csv')

# Safely parse start_date and split into Year/Month/Day
for d in (df_file1, df_file2):
    if 'start_date' in d.columns:
        d['start_date'] = pd.to_datetime(d['start_date'], errors='coerce')
        d['Year']  = d['start_date'].dt.year
        d['Month'] = d['start_date'].dt.month
        d['Day']   = d['start_date'].dt.day

print("\nTransformed file1 sample:")
print(df_file1.head())
print("\nTransformed file2 sample:")
print(df_file2.head())

# ---------------------------
# Merge file1 & file2 (concat if same schema)
# ---------------------------
merged_df = pd.concat([df_file1, df_file2], ignore_index=True)
print("\nMerged df shape:", merged_df.shape)

# ---------------------------
# Read JSON (apple.json) if present, and unify formats
# ---------------------------
import os
json_path = "apple.json"   # sample provided: /mnt/data/apple.json (download it into working dir or move)
if os.path.exists(json_path):
    df_json = pd.read_json(json_path)
    print("\nLoaded JSON file:", json_path, "rows:", len(df_json))
    print(df_json.head())
    # try to align columns: if JSON has 'Sales' and 'P Type' like CSV, concat it
    common_cols = set(df.columns).intersection(df_json.columns)
    if 'Sales' in df_json.columns:
        df_json['Sales'] = pd.to_numeric(df_json['Sales'], errors='coerce').fillna(0).astype(int)
    # ensure Channel column consistent
    if 'Channel' in df_json.columns:
        df_json['Channel'] = df_json['Channel'].fillna('Not known').str.strip().str.lower()
        df_json['Channel'] = df_json['Channel'].replace(['offline','online'], [0,1])
    # Concatenate JSON rows into main df if columns align (or at least Sales exists)
    df = pd.concat([df, df_json[df.columns.intersection(df_json.columns)]], ignore_index=True, sort=False)
    print("After adding JSON, total rows:", len(df))
else:
    print("\nNo JSON file found at", json_path, "- skipping JSON load.")

# ---------------------------
# Quick duplicate check & drop
# ---------------------------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)
print(f"\nDropped duplicates: {before-after} rows removed")

# ---------------------------
# Descriptive stats & aggregations
# ---------------------------
print("\nDescriptive statistics (Sales):")
print(df['Sales'].describe())

total_sales = df['Sales'].sum()
avg_order_value = df['Sales'].mean()
print("\nTotal Sales:", total_sales)
print("Average Order Value:", round(avg_order_value, 2))

print("\nProduct Category Distribution (counts):")
print(df['P Type'].value_counts(dropna=False))

# ---------------------------
# Aggregate example: total sales by product type
# ---------------------------
product_type_sales = df.groupby('P Type')['Sales'].sum().sort_values(ascending=False)
print("\nTotal sales by product type:\n", product_type_sales)

# ---------------------------
# Plots: bar (sales by product), pie (product distribution), boxplot (sales distribution)
# ---------------------------
plt.figure(figsize=(10,6))
product_type_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Product Type')
plt.xlabel('Product Type')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,7))
prod_counts = df['P Type'].value_counts()
plt.pie(prod_counts, labels=prod_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Product Category Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
# if many zeros from fill, you might want to ignore zeros: use replace(0, np.nan).dropna()
plt.boxplot(df['Sales'].replace(0, np.nan).dropna())
plt.title('Sales distribution (boxplot)')
plt.ylabel('Sales')
plt.show()

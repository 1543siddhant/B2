import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('uber.csv')
df.head()

# Shape & info
df.shape
df.info()
df.describe()

# Convert datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

# Drop unnecessary columns (only if they exist)
for col in ['Unnamed: 0', 'key', 'pickup_datetime']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Check missing values
df.isnull().sum()

# Fill missing values ONLY for coordinates
for col in ['dropoff_longitude','dropoff_latitude']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Remove rows where fare is missing
df = df.dropna(subset=['fare_amount'])

# --- Outlier Removal ---
sns.boxplot(x=df["fare_amount"])
plt.show()

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["fare_amount"] >= lower) & (df["fare_amount"] <= upper)]

sns.boxplot(x=df["fare_amount"])
plt.show()

# Correlation heatmap
corr = df.corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                title='Correlation Matrix')
fig.update_layout(title_x=0.5, width=900, height=600)
fig.show()

# Split data
X = df.drop(columns=['fare_amount'])
y = df['fare_amount']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.2f}")

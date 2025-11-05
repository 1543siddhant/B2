'''Feature Transformation (Any one)
A. To use PCA Algorithm for dimensionality reduction.
You have a dataset that includes measurements for different variables on wine
(alcohol, ash, magnesium, and so on). Apply PCA algorithm & transform this data
so that most variations in the measurements of the variables are captured by a small
number of principal components so that it is easier to distinguish between red and
white wine by inspecting these principal components.
Dataset Link: https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Wine.csv')

df

df.shape

# Variable descriptions
df.info()


#Checking Datatypes of each column
df.dtypes

# Get initial statistics
df.describe()

#Checking For null values
df.isnull().sum()


# Separating features and target variable
X = df.drop(columns=['Customer_Segment'])  # Dropping the Target Component for training
y = df['Customer_Segment']  # As it is the Target variable (Type of wine)


X


y



#Scaling the data using Standard Scalar

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Appling PCA with 2 components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Creating a new DataFrame with the first two principal components
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Customer_Segment'] = y


pca_df


plt.figure(figsize=(15, 6))
colors = {1: 'blue', 2: 'red', 3: 'green'}

# Scatter plot for Customer Segment 1
plt.scatter(
    X[y == 1]['Alcohol'],
    X[y == 1]['Malic_Acid'],
    color = colors[1], 
    label = 'Customer Segment 1',
    alpha = 0.7
)

# Scatter plot for Customer Segment 2
plt.scatter(
    X[y == 2]['Alcohol'],
    X[y == 2]['Malic_Acid'],
    color = colors[2], 
    label = 'Customer Segment 2',
    alpha = 0.7
)

# Scatter plot for Customer Segment 3
plt.scatter(
    X[y == 3]['Alcohol'],
    X[y == 3]['Malic_Acid'],
    color = colors[3],  
    label = 'Customer Segment 3',
    alpha = 0.7
)

# Set labels and title
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.legend()
plt.title('Scatter Plot (Original Data)')
plt.show()


plt.figure(figsize=(15, 6))
colors = {1: 'blue', 2: 'red', 3: 'green'}

# Scatter plot for Customer Segment 1
plt.scatter(
    pca_df[pca_df['Customer_Segment'] == 1]['Principal Component 1'],
    pca_df[pca_df['Customer_Segment'] == 1]['Principal Component 2'],
    color = colors[1],  # Set color for segment 1
    label = 'Customer Segment 1',
    alpha = 0.7
)

# Scatter plot for Customer Segment 2
plt.scatter(
    pca_df[pca_df['Customer_Segment'] == 2]['Principal Component 1'],
    pca_df[pca_df['Customer_Segment'] == 2]['Principal Component 2'],
    color = colors[2],  # Set color for segment 2
    label = 'Customer Segment 2',
    alpha = 0.7
)

# Scatter plot for Customer Segment 3
plt.scatter(
    pca_df[pca_df['Customer_Segment'] == 3]['Principal Component 1'],
    pca_df[pca_df['Customer_Segment'] == 3]['Principal Component 2'],
    color = colors[3],  # Set color for segment 3
    label = 'Customer Segment 3',
    alpha = 0.7
)


plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('PCA: Wine Dataset')
plt.show()


pca_new = PCA(n_components=3)
X_pca_new = pca_new.fit_transform(X_scaled)


# Creating a new DataFrame with the principal components
pca_df_new = pd.DataFrame(data = X_pca_new, columns = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])
pca_df_new['Customer_Segment'] = y


pca_df_new


# Creating a 3D scatter plot using Plotly
fig = go.Figure()

# Add scatter plots for each customer segment
for segment in pca_df_new['Customer_Segment'].unique():
    segment_data = pca_df_new[pca_df_new['Customer_Segment'] == segment]
    fig.add_trace(go.Scatter3d(
        x=segment_data['Principal Component 1'],
        y=segment_data['Principal Component 2'],
        z=segment_data['Principal Component 3'],
        mode='markers',
        marker=dict(size=5, opacity=0.7),
        name=f'Customer Segment {segment}'
    ))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='PC 1',
        yaxis_title='PC 2',
        zaxis_title='PC 3'
    ),
    title='PCA: Wine Dataset',
    width=1000,  
    height=700 
)

# Show the plot
fig.show()
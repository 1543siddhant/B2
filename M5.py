import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv(r"D:\BE_Practicals\MLcodes\car_evaluation.csv")

# Rename columns (if needed)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Quick checks
print(df.head())
print(df.info())
print(df['class'].value_counts())

# --- Simple explicit encoding (ordinal where order exists) ---
# Map ordinal categories to integers (ensure correct order)
buy_maint_map = {'low':1, 'med':2, 'high':3, 'vhigh':4}
lug_map = {'small':1, 'med':2, 'big':3}
safety_map = {'low':1, 'med':2, 'high':3}
doors_map = {'2':2, '3':3, '4':4, '5more':5}
persons_map = {'2':2, '4':4, 'more':5}

df['buying'] = df['buying'].map(buy_maint_map)
df['maint']  = df['maint'].map(buy_maint_map)
df['lug_boot'] = df['lug_boot'].map(lug_map)
df['safety'] = df['safety'].map(safety_map)
df['doors'] = df['doors'].map(doors_map)
df['persons'] = df['persons'].map(persons_map)

# If any NaNs after mapping (unexpected), drop them
df = df.dropna()

# Features and target
X = df.drop('class', axis=1)
y = df['class']   # labels are strings (e.g., 'unacc','acc','good','vgood'), RF handles these

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Train RandomForest
rfc = RandomForestClassifier(random_state=0)   # default n_estimators=100
rfc.fit(X_train, y_train)

# Predict & evaluate
y_pred = rfc.predict(X_test)
print(f"Model accuracy score with {rfc.n_estimators} decision-trees : {accuracy_score(y_test, y_pred):0.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=rfc.classes_, yticklabels=rfc.classes_, cmap='Blues')
plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.title('Confusion Matrix')
plt.show()

# Feature importances
feat_imp = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Feature importances (Random Forest)')
plt.xlabel('Importance')
plt.show()

# Optional: try 50 trees (like you did)
rfc_50 = RandomForestClassifier(n_estimators=50, random_state=0)
rfc_50.fit(X_train, y_train)
y_pred_50 = rfc_50.predict(X_test)
print(f"Model accuracy score with {rfc_50.n_estimators} decision-trees : {accuracy_score(y_test, y_pred_50):0.4f}")

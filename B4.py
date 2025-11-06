# If needed, uncomment to install:
# !pip install numpy pandas scikit-learn matplotlib
# Assignment: Machine Learning for Genomic Data. Task: Apply machine learning algorithms, such as random
# forests or support vector machines, to classify genomic data based on specific features or markers. Deliverable: A
# comprehensive analysis report presenting the classification results, model performance evaluation, and insights
# into the predictive features. 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt

# -------------------------
# Load Dataset
# -------------------------
df_dna = pd.read_csv("human.csv")   # file must contain 'sequence' and 'class' columns
# Quick checks
if "sequence" not in df_dna.columns or "class" not in df_dna.columns:
    raise ValueError("CSV must contain 'sequence' and 'class' columns")

# Drop rows with missing values
df_dna = df_dna.dropna(subset=["sequence", "class"]).reset_index(drop=True)

# Ensure sequences are uppercase and composed of A/C/G/T (basic cleaning)
df_dna["sequence"] = df_dna["sequence"].astype(str).str.upper().str.replace("[^ACGT]", "", regex=True)

# -------------------------
# Set Target Feature
# -------------------------
y = df_dna["class"]

# -------------------------
# Grouping the sequence into k-mers
# -------------------------
kmer_length = 4
def seq_to_kmers(seq, k):
    return [seq[i:i+k] for i in range(0, len(seq) - k + 1)] if len(seq) >= k else []

df_dna["words"] = df_dna["sequence"].apply(lambda seq: seq_to_kmers(seq, kmer_length))

# Convert k-mer lists into space-joined strings for CountVectorizer
df_dna["kmer_text"] = df_dna["words"].apply(lambda x: " ".join(x))

# -------------------------
# Vectorize the k-mers (sparse matrix)
# -------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_dna["kmer_text"])  # X is sparse CSR matrix

# -------------------------
# Train / Validation / Test split (stratified)
# -------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# -------------------------
# Optional scaling for linear SVM (MaxAbsScaler works with sparse data)
# -------------------------
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Training and Validating Random Forest Classifier
# -------------------------
# RandomForest in scikit-learn accepts dense arrays better; convert (watch memory for large data)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train.toarray(), y_train)   # convert to dense for RF

# Validate Random Forest
y_predicted_valid_rf = rf_classifier.predict(X_valid.toarray())
accuracy_valid_rf = accuracy_score(y_valid, y_predicted_valid_rf)

# -------------------------
# Training and Validating SVM (linear)
# -------------------------
svm_classifier = LinearSVC(random_state=42, max_iter=10000)
svm_classifier.fit(X_train_scaled, y_train)

# Validate SVM
y_predicted_valid_svm = svm_classifier.predict(X_valid_scaled)
accuracy_valid_svm = accuracy_score(y_valid, y_predicted_valid_svm)

# -------------------------
# Testing both models on the test set
# -------------------------
y_predicted_test_rf = rf_classifier.predict(X_test.toarray())
accuracy_test_rf = accuracy_score(y_test, y_predicted_test_rf)

y_predicted_test_svm = svm_classifier.predict(X_test_scaled)
accuracy_test_svm = accuracy_score(y_test, y_predicted_test_svm)

# -------------------------
# Print model accuracy and reports
# -------------------------
print("Validation Accuracy - Random Forest:", accuracy_valid_rf)
print("Validation Accuracy - SVM:", accuracy_valid_svm)
print("Test Accuracy - Random Forest:", accuracy_test_rf)
print("Test Accuracy - SVM:", accuracy_test_svm)
print("\nRandom Forest classification report (test):\n", classification_report(y_test, y_predicted_test_rf))
print("SVM classification report (test):\n", classification_report(y_test, y_predicted_test_svm))
print("Random Forest confusion matrix (test):\n", confusion_matrix(y_test, y_predicted_test_rf))
print("SVM confusion matrix (test):\n", confusion_matrix(y_test, y_predicted_test_svm))

# -------------------------
# Feature importance (Random Forest) - show top k-mers
# -------------------------
feature_names = vectorizer.get_feature_names_out()
importances = rf_classifier.feature_importances_
top_idx = np.argsort(importances)[-20:][::-1]   # top 20
print("\nTop k-mer features by Random Forest importance:")
for idx in top_idx:
    if importances[idx] > 0:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# -------------------------
# Plotting accuracy comparison
# -------------------------
models = ['Random Forest', 'SVM']
validation_accuracies = [accuracy_valid_rf, accuracy_valid_svm]
test_accuracies = [accuracy_test_rf, accuracy_test_svm]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, validation_accuracies, width, label='Validation Accuracy')
rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0,1)
ax.legend()

for i, v in enumerate(validation_accuracies):
    ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
for i, v in enumerate(test_accuracies):
    ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.show()

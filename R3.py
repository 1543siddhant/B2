import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset (handle common encodings / common column names)
# Make sure "spam.csv" is in the working directory.
# Typical SMS Spam Collection has columns like 'v1' (label) and 'v2' (text),
# but your code expects 'Category' and 'Message'. We try both.
spam_df = pd.read_csv("spam.csv", encoding="latin-1", low_memory=False)

# Robustness: if different column names are present, rename to expected ones
if 'Category' not in spam_df.columns or 'Message' not in spam_df.columns:
    # common SMS dataset column names: 'v1' -> label, 'v2' -> text
    if 'v1' in spam_df.columns and 'v2' in spam_df.columns:
        spam_df = spam_df.rename(columns={'v1': 'Category', 'v2': 'Message'})
    else:
        # If dataset already uses 'label'/'text' style
        possible_label_cols = [c for c in spam_df.columns if 'label' in c.lower() or 'cat' in c.lower()]
        possible_text_cols = [c for c in spam_df.columns if 'message' in c.lower() or 'text' in c.lower()]
        if possible_label_cols and possible_text_cols:
            spam_df = spam_df.rename(columns={possible_label_cols[0]: 'Category', possible_text_cols[0]: 'Message'})
        else:
            raise ValueError("Couldn't find 'Category'/'Message' or common alternatives in spam.csv columns.")

# 2. Inspect (uncomment if you want to view)
# print(spam_df.head())
# print(spam_df.groupby('Category').size())

# 3. Create binary label: 1 => spam, 0 => ham
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if str(x).strip().lower() == 'spam' else 0)

# 4. Train-test split (keep close to original but add stratify and random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    spam_df.Message,
    spam_df.spam,
    test_size=0.25,
    stratify=spam_df.spam,   # keep class distribution in both sets
    random_state=42
)

# 5. Vectorize text with CountVectorizer (Bag-of-Words)
cv = CountVectorizer()   # you can later add stop_words='english' or ngram_range=(1,2)
x_train_count = cv.fit_transform(X_train.values)

# 6. Train Multinomial Naive Bayes
model = MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing (default)
model.fit(x_train_count, y_train)

# 7. Evaluate on test set
x_test_count = cv.transform(X_test)
y_pred = model.predict(x_test_count)

print("Accuracy (test):", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Simple live predictions (same style as your original examples)
email_ham = ['Hey how are you?']
email_ham_count = cv.transform(email_ham)
print("email_ham predicted label (0=ham,1=spam):", model.predict(email_ham_count))

email_spam = ["Free entry "]
email_spam_count = cv.transform(email_spam)
print("email_spam predicted label (0=ham,1=spam):", model.predict(email_spam_count))

# 9. Sample from user
sample = ["HI how are you"]
sample_vec = cv.transform(sample)
p = model.predict(sample_vec)
if p == 0:
    print("\nPrediction for sample: Ham")
else:
    print("\nPrediction for sample: Spam")  # 1 means spam

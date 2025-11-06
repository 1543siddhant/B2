# If using this for the first time, uncomment the downloads below.
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# -----------------------------
# Example: stop-word removal
# -----------------------------
example_sent = """This is a sample sentence,
                showing off the stop words filtration."""

# Build stopword set
stop_words = set(stopwords.words('english'))

# Tokenize (punkt tokenizer)
word_tokens = word_tokenize(example_sent)

# Remove stopwords and punctuation (case-insensitive)
filtered_sentence = [w for w in word_tokens
                     if w.lower() not in stop_words and w not in string.punctuation]

print("Original tokens:", word_tokens)
print("After stopword & punctuation removal:", filtered_sentence)

# -----------------------------
# Stemming (Porter) on a dataset
# -----------------------------
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

dataset = ["The quick brown fox jumps over the lazy dog",
           "Python is a high-level programming language",
           "Data science is an interdisciplinary field"]

stemmer = PorterStemmer()
stemmed_dataset = []
for sentence in dataset:
    words = word_tokenize(sentence)
    # optional: remove stopwords and punctuation before stemming
    words = [w for w in words if w.lower() not in stop_words and w not in string.punctuation]
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_sentence = " ".join(stemmed_words)
    stemmed_dataset.append(stemmed_sentence)

print("\nStemmed dataset:")
for s in stemmed_dataset:
    print(s)

# -----------------------------
# Additional step: Lemmatization
# -----------------------------
lemmatizer = WordNetLemmatizer()
lemmatized_dataset = []
for sentence in dataset:
    words = word_tokenize(sentence)
    words = [w for w in words if w.lower() not in stop_words and w not in string.punctuation]
    # simple lemmatization (default pos='n' for noun). For better results use POS tagging.
    lem_words = [lemmatizer.lemmatize(w.lower()) for w in words]
    lemmatized_dataset.append(" ".join(lem_words))

print("\nLemmatized dataset:")
for s in lemmatized_dataset:
    print(s)

# -----------------------------
# Optional: wrap as reusable function
# -----------------------------
def preprocess_text(text, do_lower=True, remove_stopwords=True,
                    remove_punct=True, do_stem=False, do_lemma=False):
    tokens = word_tokenize(text)
    if do_lower:
        tokens = [t.lower() for t in tokens]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    if remove_punct:
        tokens = [t for t in tokens if t not in string.punctuation]
    if do_stem:
        tokens = [stemmer.stem(t) for t in tokens]
    if do_lemma:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

print("\nPreprocess (example):", preprocess_text(example_sent, do_stem=True))

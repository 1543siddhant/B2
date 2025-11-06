# Simple Inverted-File Retrieval (keeps style close to your original)
from nltk.tokenize import RegexpTokenizer
import string
from collections import defaultdict, Counter

# Fallback small stopword list (so code runs without nltk downloads)
FALLBACK_STOPWORDS = {
    "a","an","the","is","are","was","were","to","of","in","and","or","on","for","with","such","this","that","it","be","as","by"
}

tokenizer = RegexpTokenizer(r'\w+')   # splits on non-word chars (no punkt required)
stop_words = FALLBACK_STOPWORDS

# Documents: [doc_id, text]
doc_dict = [
    ["d1", "Welcome to hotel heaven such a lovely place"],
    ["d2", "She is buying a stairway to heaven"],
    ["d3", "Don't make it bad"],
    ["d4", "Take me to the heaven"]
]

def normalize_and_filter(text):
    tokens = tokenizer.tokenize(text)                     # tokenize
    filtered = [t.lower() for t in tokens                 # lowercase
                if t.lower() not in stop_words]           # remove stopwords
    return filtered                                       # keep duplicates for freq if needed

# Build per-doc token lists and an inverted index: term -> {doc_id: freq}
doc_tokens = {}
inverted = defaultdict(lambda: defaultdict(int))  # inverted[term][doc_id] = freq

for doc_id, text in doc_dict:
    toks = normalize_and_filter(text)
    doc_tokens[doc_id] = set(toks)
    freqs = Counter(toks)
    for term, f in freqs.items():
        inverted[term][doc_id] += f

# Simple query handling
query = input("Enter query: ").strip()
q_tokens = normalize_and_filter(query)
q_set = set(q_tokens)

# Boolean retrieval
any_match_docs = set()
all_match_docs = set(doc_tokens.keys())
for t in q_set:
    postings = set(inverted.get(t, {}).keys())
    any_match_docs |= postings
    all_match_docs &= postings

# Simple ranking by overlap count (how many query terms appear in each doc)
scores = Counter()
for t in q_set:
    for doc_id in inverted.get(t, {}):
        scores[doc_id] += 1

ranked = [doc for doc, _ in scores.most_common()]

print("Query tokens:", q_tokens)
print("Any-match documents (at least one token):", sorted(any_match_docs))
print("All-match documents (contain all query tokens):", sorted(all_match_docs))
print("Ranked documents by overlap (best first):", ranked)

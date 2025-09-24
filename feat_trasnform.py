import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from numpy.linalg import norm

# ========================
# 1. Load Dataset
# ========================
file_path = "C:/Users/Admin/.cache/kagglehub/datasets/saurabhshahane/fake-news-classification/versions/77/WELFake_Dataset.csv"
df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print("Columns:", df.columns)

# ========================
# 2. Combine Title and Text
# ========================
df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Extract features and labels
X_text = df['combined_text']
y = df['label']

print(f"Total Samples: {len(X_text)}")
print("\nExample combined text:\n", X_text.iloc[0])

# ========================
# 3. Apply TF-IDF Vectorization
# ========================
tfidf = TfidfVectorizer(
    max_features=5000,        # Limit vocabulary size
    ngram_range=(1, 2),       # Unigrams + Bigrams
    stop_words='english',     # Remove common English words
    min_df=2,                 # Ignore terms in fewer than 2 documents
    max_df=0.8,               # Ignore very frequent terms
    sublinear_tf=True         # Apply sublinear scaling
)

X = tfidf.fit_transform(X_text)
print(f"\n✅ TF-IDF Transformation Complete!")
print(f"Matrix Shape: {X.shape}")

# ========================
# 4. Save Transformed Data
# ========================
os.makedirs("data/transformed", exist_ok=True)

with open("data/transformed/X.pkl", "wb") as f:
    pickle.dump(X, f)

with open("data/transformed/y.pkl", "wb") as f:
    pickle.dump(y, f)

with open("data/transformed/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\n✅ Feature transformation complete. Files saved in 'data/transformed/'")

# Check norms for first 5 rows
for i in range(5):
    print(f"Row {i} norm:", norm(X[i].toarray()))


# ========================
# 5. Reusable Helper Function
# ========================
def create_tfidf_features(train_text, test_text, ngram_range=(1, 2), max_features=5000, use_stemming=False):
    """
    Creates TF-IDF features for train and test text data.

    Args:
        train_text (Series): Training text
        test_text (Series): Testing text
        ngram_range (tuple): N-gram range
        max_features (int): Maximum features
        use_stemming (bool): Apply stemming if True

    Returns:
        X_train, X_test, tfidf_vectorizer
    """
    if use_stemming:
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        train_text = train_text.apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
        test_text = test_text.apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)

    return X_train, X_test, tfidf

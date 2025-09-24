import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from feat_trasnform import create_tfidf_features  # Ensure correct import

# ========================
# 1. Load Data
# ========================
file_path = "C:/Users/Admin/.cache/kagglehub/datasets/saurabhshahane/fake-news-classification/versions/77/WELFake_Dataset.csv"
df = pd.read_csv(file_path)

X = df['title'].fillna('') + " " + df['text'].fillna('')
y = df['label'].astype(int)

# Split
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# 2. Create TF-IDF Features
# ========================
X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train_text, X_test_text)

# ========================
# 3. Feature Selection using Chi2
# ========================
k = 3000  # Number of features to select
selector = SelectKBest(chi2, k=min(k, X_train_tfidf.shape[1]))
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = selector.transform(X_test_tfidf)

print("✅ Feature Selection Complete")
print(f"Original shape: {X_train_tfidf.shape}, Reduced shape: {X_train_selected.shape}")

# ========================
# 4. Save Outputs
# ========================
os.makedirs("data/selected_features", exist_ok=True)

with open("data/selected_features/X_train_selected.pkl", "wb") as f:
    pickle.dump(X_train_selected, f)

with open("data/selected_features/X_test_selected.pkl", "wb") as f:
    pickle.dump(X_test_selected, f)

with open("data/selected_features/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("data/selected_features/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

with open("data/selected_features/selector.pkl", "wb") as f:
    pickle.dump(selector, f)

with open("data/selected_features/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Feature-selected data saved in 'data/selected_features/'")

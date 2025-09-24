import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

# ========================
# 1. Load Transformed Data
# ========================
with open("data/transformed/X.pkl", "rb") as f:
    X = pickle.load(f)

with open("data/transformed/y.pkl", "rb") as f:
    y = pickle.load(f)

print(" Loaded transformed data!")
print(f"Feature matrix shape: {X.shape}")
print(f"Labels length: {len(y)}")

# ========================
# 2. Split into Train/Test
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n Data split completed:")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ========================
# 3. Optional: Dimensionality Reduction
# ========================
# Using TruncatedSVD for sparse TF-IDF matrices (like PCA for sparse data)
apply_dim_reduction = False  # Change to True if you want dimensionality reduction

if apply_dim_reduction:
    print("\nApplying TruncatedSVD for dimensionality reduction...")
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)
    print(f"Reduced feature shape: Train={X_train.shape}, Test={X_test.shape}")

# ========================
# 4. Save the Splits
# ========================
os.makedirs("data/features", exist_ok=True)

with open("data/features/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("data/features/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("data/features/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("data/features/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("\n Feature extraction complete. Files saved in 'data/features/'")

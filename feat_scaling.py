import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

# ========================
# 1. Load Selected Features
# ========================
with open("data/selected_features/X_train_selected.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("data/selected_features/X_test_selected.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("data/selected_features/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("data/selected_features/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print("✅ Loaded selected features for scaling")
print(f"Original Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ========================
# 2. Choose Scaling Method
# ========================
# Options: "minmax", "standard", "maxabs"
scaling_method = "maxabs"  # Best for sparse data like TF-IDF

if scaling_method == "minmax":
    print("\nApplying Min-Max Scaling (range 0-1)...")
    scaler = MinMaxScaler()
    # Convert to dense for MinMax
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)

elif scaling_method == "standard":
    print("\nApplying Standard Scaling (mean=0, std=1)...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    # Convert to dense for StandardScaler
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)

elif scaling_method == "maxabs":
    print("\nApplying MaxAbs Scaling (preserves sparsity)...")
    scaler = MaxAbsScaler()
    # Works with sparse data, no need to convert
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

else:
    raise ValueError("Invalid scaling method! Choose 'minmax', 'standard', or 'maxabs'.")

print("✅ Scaling complete")
print(f"Scaled Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
print(f"First 10 scaled values (train sample): {X_train_scaled[0].toarray()[0][:10] if hasattr(X_train_scaled[0], 'toarray') else X_train_scaled[0][:10]}")

# ========================
# 3. Save Scaled Features & Scaler
# ========================
os.makedirs("data/scaled_features", exist_ok=True)

with open("data/scaled_features/X_train_scaled.pkl", "wb") as f:
    pickle.dump(X_train_scaled, f)

with open("data/scaled_features/X_test_scaled.pkl", "wb") as f:
    pickle.dump(X_test_scaled, f)

with open("data/scaled_features/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("data/scaled_features/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

with open("data/scaled_features/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Scaled features saved in 'data/scaled_features/'")

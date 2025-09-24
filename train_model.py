import pickle
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========================
# 1. Load Scaled Features
# ========================
print("📂 Loading scaled features...")
with open("data/scaled_features/X_train_scaled.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("data/scaled_features/X_test_scaled.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("data/scaled_features/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("data/scaled_features/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print(f"✅ Features Loaded: Train={X_train.shape}, Test={X_test.shape}")

# ========================
# 2. Define Models
# ========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": MultinomialNB()  # Requires dense data
}

results = {}
best_model = None
best_accuracy = 0

# ========================
# 3. Train and Evaluate All Models
# ========================

# ========================
# 3. Train and Evaluate All Models
# ========================
for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    if name == "Naive Bayes":
        # ✅ Use feature-selected data (non-negative) instead of scaled
        with open("data/selected_features/X_train_selected.pkl", "rb") as f:
            X_train_input = pickle.load(f)
        with open("data/selected_features/X_test_selected.pkl", "rb") as f:
            X_test_input = pickle.load(f)
    else:
        # ✅ Use scaled features for other models
        X_train_input = X_train
        X_test_input = X_test

    # Train model
    model.fit(X_train_input, y_train)

    # Predict
    y_pred = model.predict(X_test_input)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

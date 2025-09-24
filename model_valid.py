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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

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
    "Naive Bayes": MultinomialNB()
}

results = {}
best_model = None
best_accuracy = 0

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


# ========================
# 4. Save Best Model
# ========================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f"\n✅ Best Model: {type(best_model).__name__} with Accuracy: {best_accuracy:.4f}")
print("Model saved in 'models/best_model.pkl'")

# ========================
# 5. Summary of Accuracies
# ========================
print("\n📊 Accuracy Summary:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# ✅ Bar Chart for Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

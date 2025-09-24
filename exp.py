import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.preprocessing import normalize

# ========================
# 1. Load Best Model & Vectorizer
# ========================
print("📂 Loading best model and vectorizer...")
best_model = joblib.load("models/best_model.pkl")

with open("data/transformed/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

print(f"✅ Best Model: {type(best_model).__name__}")

# Get feature names from TF-IDF
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

# ========================
# 2. Feature Importance for Linear Models (Logistic Regression)
# ========================
if hasattr(best_model, "coef_"):
    print("\n🔍 Logistic Regression Feature Importance...")
    coef = best_model.coef_[0]

    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(coef))[::-1][:20]
    top_features = feature_names[sorted_idx]
    top_importance = coef[sorted_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importance, y=top_features, palette="coolwarm")
    plt.title("Top 20 Features - Logistic Regression")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.show()

# ========================
# 3. Feature Importance for Tree-Based Models (Random Forest)
# ========================
if hasattr(best_model, "feature_importances_"):
    print("\n🔍 Random Forest Feature Importance...")
    importances = best_model.feature_importances_

    sorted_idx = np.argsort(importances)[::-1][:20]
    top_features = feature_names[sorted_idx]
    top_importance = importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importance, y=top_features, palette="viridis")
    plt.title("Top 20 Features - Random Forest")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show()

# ========================
# 4. SHAP for Global & Local Interpretability
# ========================
print("\n🔍 Running SHAP Analysis...")
explainer = shap.Explainer(best_model, feature_names=feature_names)

# SHAP values for a subset (first 100 samples for speed)
with open("data/scaled_features/X_test_scaled.pkl", "rb") as f:
    X_test = pickle.load(f)

# Convert sparse to dense if needed
X_test_input = X_test.toarray() if hasattr(X_test, "toarray") else X_test
X_sample = X_test_input[:100]

shap_values = explainer(X_sample)

# Summary plot (global importance)
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=True)

# Force plot for a single prediction (local explanation)
shap.initjs()
shap.force_plot(shap_values[0], X_sample[0], feature_names=feature_names)

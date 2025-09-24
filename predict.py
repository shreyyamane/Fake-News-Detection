import pickle
import joblib
import numpy as np

# ========================
# Load Model & Transformers
# ========================
print("📂 Loading model and transformers...")
with open("data/transformed/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("data/selected_features/selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open("data/scaled_features/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

best_model = joblib.load("models/best_model.pkl")
print("✅ Model, selector, and vectorizer loaded successfully!")


# ========================
# Prediction Function
# ========================
def predict_fake_news(title, text):
    """
    Predict whether a news article is fake or real.

    Args:
        title (str): The title of the news.
        text (str): The body text of the news.

    Returns:
        label (str): 'FAKE' or 'REAL'
        confidence (float): Probability of the prediction
    """
    combined_text = title + " " + text

    X_tfidf = tfidf_vectorizer.transform([combined_text])
    X_selected = selector.transform(X_tfidf)
    X_scaled = scaler.transform(X_selected)

    prediction = best_model.predict(X_scaled)[0]
    label = "FAKE" if prediction == 0 else "REAL"

    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(X_scaled)[0]
        confidence = float(np.max(prob))
    else:
        confidence = None

    return label, confidence


# ========================
# Test Prediction
# ========================
if __name__ == "__main__":
    title = "LAW ENFORCEMENT ON HIGH ALERT Following Threats Against Cops And Whites On 9-11 ."
    text = "No comment is expected from Barack Obama Members of the #FYF911 or #FukYoFlag..."

    label, confidence = predict_fake_news(title, text)
    if confidence:
        print(f"Prediction: {label}, Confidence: {confidence:.2f}")
    else:
        print(f"Prediction: {label}")

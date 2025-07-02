import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Paths to reuse saved objects (or train again if needed)
MODEL_PATH = "models/xgboost_model.pkl"
TFIDF_PATH = "models/tfidf.pkl23"
SVD_PATH = "models/svd.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_pipeline():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, tfidf, svd, scaler

def simulate_review_cases():
    return pd.DataFrame([
        {
            "scenario": "Short positive verified",
            "clean_text": "Excellent product. Works perfectly.",
            "verified": 1,
            "helpfulness_ratio": 0.9
        },
        {
            "scenario": "Long negative unverified",
            "clean_text": "This is the worst product I have ever purchased. It broke after one use. Waste of money.",
            "verified": 0,
            "helpfulness_ratio": 0.1
        },
        {
            "scenario": "Helpful neutral review",
            "clean_text": "It's okay, not great but not terrible either. Might work better for someone else.",
            "verified": 1,
            "helpfulness_ratio": 0.8
        },
        {
            "scenario": "Unverified long positive",
            "clean_text": "Absolutely love this! Amazing performance and excellent build quality. Highly recommend to everyone.",
            "verified": 0,
            "helpfulness_ratio": 0.2
        },
        {
            "scenario": "Verified short neutral",
            "clean_text": "It works.",
            "verified": 1,
            "helpfulness_ratio": 0.5
        }
    ])

def simulate_predictions():
    model, tfidf, svd, scaler = load_pipeline()
    df = simulate_review_cases()

    # Vectorize and reduce
    tfidf_matrix = tfidf.transform(df["clean_text"])
    svd_matrix = svd.transform(tfidf_matrix)

    # Scale other features
    numeric = scaler.transform(df[["verified", "helpfulness_ratio"]])

    # Final feature set
    X_sim = np.hstack([svd_matrix, numeric])
    df["prob_5_star"] = model.predict_proba(X_sim)[:, 1]

    print("\n Simulated Predictions:")
    print(df[["scenario", "prob_5_star"]])

    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/simulation_results.csv", index=False)
    print("\n Results saved to data/results/simulation_results.csv")

if __name__ == "__main__":
    simulate_predictions()

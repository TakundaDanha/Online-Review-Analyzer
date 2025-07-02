import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def load_and_preprocess():
    df = pd.read_csv("data/processed/cleaned_reviews.csv")
    df["label"] = (df["Score"] == 5).astype(int)
    df["verified"] = df["verified"].map({"Yes": 1, "No": 0})
    df["clean_text"] = df["clean_text"].fillna("")
    df["helpfulness_ratio"] = df["helpfulness_ratio"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        df[["clean_text", "verified", "helpfulness_ratio"]],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train["clean_text"])
    X_test_tfidf = tfidf.transform(X_test["clean_text"])

    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    return X_train_svd, X_test_svd, tfidf, svd

def scale_numeric(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[["verified", "helpfulness_ratio"]])
    X_test_scaled = scaler.transform(X_test[["verified", "helpfulness_ratio"]])
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{name}:\n")
    print(classification_report(y_test, y_pred, digits=3))
    return metrics, y_pred, y_proba

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Feature Engineering
    X_train_svd, X_test_svd, tfidf, svd = vectorize_text(X_train, X_test)
    X_train_num, X_test_num, scaler = scale_numeric(X_train, X_test)

    # Final feature matrix
    X_train_final = np.hstack([X_train_svd, X_train_num])
    X_test_final = np.hstack([X_test_svd, X_test_num])

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    all_metrics = []
    os.makedirs("data/results", exist_ok=True)

    for name, model in models.items():
        metrics, y_pred, y_proba = train_and_evaluate_model(
            name, model, X_train_final, y_train, X_test_final, y_test
        )
        all_metrics.append(metrics)

        preds_df = pd.DataFrame({
            "actual": y_test.reset_index(drop=True),
            "predicted": y_pred,
            "probability": y_proba
        })
        preds_df.to_csv(f"data/results/predictions_{name.replace(' ', '_')}.csv", index=False)

    # Save model performance
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("data/results/model_metrics.csv", index=False)
    print("\nSaved all model results to data/results/")

    # Save best model and pipeline components
    best_model_name = "XGBoost"
    best_model = models[best_model_name]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/xgboost_model.pkl")
    joblib.dump(tfidf, "models/tfidf.pkl23")
    joblib.dump(svd, "models/svd.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"\n Saved best model ({best_model_name}) and pipeline components to /models/")

if __name__ == "__main__":
    main()

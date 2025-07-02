import pandas as pd
import numpy as np
import string
import random
import os

def clean_text(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

def add_verified_flag(df, prob_verified=0.7):
    # Add synthetic 'verified' flag with given probability
    df['verified'] = df['UserId'].apply(lambda _: 'Yes' if random.random() < prob_verified else 'No')
    return df

def main():
    raw_path = "data/raw/Reviews.csv"
    processed_path = "data/processed/cleaned_reviews.csv"
    
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    print("Converting timestamp to datetime...")
    df["review_date"] = pd.to_datetime(df["Time"], unit='s')
    
    print("Calculating helpfulness ratio...")
    df["helpfulness_ratio"] = np.where(
        df["HelpfulnessDenominator"] > 0,
        df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"],
        0.0
    )
    
    print("Combining summary and full text...")
    df["full_text"] = df["Summary"].fillna("") + ". " + df["Text"].fillna("")
    
    print("Cleaning text...")
    df["clean_text"] = df["full_text"].apply(clean_text)
    
    print("Adding synthetic 'verified' flag...")
    df = add_verified_flag(df)
    
    # Select columns for modeling
    columns_to_keep = [
        "Id", "ProductId", "UserId", "verified", "Score", "review_date",
        "helpfulness_ratio", "clean_text"
    ]
    df_clean = df[columns_to_keep]
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    print(f"Saving cleaned data to {processed_path}...")
    df_clean.to_csv(processed_path, index=False)
    
    print("Cleaning complete1.")

if __name__ == "__main__":
    main()

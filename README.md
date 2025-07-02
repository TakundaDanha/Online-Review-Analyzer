# 🧠 Online Review Bias Analyzer

A full-stack data science and engineering project that scrapes online product reviews, performs statistical hypothesis testing, applies Monte Carlo simulations, and builds machine learning models to detect review bias and predict fraud.

---

## 📌 Project Description

Online product reviews influence billions in consumer purchases, but are they always reliable?

This project investigates:
- Whether early reviews differ statistically from later ones
- Whether verified reviews are more positive than unverified ones
- If there are detectable patterns in potentially **fake or biased reviews**
- How a product’s average rating might evolve using **Monte Carlo simulations**
- What features (e.g. sentiment, review length, timestamps) predict rating or fraud

The pipeline includes:
- **Web scraping** reviews from live e-commerce platforms (or from public datasets)
- **PySpark & Pandas** processing of large, messy review data
- **Statistical testing** (A/B tests, bootstrapping, t-tests)
- **Text feature engineering** (TF-IDF, sentiment analysis, SVD)
- **Predictive modeling** (classification, regression, regularization)
- **Simulation modeling** (Monte Carlo to simulate future ratings)
- **Model evaluation** with cross-validation, ROC-AUC, accuracy, and confusion matrices

---

## 🛠️ Tech Stack

| Area             | Tools / Libraries                              |
|------------------|------------------------------------------------|
| Web Scraping     | `BeautifulSoup`, `requests`, `Selenium`        |
| Data Processing  | `pandas`, `pyspark`                            |
| NLP              | `nltk`, `scikit-learn`, `textblob`, `spacy`    |
| Stats/Sim        | `scipy`, `numpy`, `bootstrapped`, `matplotlib` |
| ML Modeling      | `scikit-learn`, `xgboost`, `pyspark.ml`        |
| Matrix Ops       | `TruncatedSVD`, `PCA`                          |
| Visualization    | `matplotlib`, `seaborn`, `plotly`              |

---

## 📁 Project Structure

online-review-analyzer/
├── data/
│ ├── raw/ # Scraped or downloaded raw review data
│ └── processed/ # Cleaned and feature-engineered data
├── notebooks/
│ ├── eda.ipynb # Exploratory analysis, SVD, plots
│ └── modeling.ipynb # Classifier comparisons, ROC curves
├── src/
│ ├── scrape_reviews.py # Scrape reviews from e-commerce pages
│ ├── clean.py # Cleaning, deduping, parsing, TF-IDF
│ ├── ab_test.py # Hypothesis testing (verified vs not)
│ ├── simulate.py # Monte Carlo simulation of future ratings
│ ├── train.py # Classification/regression ML pipeline
│ └── utils.py # Shared functions (text processing, stats)
├── requirements.txt
└── README.md


---

## 🔬 Example Questions Answered

- Do verified purchase reviews differ significantly from unverified ones?
- Is review sentiment always aligned with the star rating?
- Do reviews grow more negative or positive over time?
- What features are most predictive of a 1-star review?
- Can we simulate how a product's average rating might look after 1000 more reviews?

---

## 📈 Machine Learning Models Used

- Logistic Regression (with L1/L2 regularization)
- Random Forests
- XGBoost
- SVM with text features
- Naive Bayes on text (baseline)
- PCA/SVD for dimensionality reduction

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scrape or download data
python src/scrape_reviews.py  # or use a public dataset

# 3. Clean and transform
python src/clean.py

# 4. Run statistical tests
python src/ab_test.py

# 5. Simulate future ratings
python src/simulate.py

# 6. Train models and evaluate
python src/train.py

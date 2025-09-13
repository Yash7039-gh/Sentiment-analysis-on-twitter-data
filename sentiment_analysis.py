"""
Sentiment Analysis on Tweets (Logistic Regression)

This script uses a small built-in dataset so it runs offline. Replace the sample
data with your own CSV or connect to Twitter API if you want live tweets.

Usage:
    pip install -r requirements.txt
    python sentiment_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def get_sample_data():
    data = [
        ("I love this product, amazing experience!", 1),
        ("Worst service ever, very disappointed.", 0),
        ("I feel happy today!", 1),
        ("This is so bad, I hate it.", 0),
        ("Wonderful support team, very satisfied.", 1),
        ("Terrible experience, not recommended.", 0),
        ("Absolutely fantastic! Will buy again.", 1),
        ("Not good — the quality was poor.", 0),
        ("I'm extremely pleased with the results.", 1),
        ("This ruined my day, very upset.", 0),
    ]
    df = pd.DataFrame(data, columns=["tweet", "label"])
    return df

def main():
    # Load data
    df = get_sample_data()
    print("Sample dataset loaded with", len(df), "rows.")

    X = df["tweet"].values
    y = df["label"].values

    # Vectorize - try TF-IDF for better results
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Test on new examples
    examples = [
        "I hate waiting, worst experience!",
        "Absolutely love this service, very fast.",
        "Not satisfied, product broke quickly.",
        "Best purchase I've made this year!"
    ]
    ex_vec = vectorizer.transform(examples)
    preds = clf.predict(ex_vec)
    for t, p in zip(examples, preds):
        label = "Positive" if p==1 else "Negative"
        print(f"[{label}] {t}")

if __name__ == "__main__":
    main()

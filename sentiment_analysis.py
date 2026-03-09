# ==============================
# Sentiment Analysis using NLP
# ==============================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')


def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()

    words = [w for w in words if w not in stopwords.words('english')]

    return " ".join(words)


def main():

    print("Loading Dataset...")

    df = pd.read_csv("reviews.csv")

    print(df.head())

    # ======================
    # Data Preprocessing
    # ======================

    print("\nCleaning Text Data...")

    df['clean_text'] = df['tweet'].apply(clean_text)

    # ======================
    # Feature Extraction
    # ======================

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df['clean_text'])

    y = df['label']

    # ======================
    # Train Test Split
    # ======================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================
    # Model Training
    # ======================

    print("\nTraining Model...")

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # ======================
    # Prediction
    # ======================

    y_pred = model.predict(X_test)

    # ======================
    # Evaluation
    # ======================

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ======================
    # Confusion Matrix
    # ======================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("\nSentiment Analysis Completed!")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import re
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

try:
    df = pd.read_csv('data.csv')
    df.dropna(subset=['Review text'], inplace=True)
    df = df[df['Ratings'] != 3]
    df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
except FileNotFoundError:
    print("Data not found!")
    exit()

def clean_text(text):
    text = text.replace("READ MORE", "")
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

df['cleaned_text'] = df['Review text'].apply(clean_text)

mlflow.set_experiment("Flipkart_Sentiment_Analysis")

c_values = [0.1, 1.0, 10.0]
max_features_list = [3000, 5000]

for c_val in c_values:
    for max_feat in max_features_list:
        
        run_name = f"LR_C-{c_val}_Feat-{max_feat}"
        with mlflow.start_run(run_name=run_name):
            
            print(f"Training with C={c_val}, max_features={max_feat}...")

            vectorizer = CountVectorizer(max_features=max_feat, ngram_range=(1, 2), stop_words=None)
            X = vectorizer.fit_transform(df['cleaned_text'])
            y = df['sentiment']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(C=c_val, max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_param("C", c_val)
            mlflow.log_param("max_features", max_feat)
            mlflow.log_param("ngram_range", "(1, 2)")

            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", accuracy)

            with open("vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)
            mlflow.log_artifact("vectorizer.pkl")

            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="model",
                registered_model_name="Flipkart_Sentiment_LR" 
            )
            
            print(f"Run Finished: F1={f1:.4f}")

print("All runs completed!")
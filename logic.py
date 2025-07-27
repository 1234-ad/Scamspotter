
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

def train_model(data_path='data/spam.csv'):
    df = pd.read_csv(data_path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['text']
    y = df['label']

    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])

    model.fit(X, y)
    joblib.dump(model, 'src/model.pkl')

def predict_message(text, model_path='src/model.pkl'):
    model = joblib.load(model_path)
    prediction = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    return ('Scam', prob[1]) if prediction == 1 else ('Safe', prob[0])

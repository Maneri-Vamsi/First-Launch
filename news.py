from flask import Flask, request, render_template_string
import pandas as pd
import nltk
import string
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

stop_words = set(stopwords.words('english')) - {'no', 'nor', 'not', 'against'}

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned)

if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake['label'] = 0
    true['label'] = 1

    min_len = min(len(fake), len(true))
    fake = fake.sample(min_len, random_state=42)
    true = true.sample(min_len, random_state=42)
    data = pd.concat([fake[['text', 'label']], true[['text', 'label']]])
    data = data.sample(frac=1, random_state=42)

    data['text'] = data['text'].apply(clean_text)
    X = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer(max_df=0.7, max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print("=== Classification Report ===")
    print(classification_report(y_test, model.predict(X_test)))

    model.fit(X_vec, y)
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

template = """
<!doctype html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body {
            background: linear-gradient(to right, #74ebd5, #9face6);
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
        }
        .container {
            background-color: white;
            width: 600px;
            margin: 60px auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        h2 {
            text-align: center;
            color: #4A00E0;
        }
        .note {
            text-align: center;
            font-size: 1em;
            color: #666;
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ccc;
            font-size: 1em;
        }
        button {
            background-color: #4A00E0;
            color: white;
            padding: 12px;
            width: 100%;
            margin-top: 15px;
            border: none;
            font-size: 1em;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #3800b0;
        }
        .result {
            font-weight: bold;
            text-align: center;
            margin-top: 25px;
            font-size: 1.2em;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            margin-top: 30px;
            color: #444;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>📰 Fake News Detector</h2>
        <div class="note">Thanks for choosing me 😊</div>
        <form method="POST">
            <textarea name="news" placeholder="Enter news text here..."></textarea>
            <button type="submit">Check News</button>
        </form>
        {% if prediction is not none %}
            <div class="result">
                Prediction: 
                <span style="color: {{ 'red' if 'Fake' in prediction else 'green' }}">{{ prediction }}</span>
            </div>
        {% endif %}
        <div class="footer">
            Dataset used: Fake & Real news articles (till 2017)
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["news"]
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0][pred]
        prediction = f"{'Real' if pred == 1 else 'Fake'} ({prob*100:.2f}%)"
    return render_template_string(template, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

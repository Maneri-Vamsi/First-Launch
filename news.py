from flask import Flask, request, render_template_string
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Download stopwords if not already downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Check if model/vectorizer exist, else train and save
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake['label'] = 0
    true['label'] = 1

    data = pd.concat([fake[['text', 'label']], true[['text', 'label']]], axis=0)
    X = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# HTML Template
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
                <span style="color: {{ 'red' if prediction == 'Fake' else 'green' }}">{{ prediction }}</span>
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
        vect_text = vectorizer.transform([text])
        pred = model.predict(vect_text)[0]
        prediction = "Real" if pred == 1 else "Fake"
    return render_template_string(template, prediction=prediction)

# Required for Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

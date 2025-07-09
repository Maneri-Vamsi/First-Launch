from flask import Flask, request, render_template_string
import joblib
import os
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# File paths
MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

# Preprocessing
def clean_text(text):
    text = re.sub(r"\W", " ", text)
    text = text.lower()
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Train the model if not already trained
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    # Load dataset
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["label"] = 0
    df_true["label"] = 1
    data = pd.concat([df_fake, df_true], axis=0)
    data = data[["text", "label"]]
    data["text"] = data["text"].apply(clean_text)

    # Train vectorizer and model
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"), max_df=0.7)
    X = vectorizer.fit_transform(data["text"])
    y = data["label"]

    model = PassiveAggressiveClassifier()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
else:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Fake News Detector</title></head>
<body>
    <h2>Fake News Detector</h2>
    <form method="post">
        <textarea name="news" rows="10" cols="70" placeholder="Enter news text here..." required></textarea><br><br>
        <input type="submit" value="Check">
    </form>
    {% if prediction is not none %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        prediction = "Real News 📰" if pred == 1 else "Fake News 🚫"
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

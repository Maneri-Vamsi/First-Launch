from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
from nltk.corpus import stopwords
import os

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Load and prepare data
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

data = pd.concat([fake_df, true_df]).sample(frac=1).reset_index(drop=True)
X = data["text"]
y = data["label"]

# Preprocessing
def clean_text(text):
    return " ".join([word for word in text.lower().split() if word not in STOPWORDS])

X_cleaned = X.apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X_cleaned)

# Model
model = LogisticRegression()
model.fit(X_vect, y)

# Flask App
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Fake News Detector</title></head>
<body>
    <h2>Enter News Text</h2>
    <form method="POST">
        <textarea name="news" rows="10" cols="60"></textarea><br><br>
        <input type="submit" value="Check">
    </form>
    {% if prediction is not none %}
        <h3>Prediction: {{ 'Real' if prediction == 1 else 'Fake' }}</h3>
    {% endif %}
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

# Important: Bind to the PORT environment variable for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

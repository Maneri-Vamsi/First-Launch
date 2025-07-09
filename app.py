import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import nltkfrom flask import Flask, request, render_template_string
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

from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gc

# ====================== INITIAL SETUP ======================
app = Flask(__name__)
MODEL_PATH = "model.pt"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
VOCAB_SIZE = 2000
MAX_TEXT_LENGTH = 10000

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ====================== MODEL DEFINITION ======================
class NewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# ====================== TEXT CLEANING ======================
def clean_text(text):
    return ' '.join([
        lemmatizer.lemmatize(word) for word in 
        re.sub(r'[^a-zA-Z\s]', '', text.lower()).split() 
        if word not in stop_words and len(word) > 2
    ])

# ====================== TRAINING FUNCTION ======================
def train_and_save_model():
    if not (os.path.exists("Fake.csv") and os.path.exists("True.csv")):
        print("❌ Training data not found.")
        return

    print("Training model...")
    df_fake = pd.read_csv("Fake.csv", encoding="utf-8", on_bad_lines="skip").assign(label=0)
    df_real = pd.read_csv("True.csv", encoding="utf-8", on_bad_lines="skip").assign(label=1)
    df = pd.concat([df_fake, df_real])
    df['text'] = (df['title'] + ' ' + df['text']).fillna('').apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

    vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, ngram_range=(1, 3), binary=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = NewsClassifier(VOCAB_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train_vec.toarray(), dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train.values, dtype=torch.long))
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("✅ Model and vectorizer saved.")

# ====================== LOAD OR TRAIN ======================
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    train_and_save_model()

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = NewsClassifier(VOCAB_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None

# ====================== HTML TEMPLATE ======================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake News Detector</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(to right, #f8f9fa, #e3f2fd);
            font-family: 'Segoe UI', sans-serif;
        }
        .container {
            margin-top: 5%;
            padding: 30px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
        textarea {
            resize: none;
        }
        .btn-custom {
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            font-size: 1.2rem;
            font-weight: 500;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 1rem;
        }
    </style>
</head>
<body>
    <div class="container col-md-8 col-lg-6">
        <h2 class="text-center mb-4">📰 Fake News Detector</h2>
        <form method="POST">
            <div class="form-group mb-3">
                <textarea name="article" class="form-control" rows="8" placeholder="Paste or type the news article here..."></textarea>
            </div>
            <div class="d-grid">
                <button type="submit" class="btn btn-custom">Check News</button>
            </div>
        </form>
        {% if result %}
            <div class="result-box mt-4 text-center" style="background-color: {{ result[2] }}; color: white;">
                {{ result[0] }} <br>(Confidence: {{ result[1] }}%)
            </div>
        {% endif %}
        <div class="text-muted text-center mt-3" style="font-size: 0.9rem;">
            <em>Note: This model was trained on news data up to 2017. Predictions may not reflect recent trends.</em>
        </div>
        <div class="footer">
            ☺️ Thanks for choosing me.
        </div>
    </div>
</body>
</html>
'''

# ====================== FLASK ROUTES ======================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        article = request.form.get("article", "").strip()
        if not model or not vectorizer:
            result = ("Model not available", 0, "#9E9E9E")
        elif len(article) < 20:
            result = ("Text too short", 0, "#FF9800")
        else:
            cleaned = clean_text(article[:MAX_TEXT_LENGTH])
            if not cleaned:
                result = ("No valid content to analyze", 0, "#FF9800")
            else:
                vec = vectorizer.transform([cleaned])
                with torch.no_grad():
                    probs = model(torch.tensor(vec.toarray(), dtype=torch.float32))
                    conf, pred = torch.max(probs, 1)
                    label = "Fake News" if pred.item() == 0 else "Real News"
                    color = "#FF5252" if pred.item() == 0 else "#4CAF50"
                    result = (label, round(conf.item() * 100, 2), color)
    return render_template_string(HTML_TEMPLATE, result=result)

@app.after_request
def cleanup(response):
    gc.collect()
    return response

# ====================== RUN APP ======================
if __name__ == "__main__":
    app.run(debug=True, port=5000)

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gc

# ======================
# 1. INITIAL SETUP
# ======================
app = Flask(__name__)

CONFIG = {
    'MODEL_PATH': os.path.join(os.path.dirname(__file__), "model.pt"),
    'VECTORIZER_PATH': os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl"),
    'DATA_FILES': [
        os.path.join(os.path.dirname(__file__), 'Fake.csv'),
        os.path.join(os.path.dirname(__file__), 'True.csv')
    ],
    'VOCAB_SIZE': 2000,
    'PORT': int(os.environ.get('PORT', 5000)),
    'MAX_TEXT_LENGTH': 10000
}

# ======================
# 2. NLP SETUP
# ======================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ======================
# 3. MODEL ARCHITECTURE
# ======================
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

# ======================
# 4. PREDICTION SYSTEM
# ======================
class Predictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def load_components(self):
        print("Loading model and vectorizer...")
        if all(os.path.exists(path) for path in [CONFIG['MODEL_PATH'], CONFIG['VECTORIZER_PATH']]):
            try:
                self.model = NewsClassifier(input_dim=CONFIG['VOCAB_SIZE'])
                self.model.load_state_dict(
                    torch.load(CONFIG['MODEL_PATH'], map_location=torch.device('cpu')))
                self.model.eval()
                self.vectorizer = joblib.load(CONFIG['VECTORIZER_PATH'])
                print("Model and vectorizer loaded successfully.")
                return
            except Exception as e:
                print(f"Error loading saved model/vectorizer: {e}")
        
        print("Training model from scratch...")
        self.model, self.vectorizer = self.train_model()

    def train_model(self):
        df = pd.concat([
            pd.read_csv(f, encoding='utf-8', on_bad_lines='skip').assign(label=i)
            for i, f in enumerate(CONFIG['DATA_FILES'])
        ])
        df['text'] = df['title'] + ' ' + df['text']
        df['text'] = df['text'].apply(lambda x: ' '.join(
            [lemmatizer.lemmatize(word) for word in 
             re.sub(r'[^a-zA-Z\s]', '', x.lower()).split() 
             if word not in stop_words and len(word) > 2]
        ))

        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(
            max_features=CONFIG['VOCAB_SIZE'],
            ngram_range=(1, 3),
            binary=True
        )
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        model = NewsClassifier(input_dim=CONFIG['VOCAB_SIZE'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_acc = 0
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            outputs = model(torch.tensor(X_train.toarray(), dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(y_train.values, dtype=torch.long))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                _, preds = torch.max(model(torch.tensor(X_test.toarray(), dtype=torch.float32)), 1)
                acc = (preds == torch.tensor(y_test.values, dtype=torch.long)).float().mean()
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), CONFIG['MODEL_PATH'])
                    joblib.dump(vectorizer, CONFIG['VECTORIZER_PATH'])

        return model, vectorizer

    def predict(self, text):
        if self.model is None or self.vectorizer is None:
            return ("Model is still loading, please try again in a few seconds.", 0, "#FF9800")

        if not isinstance(text, str) or len(text.strip()) < 20:
            return ("Invalid input", 0, "#FF9800")

        text = text[:CONFIG['MAX_TEXT_LENGTH']]
        cleaned = ' '.join([
            lemmatizer.lemmatize(word) for word in 
            re.sub(r'[^a-zA-Z\s]', '', text.lower()).split() 
            if word not in stop_words and len(word) > 2
        ])

        if not cleaned:
            return ("No features found", 0, "#FF9800")

        try:
            vec = self.vectorizer.transform([cleaned])
            with torch.no_grad():
                probs = self.model(torch.tensor(vec.toarray(), dtype=torch.float32))
                conf, pred = torch.max(probs, 1)
                color = "#FF5252" if pred.item() == 0 else "#4CAF50"
                return ("Fake News" if pred.item() == 0 else "Real News", 
                        round(conf.item() * 100, 2), 
                        color)
        except Exception as e:
            return (f"Error: {str(e)}", 0, "#FF9800")

# ======================
# 5. FLASK APPLICATION
# ======================
predictor = Predictor()

@app.before_first_request
def initialize():
    predictor.load_components()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        article = request.form.get("article", "").strip()
        if article:
            result = predictor.predict(article)
    return render_template_string("""<!DOCTYPE html>
    <html>
    <head>
        <title>News Authenticity Checker</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Poppins', sans-serif; background: #F5F5F6; color: #1C1B1F; margin: 0; padding: 20px; display: flex; justify-content: center; min-height: 100vh; }
            .container { width: 100%; max-width: 800px; background: white; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding: 32px; margin: 20px 0; }
            h1 { color: #6200EE; text-align: center; margin-bottom: 32px; font-weight: 600; }
            .description { text-align: center; margin-bottom: 24px; color: #333; opacity: 0.8; }
            textarea { width: 100%; padding: 16px; border: 2px solid #E0E0E0; border-radius: 8px; min-height: 200px; font-size: 16px; margin-bottom: 16px; }
            textarea:focus { outline: none; border-color: #6200EE; }
            button { background: #6200EE; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 8px; cursor: pointer; width: 100%; font-weight: 500; transition: all 0.3s; margin-bottom: 24px; }
            button:hover { background: #3700B3; transform: translateY(-2px); }
            .result { padding: 24px; border-radius: 8px; margin-top: 24px; text-align: center; color: white; font-weight: 500; background: {% if result %}{{ result[2] }}{% else %}#E0E0E0{% endif %}; transition: all 0.5s; }
            .confidence { height: 24px; background: rgba(255,255,255,0.3); border-radius: 12px; margin: 16px 0; overflow: hidden; }
            .confidence-bar { height: 100%; background: white; border-radius: 12px; width: {% if result %}{{ result[1] }}%{% else %}0%{% endif %}; }
            .footer { text-align: center; margin-top: 32px; font-size: 14px; color: #333; opacity: 0.6; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>News Authenticity Checker</h1>
            <p class="description">Paste a news article below to check its authenticity using AI</p>
            <form method="POST">
                <textarea name="article" placeholder="Paste news article here (minimum 20 characters)..." required></textarea>
                <button type="submit">Analyze Article</button>
            </form>
            {% if result %}
            <div class="result">
                <h3>{{ result[0] }}</h3>
                {% if result[1] > 0 %}
                <p>Confidence: {{ result[1] }}%</p>
                <div class="confidence">
                    <div class="confidence-bar"></div>
                </div>
                {% endif %}
            </div>
            {% endif %}
            <div class="footer">
                <p>This AI model analyzes news content to detect potential misinformation</p>
            </div>
        </div>
    </body>
    </html>""", result=result)

@app.after_request
def cleanup(response):
    gc.collect()
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=CONFIG['PORT'], threaded=True)

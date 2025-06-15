import os
import re
import zipfile
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

# ======================
# 1. INITIAL SETUP
# ======================
app = Flask(__name__)

# Configuration - MUST MATCH YOUR FILENAMES EXACTLY
CONFIG = {
    'MODEL_PATH': "model.pt",
    'VECTORIZER_PATH': "tfidf_vectorizer.pkl",
    'DATA_FILES': ['Fake.csv', 'True.csv'],  # Case-sensitive!
    'VOCAB_SIZE': 1000,  # Don't change this
    'PORT': 5000
}

# ======================
# 2. NLP SETUP
# ======================
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# ======================
# 3. DATA LOADING
# ======================
def load_data():
    """Load and combine datasets with validation"""
    dfs = []
    for filename, label in zip(CONFIG['DATA_FILES'], [0, 1]):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing dataset: {filename}")
        
        try:
            df = pd.read_csv(filename)
            # Validate required columns
            if 'text' not in df.columns or 'title' not in df.columns:
                raise ValueError(f"{filename} missing 'text' or 'title' column")
            
            df['combined_text'] = df['title'] + ' ' + df['text']
            df['label'] = label
            dfs.append(df[['combined_text', 'label']])
            
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {str(e)}")

    if not dfs:
        raise ValueError("No valid data loaded")
    
    return pd.concat(dfs, ignore_index=True)

# ======================
# 4. TEXT PROCESSING
# ======================
def clean_text(text):
    """Enhanced text cleaning"""
    if not isinstance(text, str):
        return ""
    
    # Normalization
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Advanced stopword removal
    tokens = [word for word in text.split() 
             if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

# ======================
# 5. MODEL ARCHITECTURE
# ======================
class NewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.net(x)

# ======================
# 6. MODEL TRAINING
# ======================
def train_model():
    """Complete training workflow"""
    print("🚀 Starting training process...")
    
    # 1. Load and prepare data
    df = load_data()
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # 3. Vectorization
    vectorizer = TfidfVectorizer(
        max_features=CONFIG['VOCAB_SIZE'],
        stop_words=list(stop_words),
        ngram_range=(1, 2),
        binary=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. Convert to tensors
    X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # 5. Model setup
    model = NewsClassifier(input_dim=CONFIG['VOCAB_SIZE'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Training loop
    best_accuracy = 0
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            model.eval()
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), CONFIG['MODEL_PATH'])
                joblib.dump(vectorizer, CONFIG['VECTORIZER_PATH'])
                
        print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f} | Test Acc: {accuracy:.2%}")
    
    print(f"✅ Training complete! Best accuracy: {best_accuracy:.2%}")
    return model, vectorizer

# ======================
# 7. PREDICTION SYSTEM
# ======================
class Predictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def load_components(self):
        """Load or train model components"""
        if os.path.exists(CONFIG['MODEL_PATH']) and os.path.exists(CONFIG['VECTORIZER_PATH']):
            try:
                self.model = NewsClassifier(input_dim=CONFIG['VOCAB_SIZE'])
                self.model.load_state_dict(torch.load(CONFIG['MODEL_PATH']))
                self.model.eval()
                self.vectorizer = joblib.load(CONFIG['VECTORIZER_PATH'])
                print("🔍 Loaded pre-trained model")
                return
            except Exception as e:
                print(f"⚠️ Error loading model: {e}. Retraining...")
                os.remove(CONFIG['MODEL_PATH'])
                os.remove(CONFIG['VECTORIZER_PATH'])
        
        self.model, self.vectorizer = train_model()
    
    def predict(self, text):
        """Make prediction on single article"""
        if not self.model or not self.vectorizer:
            self.load_components()
            
        cleaned = clean_text(text)
        if not cleaned:
            return ("Invalid input (empty after cleaning)", 0)
            
        vec = self.vectorizer.transform([cleaned])
        if vec.sum() == 0:
            return ("Invalid input (no features detected)", 0)
            
        with torch.no_grad():
            tensor = torch.tensor(vec.toarray(), dtype=torch.float32)
            probs = self.model(tensor)
            confidence, pred = torch.max(probs, dim=1)
            
        return ("Fake" if pred.item() == 0 else "Real", round(confidence.item() * 100, 2))

# ======================
# 8. FLASK APPLICATION
# ======================
predictor = Predictor()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        article = request.form.get("article", "").strip()
        if article:
            result, confidence = predictor.predict(article)
            return render_template_string(HTML_TEMPLATE, 
                                      prediction=result,
                                      confidence=confidence)
    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; padding: 10px; min-height: 200px; margin-bottom: 10px; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .fake { background: #ffebee; border-left: 5px solid #f44336; }
        .real { background: #e8f5e9; border-left: 5px solid #4CAF50; }
        .error { background: #fff3e0; border-left: 5px solid #ff9800; }
        .confidence { 
            height: 20px; 
            background: #e0e0e0; 
            border-radius: 10px; 
            margin: 10px 0;
            overflow: hidden;
        }
        .confidence-bar {
            height: 100%;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <h1>Fake News Detector</h1>
    <form method="POST">
        <textarea name="article" placeholder="Paste news article here..." required></textarea>
        <button type="submit">Analyze</button>
    </form>
    
    {% if prediction %}
    <div class="result {{ 'error' if 'Invalid' in prediction else prediction.lower() }}">
        <h3>{{ prediction }}</h3>
        {% if 'Invalid' not in prediction %}
        <p>Confidence: {{ confidence }}%</p>
        <div class="confidence">
            <div class="confidence-bar" style="width: {{ confidence }}%; 
                 background: {{ '#f44336' if prediction == 'Fake' else '#4CAF50' }}"></div>
        </div>
        {% endif %}
    </div>
    {% endif %}
</body>
</html>
"""

# ======================
# 9. APPLICATION START
# ======================
if __name__ == '__main__':
    # Initialize predictor
    predictor.load_components()
    
    # Start server
    print(f"🌐 Server running on http://localhost:{CONFIG['PORT']}")
    app.run(host='0.0.0.0', port=CONFIG['PORT'])

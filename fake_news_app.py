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
from sklearn.externals import joblib  # NEW
import nltk
from nltk.corpus import stopwords

# ------------------------------
# 1. Initial Setup with Error Handling
# ------------------------------
app = Flask(__name__)

# Configure paths
MODEL_PATH = "model.pt"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"  # NEW
DATA_FILES = ['Fake.csv', 'True.csv']

# ------------------------------
# 2. NLTK Setup with Cache Check
# ------------------------------
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# ------------------------------
# 3. Memory-Efficient Data Loading
# ------------------------------
def load_data():
    """Load and combine datasets with memory efficiency"""
    dfs = []
    for filename, label in zip(DATA_FILES, [0, 1]):
        if os.path.exists(filename):
            df = pd.read_csv(filename, usecols=['text'])
            df['label'] = label
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError("No data files found")
    
    return pd.concat(dfs, ignore_index=True)

def extract_if_needed():
    """Extract zip file if data files don't exist"""
    if not all(os.path.exists(f) for f in DATA_FILES):
        zip_path = 'Fake.csv.zip'
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            print("✅ Extracted data files")
        else:
            raise FileNotFoundError("No data files or zip archive found")

# ------------------------------
# 4. Text Cleaning Function
# ------------------------------
def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ------------------------------
# 5. Model Definition
# ------------------------------
class NewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        return self.softmax(self.fc2(out))

# ------------------------------
# 6. Initialize Vectorizer and Model
# ------------------------------
vectorizer = TfidfVectorizer(max_features=1000)
model = None

def initialize_model():
    """Initialize or load the trained model"""
    global model, vectorizer
    
    # Check if both model and vectorizer exist (NEW)
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("🚀 Loading pre-trained model and vectorizer...")
        model = NewsClassifier(input_dim=1000)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        vectorizer = joblib.load(VECTORIZER_PATH)  # NEW: Load fitted vectorizer
        return
    
    print("⏳ Training new model...")
    extract_if_needed()
    df = load_data()
    df['text'] = df['text'].apply(clean_text)
    
    X_train, _, y_train, _ = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # NEW: Explicitly fit the vectorizer before transform
    vectorizer.fit(X_train)  # This was missing!
    X_train_tfidf = vectorizer.transform(X_train).toarray()
    
    # NEW: Save the fitted vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    
    model = NewsClassifier(input_dim=1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3 | Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model and vectorizer saved")  # Updated message

# ------------------------------
# 7. Prediction Function (No changes needed here)
# ------------------------------
def predict_article(article):
    """Predict if article is fake or real"""
    if model is None:
        initialize_model()
    
    cleaned = clean_text(article)
    vec = vectorizer.transform([cleaned]).toarray()
    tensor_input = torch.tensor(vec, dtype=torch.float32)
    
    with torch.no_grad():
        probs = model(tensor_input)
        confidence, pred = torch.max(probs, dim=1)
    
    return ("Fake" if pred.item() == 0 else "Real", round(confidence.item() * 100, 2))

# ------------------------------
# 8. Flask Web App (No changes needed here)
# ------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>📰 Fake News Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; padding: 10px; }
        input[type="submit"] { margin-top: 10px; padding: 10px 20px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .fake { background-color: #ffdddd; border-left: 5px solid #f44336; }
        .real { background-color: #ddffdd; border-left: 5px solid #4CAF50; }
    </style>
</head>
<body>
    <h1>🧠 Fake News Classifier</h1>
    <form method="POST">
        <textarea name="article" rows="10" placeholder="Paste news article here..."></textarea><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <div class="result {{ prediction.lower() }}">
            <h2>Prediction: {{ prediction }}</h2>
            <p>Confidence: {{ confidence }}%</p>
        </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence = None, None
    if request.method == "POST":
        article = request.form.get("article", "").strip()
        if article:
            try:
                prediction, confidence = predict_article(article)
            except Exception as e:
                prediction = f"Error: {str(e)}"
                confidence = 0
    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence)

# ------------------------------
# 9. Run the App
# ------------------------------
if __name__ == '__main__':
    initialize_model()  # Ensures model and vectorizer are loaded
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Launching Fake News Detector on port {port}")
    app.run(host='0.0.0.0', port=port)

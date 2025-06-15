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
import nltk
from nltk.corpus import stopwords

# ------------------------------
# 1. Download NLTK Requirements
# ------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# ------------------------------
# 2. Extract the ZIP File
# ------------------------------
zip_path = 'Fake.csv.zip'
extract_folder = '.'
fake_csv_filename = "Fake.csv"
fake_csv_path = os.path.join(extract_folder, fake_csv_filename)

if not os.path.exists(fake_csv_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"✅ Extracted {zip_path}")
else:
    print(f"📄 {fake_csv_filename} already exists. Skipping extraction.")

# ------------------------------
# 3. Load and Prepare Dataset
# ------------------------------
df_fake = pd.read_csv(fake_csv_path)
df_fake['label'] = 0  # Fake = 0

true_csv_path = os.path.join(extract_folder, "True.csv")
if os.path.exists(true_csv_path):
    df_real = pd.read_csv(true_csv_path)
    df_real['label'] = 1  # Real = 1
    df = pd.concat([df_fake, df_real], ignore_index=True)
    print("✅ Combined Fake & True datasets.")
else:
    df = df_fake
    print("⚠️ True.csv not found. Using Fake.csv only.")

# ------------------------------
# 4. Clean the Text
# ------------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['text'] = df['text'].apply(clean_text)

# ------------------------------
# 5. Vectorization
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# ------------------------------
# 6. Define and Train Model
# ------------------------------
class NewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        return self.fc2(out)

model = NewsClassifier(input_dim=5000)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🏋️ Training model...")
for epoch in range(5):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/5 | Loss: {loss.item():.4f}")

# ------------------------------
# 7. Flask Web App
# ------------------------------
app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>📰 Fake News Detector</title>
</head>
<body>
    <h1>🧠 Fake News Classifier</h1>
    <form method="POST">
        <textarea name="article" rows="10" cols="60" placeholder="Paste news article here..."></textarea><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h2>Prediction: {{ prediction }}</h2>
        <p>Confidence: {{ confidence }}%</p>
    {% endif %}
</body>
</html>
"""

def predict_article(article):
    cleaned = clean_text(article)
    vec = vectorizer.transform([cleaned]).toarray()
    tensor_input = torch.tensor(vec, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return ("Fake" if pred.item() == 0 else "Real", round(confidence.item() * 100, 2))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence = None, None
    if request.method == "POST":
        article = request.form.get("article")
        if article:
            prediction, confidence = predict_article(article)
    return render_template_string(html_template, prediction=prediction, confidence=confidence)

# ------------------------------
# 8. Run the App
# ------------------------------
if __name__ == '__main__':
    print("🚀 Launching Fake News Detector on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
torch.save(model.state_dict(), "model.pt")


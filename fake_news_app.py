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
# 2. Extract the ZIP File for Fake.csv If Needed
# ------------------------------
zip_path = r'C:\Users\vamsi\OneDrive\Documents\Fake.csv.zip'
extract_folder = r'C:\Users\vamsi\OneDrive\Documents\ML_project'
fake_csv_filename = "Fake.csv"
fake_csv_path = os.path.join(extract_folder, fake_csv_filename)

if not os.path.exists(fake_csv_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Extracted {zip_path} to {extract_folder}")
else:
    print(f"{fake_csv_filename} already exists. Skipping extraction.")

# ------------------------------
# 3. Load and Preprocess the Dataset
# ------------------------------
df_fake = pd.read_csv(fake_csv_path)
df_fake['label'] = 0  # Fake news is labeled as 0

true_csv_filename = "True.csv"
true_csv_path = os.path.join(extract_folder, true_csv_filename)

# **File Existence Check**
if os.path.exists(true_csv_path):
    print("✅ True.csv found successfully!")
    df_real = pd.read_csv(true_csv_path)
    df_real['label'] = 1
    df = pd.concat([df_fake, df_real], ignore_index=True)
else:
    print("❌ True.csv NOT FOUND! Please check the file name and location.")
    df = df_fake

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['text'] = df['text'].apply(clean_text)
print("Data distribution by label:")
print(df['label'].value_counts())

# ------------------------------
# 4. Split Data and TF-IDF Vectorization
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
# 5. Define the Neural Network Model
# ------------------------------
class NewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

model = NewsClassifier(input_dim=5000)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 6. Train the Model
# ------------------------------
epochs = 5
print("Training model...")
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ------------------------------
# 7. Create a Flask Web Application
# ------------------------------
app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
</head>
<body>
    <h1>Fake News Detector</h1>
    <form method="POST">
        <textarea name="article" rows="10" cols="50" placeholder="Enter your news article here..."></textarea><br>
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
    
    prediction = "Fake" if pred.item() == 0 else "Real"
    return prediction, confidence.item() * 100

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    if request.method == "POST":
        article = request.form.get("article")
        prediction, confidence = predict_article(article)
    
    return render_template_string(html_template, prediction=prediction, confidence=confidence)

# ------------------------------
# 8. Run Flask with Port Management
# ------------------------------
from flask import Flask

app = Flask(__name__)

# your routes here...

if __name__ == '__main__':
    try:
        print("🚀 Starting Flask app...")
        app.run(debug=True, port=5000)
    except Exception as e:
        print("❌ Flask encountered an error:", e)

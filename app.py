from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Train model
data = pd.read_csv("dataset.csv")
data['text'] = data['text'].str.lower().str.replace(r'[^a-zA-Z ]', '', regex=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = LogisticRegression()
model.fit(X, y)

def predict(text):
    text = text.lower()
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# HTML Template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Text Classifier</title>
    <style>
        body {
            font-family: Arial;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            text-align: center;
            padding-top: 100px;
        }
        .box {
            background: white;
            color: black;
            padding: 30px;
            border-radius: 10px;
            display: inline-block;
        }
        input {
            width: 250px;
            padding: 10px;
            margin: 10px;
        }
        button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            cursor: pointer;
        }
        h2 {
            margin-bottom: 20px;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="box">
        <h2>Harassment Detector</h2>
        <form method="post">
            <input type="text" name="text" placeholder="Enter text" required />
            <br>
            <button type="submit">Predict</button>
        </form>
        {% if result %}
            <div class="result">Result: {{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        result = predict(text)
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
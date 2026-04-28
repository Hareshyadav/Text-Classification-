import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset.csv")

data['text'] = data['text'].str.lower().str.replace(r'[^a-zA-Z ]', '', regex=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

def predict(text):
    text = text.lower().strip()

    # vectorize
    vec = vectorizer.transform([text])

    # get probabilities for both classes
    probs = model.predict_proba(vec)[0]
    max_prob = probs.max()

    # threshold: if confidence is low → unknown
    if max_prob < 0.70:
        return "unknown"

    return model.classes_[probs.argmax()]

print(predict("you are useless"))
print(predict("great work"))
print(predict("you are useless"))   # harassment
print(predict("great work"))       # normal
print(predict("laptop"))           # unknown
print(predict("weather is good"))  # unknown
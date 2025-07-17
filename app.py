from flask import Flask, request, jsonify
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import re
from urllib.parse import urlparse

class SpamFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spam_keywords = [
            'free', 'win', 'winner', 'prize', 'congratulations', 'urgent',
            'offer', 'money', 'cash', 'click', 'buy', 'purchase', 'limited',
            'act now', 'call now', 'guaranteed', 'no cost', 'risk-free'
        ]

    def extract_urls(self, text):
        return re.findall(r'https?://\S+|www\.\S+', text)

    def url_features(self, urls):
        features = {
            'num_urls': len(urls),
            'num_suspicious_domains': 0
        }
        suspicious_domains = ['bit.ly', 'tinyurl.com', 'goo.gl', 'grabify.link', 'shorturl.at']

        for url in urls:
            domain = urlparse(url).netloc.lower()
            if any(susp in domain for susp in suspicious_domains):
                features['num_suspicious_domains'] += 1

        return features

    def keyword_features(self, text):
        lower_text = text.lower()
        return {
            'num_spam_keywords': sum(word in lower_text for word in self.spam_keywords)
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for email in X:
            urls = self.extract_urls(email)
            url_feats = self.url_features(urls)
            keyword_feats = self.keyword_features(email)
            combined = {**url_feats, **keyword_feats}
            features.append(list(combined.values()))
        return features

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("spam_classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    subject = data.get("subject", "")
    body = data.get("body", "")

    combined_text = subject + " " + body
    features = vectorizer.transform([combined_text])
    prediction = model.predict(features)[0]
    
    return jsonify({"verdict": "spam" if prediction == 1 else "ham"})

if __name__ == "__main__":
    app.run(debug=True)

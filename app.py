from flask import Flask, request, jsonify
import joblib

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
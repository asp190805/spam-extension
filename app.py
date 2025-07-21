import streamlit as st
import os
import pickle
import joblib
import base64
import re
import time

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("spam_classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

@st.cache_resource
def gmail_authenticate():
    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('gmail', 'v1', credentials=creds)
    return service

def get_email_snippets(service, max_results=10):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])

    email_data = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_data['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(No Subject)')
        snippet = msg_data.get('snippet', '')
        email_data.append({'subject': subject, 'snippet': snippet})
    return email_data

def preprocess(text):
    text = re.sub(r'\W+', ' ', text)  # Remove non-words
    return text.lower()

def classify_email(text):
    X = vectorizer.transform([preprocess(text)])
import pandas as pd

# Create a DataFrame with proper columns the model was trained on
email_input = pd.DataFrame([{
    "subject": subject,
    "body": body
}])

# Predict
prediction = model.predict(email_input)

    return "📬 HAM" if prediction == 0 else "🚨 SPAM"

# === Streamlit UI ===
st.title("📧 Gmail Spam Detector")
st.write("This app fetches your latest Gmail emails and classifies them using your trained spam model.")

if st.button("🔍 Scan My Inbox"):
    try:
        with st.spinner("Authenticating with Gmail..."):
            service = gmail_authenticate()

        with st.spinner("Fetching emails..."):
            emails = get_email_snippets(service, max_results=10)

        st.success("Fetched and classified successfully!")

        for i, email in enumerate(emails):
            st.markdown(f"### 📩 Email #{i+1}")
            st.write(f"**Subject:** {email['subject']}")
            st.write(f"**Snippet:** {email['snippet']}")
            result = classify_email(email['subject'] + " " + email['snippet'])
            st.write(f"**Prediction:** {result}")
            st.markdown("---")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

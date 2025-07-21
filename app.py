import streamlit as st
import os
import re
import joblib
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# === Load model (assumes pipeline handles vectorizer + preprocessing) ===
model = joblib.load("spam_classifier.joblib")

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
        email_data.append({'subject': subject, 'body': snippet})
    return email_data

# Classifier function
def classify_email(subject, snippet):
    email_input = pd.DataFrame([{
        "subject": subject,
        "body": snippet
    }])
    prediction = model.predict(email_input)[0]
    return "📬 HAM" if prediction == 0 else "🚨 SPAM"

# === Streamlit UI ===
st.set_page_config(page_title="Gmail Spam Detector", page_icon="📧")
st.title("📧 Gmail Spam Detector")
st.write("Detect spam or ham in your Gmail inbox using your trained ML model.")

if st.button("📥 Scan My Inbox"):
    try:
        with st.spinner("🔐 Authenticating with Gmail..."):
            service = gmail_authenticate()

        with st.spinner("📩 Fetching and classifying emails..."):
            emails = get_email_snippets(service, max_results=10)

        st.success("✅ Emails fetched and classified!")

        for i, email in enumerate(emails):
            st.markdown(f"### ✉️ Email #{i+1}")
            st.write(f"**Subject:** {email['subject']}")
            st.write(f"**Snippet:** {email['body']}")
            result = classify_email(email['subject'], email['body'])
            st.write(f"**Prediction:** {result}")
            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

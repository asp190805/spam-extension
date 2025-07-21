import streamlit as st
import os
import joblib
import re
import pandas as pd

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Load model pipeline (includes vectorizer)
model = joblib.load("spam_classifier.joblib")

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
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    return text.lower()

def classify_email(subject, body):
    # Preprocess both parts
    cleaned_subject = preprocess(subject)
    cleaned_body = preprocess(body)

    # Create input in format the model expects
    email_input = pd.DataFrame([{
        "subject": cleaned_subject,
        "body": cleaned_body
    }])

    # Predict
    prediction = model.predict(email_input)[0]
    return "📬 HAM" if prediction == 0 else "🚨 SPAM"

# === Streamlit UI ===
st.set_page_config(page_title="Gmail Spam Detector", page_icon="📧")
st.title("📧 Gmail Spam Detector")
st.write("This app fetches your latest Gmail emails and classifies them using your trained spam model.")

if st.button("🔍 Scan My Inbox"):
    try:
        with st.spinner("Authenticating with Gmail..."):
            service = gmail_authenticate()

        with st.spinner("Fetching emails..."):
            emails = get_email_snippets(service, max_results=10)

        st.success("✅ Emails fetched and classified!")

        for i, email in enumerate(emails):
            st.markdown(f"### 📩 Email #{i+1}")
            st.write(f"**Subject:** {email['subject']}")
            st.write(f"**Snippet:** {email['snippet']}")
            result = classify_email(email['subject'], email['snippet'])
            st.write(f"**Prediction:** {result}")
            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")


from __future__ import print_function
import os
import json
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    creds = None
    # Load token.json if it exists
    if os.path.exists('..\\..\\credentials\\token.json'):
        creds = Credentials.from_authorized_user_file('..\\..\\credentials\\token.json', SCOPES)
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '..\\..\\credentials\\credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save credentials for later use
        with open('..\\..\\credentials\\token.json', 'w') as token:
            token.write(creds.to_json())

    # Build Gmail API service
    service = build('gmail', 'v1', credentials=creds)

    # Today's date in YYYY/MM/DD for Gmail query
    today = datetime.date.today().strftime("%Y/%m/%d")
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    tomorrow = today + datetime.timedelta(days=1)

    # Gmail search query (emails from today only)
    query = f"after:{yesterday.strftime('%Y/%m/%d')} before:{tomorrow.strftime('%Y/%m/%d')}"

    # Search for today's emails
    results = service.users().messages().list(
        userId='me',
        q=query
    ).execute()

    messages = results.get('messages', [])
    emails_data = []

    if not messages:
        print("No messages found for today.")
    else:
        print(f"Found {len(messages)} emails today.\n")
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()

            headers = msg_data["payload"].get("headers", [])
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
            from_ = next((h["value"] for h in headers if h["name"] == "From"), "")
            date_ = next((h["value"] for h in headers if h["name"] == "Date"), "")
            snippet = msg_data.get("snippet", "")

            emails_data.append({
                "from": from_,
                "subject": subject,
                "date": date_,
                "snippet": snippet
            })
            print(f"From: {from_}\nSubject: {subject}\nDate: {date_}\n---")

    # Ensure data folder exists
    os.makedirs("..\\..\\data", exist_ok=True)

    # Save results to JSON
    output_file = f"..\\..\\data\\emails_{datetime.date.today().strftime('%Y_%m_%d')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(emails_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {len(emails_data)} emails to {output_file}")

if __name__ == '__main__':
    main()

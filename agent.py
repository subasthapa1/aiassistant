"""
gmail_agent.py
An Email Agent with LangChain + LangGraph + Gmail API
- Fetch unread emails
- Summarize them
- Check if reply needed
- Draft response
- Send reply if needed
"""

import os
from dotenv import load_dotenv
import base64
from typing import TypedDict, Optional
from email.mime.text import MIMEText

# --- Gmail API ---
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --- LangChain / LangGraph ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.graph import StateGraph, END



# --------------------------------------------------
# Gmail Setup
# --------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Load the .env file for credentials
load_dotenv('credentials\\.env')

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")


def get_gmail_service():
    creds = None
    if os.path.exists("credentials\\token.json"):
        creds = Credentials.from_authorized_user_file("credentials\\token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials\\credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("credentials\\token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# --------------------------------------------------
# Tools
# --------------------------------------------------
@tool("summarize_email", return_direct=False)
def summarize_email_tool(email: str) -> str:
    """Summarize an email in 3 concise bullet points."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You summarize emails clearly."),
        ("human", "Summarize this email in 3 concise bullet points:\n{email}")
    ])
    result = llm.invoke(prompt.format(email=email))
    return result.content.strip()


@tool("check_reply", return_direct=False)
def check_reply_tool(email: str) -> str:
    """Check if an email requires a reply. Returns 'YES' or 'NO'."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Decide if the email needs a reply."),
        ("human", "Email:\n{email}\nAnswer 'YES' or 'NO'.")
    ])
    result = llm.invoke(prompt.format(email=email))
    return result.content.strip()


@tool("generate_reply", return_direct=False)
def generate_reply_tool(email: str) -> str:
    """Generate a draft reply for the email if a reply is needed."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a polite assistant drafting professional replies."),
        ("human", "Draft a reply for this email:\n{email}")
    ])
    result = llm.invoke(prompt.format(email=email))
    return result.content.strip()


@tool("send_email", return_direct=False)
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email via Gmail API."""
    service = get_gmail_service()
    message = MIMEText(body)
    message["to"] = recipient
    message["subject"] = subject

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    send_message = {"raw": raw_message}
    service.users().messages().send(userId="me", body=send_message).execute()
    return f"Email sent to {recipient} with subject '{subject}'."


@tool("fetch_unread_emails", return_direct=False)
def fetch_unread_emails_tool(max_results: int = 5) -> list:
    """Fetch unread emails (snippet + sender + subject)."""
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", labelIds=["INBOX", "UNREAD"], maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    emails = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = msg_data["payload"]["headers"]
        subject = next(h["value"] for h in headers if h["name"] == "Subject")
        sender = next(h["value"] for h in headers if h["name"] == "From")
        snippet = msg_data.get("snippet", "")
        emails.append({"id": msg["id"], "subject": subject, "sender": sender, "snippet": snippet})
    return emails


# --------------------------------------------------
# LangGraph Workflow
# --------------------------------------------------
class EmailState(TypedDict):
    email: str
    recipient: str
    subject: str
    summary: Optional[str]
    needs_reply: Optional[bool]
    draft_reply: Optional[str]
    status: Optional[str]


def summarize_node(state: EmailState):
    summary = summarize_email_tool.invoke({"email": state["email"]})
    return {"summary": summary}


def check_reply_node(state: EmailState):
    result = check_reply_tool.invoke({"email": state["email"]})
    needs_reply = "YES" in result.upper()
    return {"needs_reply": needs_reply}


def generate_reply_node(state: EmailState):
    if not state["needs_reply"]:
        return {"draft_reply": None}
    result = generate_reply_tool.invoke({"email": state["email"]})
    return {"draft_reply": result}


def send_email_node(state: EmailState):
    if not state["draft_reply"]:
        return {"status": "No reply needed"}
    result = send_email_tool.invoke({
        "recipient": state["recipient"],
        "subject": f"Re: {state['subject']}",
        "body": state["draft_reply"]
    })
    return {"status": result}


workflow = StateGraph(EmailState)
workflow.add_node("summarize", summarize_node)
workflow.add_node("check_reply", check_reply_node)
workflow.add_node("generate_reply", generate_reply_node)
workflow.add_node("send_email", send_email_node)

workflow.set_entry_point("summarize")
workflow.add_edge("summarize", "check_reply")
workflow.add_edge("check_reply", "generate_reply")
workflow.add_edge("generate_reply", "send_email")
workflow.add_edge("send_email", END)

app = workflow.compile()

# --------------------------------------------------
# Example Run
# --------------------------------------------------
if __name__ == "__main__":
    email_text = """
    Hi John, 

    Just a reminder that the project update is due tomorrow. 
    Can you send me the latest version by end of day?

    Thanks,
    Sarah
    """
    result = app.invoke({
        "email": email_text,
        "recipient": "sarah@example.com",
        "subject": "Project update reminder"
    })

    print("📌 Summary:", result["summary"])
    print("📌 Draft Reply:", result["draft_reply"])
    print("📌 Status:", result["status"])

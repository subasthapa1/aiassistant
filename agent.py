"""
gmail_agent.py
An Email Agent with LangChain + LangGraph + Gmail API
- Fetch unread emails
- Summarize them
- Check if reply needed
- Draft response
"""

import os
from typing import Annotated, Literal, Sequence
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool

# Gmail API imports
import base64
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText

# Load .env file for OpenAI key
load_dotenv("credentials/.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# --- Gmail API setup ---
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def get_gmail_service():
    creds = None
    if os.path.exists("credentials\\token.json"):
        creds = Credentials.from_authorized_user_file("credentials\\token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials\\credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("credentials\\token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

gmail_service = get_gmail_service()

# --- LangGraph Agent ---

class MessageClassifier(BaseModel):
    message_type: Literal["reply", "summarise"] = Field(
        ..., description="Summarize email received and propose a draft response"
    )

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_all_emails():
    """Fetch the latest emails from Gmail inbox"""
    results = gmail_service.users().messages().list(userId="me", maxResults=10, labelIds=["INBOX"]).execute()
    messages = results.get("messages", [])
    email_data = []
    for msg in messages:
        msg_obj = gmail_service.users().messages().get(userId="me", id=msg["id"]).execute()
        snippet = msg_obj.get("snippet", "")
        email_data.append(snippet)
    return email_data

@tool
def identify_unanswered_email():
    """Identify unread emails in Gmail inbox"""
    results = gmail_service.users().messages().list(userId="me", labelIds=["INBOX", "UNREAD"], maxResults=5).execute()
    messages = results.get("messages", [])
    if not messages:
        return "No unread emails."
    unread_data = []
    for msg in messages:
        msg_obj = gmail_service.users().messages().get(userId="me", id=msg["id"]).execute()
        snippet = msg_obj.get("snippet", "")
        unread_data.append(snippet)
    return unread_data

@tool
def propose_draft_response(to: str, subject: str, body: str):
    """Send a draft response via Gmail"""
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"Draft response sent to {to}"

tools = [identify_unanswered_email, get_all_emails]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI email assistant. Help me read, summarize, and reply to Gmail messages.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        try:
            message.pretty_print()
        except AttributeError:
            print(message)

# Example: fetch unread emails
inputs = {"messages": [HumanMessage(content="Check my unread emails and give me an email which needs reply")]}
print_stream(app.stream(inputs, stream_mode="values"))

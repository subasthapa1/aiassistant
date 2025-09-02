"""
Streamlit Web Interface for email agent

This provides an interactive web interface to test the email assistant agent
with different execution methods.
"""

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import Agent
from agent import get_gmail_service,summarize_email_tool,check_reply_tool, generate_reply_tool, send_email_tool, fetch_unread_emails_tool

# Load environment variables
load_dotenv('..\\credentials\\.env')

# Page configuration
st.set_page_config(
    page_title="Personal Email Assistant Agent",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Personal Email Assistant Agent")
st.markdown("Email Assistant Agent")
st.write(
    "This app reads my Gmail, summarizes emails, checks if a reply is needed, drafts a response, and can send it.")

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please create a .env file with your OpenAI API key.")
    st.stop()

# Sidebar for settings
st.sidebar.header("Settings")
max_results = st.sidebar.slider("Number of emails to fetch", 1, 10, 5)

# Fetch emails button
if st.sidebar.button("Fetch Unread Emails"):
    with st.spinner("Fetching emails..."):
        emails = fetch_unread_emails_tool.invoke({"max_results": max_results})
    if not emails:
        st.info("✅ No unread emails found.")
    else:
        st.session_state["emails"] = emails

# Display fetched emails
if "emails" in st.session_state:
    st.subheader("Unread Emails")
    for idx, email in enumerate(st.session_state["emails"], start=1):
        with st.expander(f"📩 {idx}. {email['subject']} (from {email['sender']})"):
            st.write("**Snippet:**", email["snippet"])
            if st.button(f"Process this email", key=f"process_{idx}"):
                with st.spinner("Processing with AI..."):
                    result = email_app.invoke({
                        "email": email["snippet"],
                        "recipient": email["sender"],
                        "subject": email["subject"]
                    })
                st.session_state["result"] = result
                st.session_state["selected_email"] = email

# Show AI Results
if "result" in st.session_state:
    result = st.session_state["result"]
    email = st.session_state["selected_email"]

    st.subheader("AI Analysis & Draft Reply")
    st.write("**Summary:**", result["summary"])
    st.write("**Needs Reply?**", "✅ Yes" if result["needs_reply"] else "❌ No")

    if result["draft_reply"]:
        st.text_area("Draft Reply", value=result["draft_reply"], height=200, key="draft_area")

        if st.button("Send Reply"):
            with st.spinner("Sending email..."):
                send_result = email_app.invoke({
                    "email": email["snippet"],
                    "recipient": email["sender"],
                    "subject": email["subject"],
                    "draft_reply": st.session_state["draft_area"],
                    "needs_reply": True
                })
            st.success(send_result["status"])
    else:
        st.info("This email doesn’t need a reply.")

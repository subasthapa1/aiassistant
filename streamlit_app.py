"""
Streamlit Web Interface for Email Agent
----------------------------------------
This app connects to Gmail, fetches unread emails, summarizes them,
checks if a reply is needed, drafts a response, and can send it.
"""
# streamlit_app.py
import streamlit as st
from langchain_core.messages import HumanMessage
from agent import app  # Import the compiled agent graph

# --- Streamlit UI ---
st.set_page_config(page_title="📧 Gmail AI Assistant", layout="centered")

st.title("📧 Gmail AI Assistant")
st.write("Ask me to check unread emails, summarize them, or draft replies.")

# Input from user
user_query = st.text_area("Enter your request:", "Check my unread emails")

if st.button("Run Agent"):
    if user_query.strip():
        # Create input for the LangGraph agent
        inputs = {"messages": [HumanMessage(content=user_query)]}

        # Stream the response
        response_placeholder = st.empty()
        collected_response = ""

        for step in app.stream(inputs, stream_mode="values"):
            message = step["messages"][-1]
            text = getattr(message, "content", str(message))
            collected_response += text + "\n"
            response_placeholder.markdown(collected_response)

        st.success("Done ✅")
    else:
        st.warning("Please enter a request.")

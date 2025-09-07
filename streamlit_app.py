# streamlit_app.py
import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot")

# Sidebar for uploading documents
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT files", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    files = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in uploaded_files]
    with st.spinner("Uploading and processing documents..."):
        response = requests.post(f"{API_URL}/add-documents", files=files)
    if response.status_code == 200:
        st.sidebar.success("✅ Documents added successfully!")
    else:
        st.sidebar.error(f"❌ Failed: {response.json().get('detail')}")

# Chat interface
st.header("💬 Chat with your documents")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")
if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        payload = {"question": user_input, "k": 4}
        response = requests.post(f"{API_URL}/query", json=payload)

    if response.status_code == 200:
        answer = response.json().get("answer", "⚠️ No answer returned")
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))
    else:
        st.error(f"❌ Error: {response.json().get('detail')}")

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 {sender}:** {msg}")
    else:
        st.markdown(f"**🤖 {sender}:** {msg}")

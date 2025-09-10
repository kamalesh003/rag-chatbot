# streamlit_app_standalone.py
import streamlit as st
import uuid
from datetime import datetime
from pathlib import Path
import os
import logging
from typing import List, Dict, Any
import tempfile

# Import the local modules directly
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "current_session" not in st.session_state:
    st.session_state.current_session = []
if "show_chat_history" not in st.session_state:
    st.session_state.show_chat_history = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

# Initialize components
def init_components():
    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if "vectorstore_loaded" not in st.session_state:
        st.session_state.vectorstore_loaded = False

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Document management
st.sidebar.subheader("📂 Document Management")

# Create data directory if it doesn't exist
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT files", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

def save_uploaded_files(uploaded_files):
    """Save uploaded files to data directory"""
    saved_files = []
    for uploaded_file in uploaded_files:
        try:
            file_path = data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
            st.sidebar.success(f"Saved: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to save {uploaded_file.name}: {str(e)}")
    return saved_files

def process_uploaded_files(file_paths):
    """Process uploaded files and update vectorstore"""
    try:
        processor = st.session_state.processor
        
        # Check if vectorstore exists
        existing_store = processor.load_existing_vectorstore()
        if existing_store:
            # Add to existing store
            chunks_added = processor.add_documents_to_existing_store(file_paths)
            st.session_state.vectorstore_loaded = True
            return {"message": "Documents added to existing vectorstore", "chunks_added": chunks_added}
        else:
            # Create new vectorstore
            vectorstore, chunk_count = processor.process_documents()
            st.session_state.vectorstore_loaded = True
            return {"message": "New vectorstore created", "chunks_processed": chunk_count}
            
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

if st.sidebar.button("📤 Upload & Process Documents") and uploaded_files:
    with st.sidebar:
        with st.spinner("Saving and processing documents..."):
            try:
                # Save uploaded files
                saved_files = save_uploaded_files(uploaded_files)
                if saved_files:
                    # Process the files
                    result = process_uploaded_files(saved_files)
                    st.sidebar.success("✅ Documents processed successfully!")
                    st.sidebar.json(result)
                else:
                    st.sidebar.error("❌ No files were saved successfully")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to process documents: {str(e)}")

def list_documents():
    """List all documents in data directory"""
    try:
        processor = st.session_state.processor
        files = processor.get_supported_files()
        return {"documents": [file.name for file in files], "count": len(files)}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return {"documents": [], "count": 0, "error": str(e)}

if st.sidebar.button("📋 List Documents"):
    with st.sidebar:
        result = list_documents()
        if result.get("error"):
            st.sidebar.error(f"❌ Failed: {result['error']}")
        else:
            st.sidebar.success("Available documents:")
            st.sidebar.json(result)

def load_vectorstore():
    """Load the vectorstore"""
    try:
        processor = st.session_state.processor
        rag_pipeline = st.session_state.rag_pipeline
        
        # Try to load vectorstore
        if processor.load_existing_vectorstore():
            # Also load it in the RAG pipeline
            if rag_pipeline.load_vectorstore():
                st.session_state.vectorstore_loaded = True
                return {"message": "Vectorstore loaded successfully", "exists": True}
            else:
                return {"message": "Vectorstore exists but failed to load in RAG pipeline", "exists": True}
        else:
            return {"message": "No vectorstore found", "exists": False}
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        return {"message": f"Failed to load vectorstore: {str(e)}", "exists": False}

if st.sidebar.button("🔍 Load Vectorstore"):
    with st.sidebar:
        with st.spinner("Loading vectorstore..."):
            result = load_vectorstore()
            if result.get("exists"):
                st.sidebar.success("✅ Vectorstore loaded!")
                st.session_state.vectorstore_loaded = True
            else:
                st.sidebar.warning("⚠️ " + result.get("message", "Vectorstore not available"))

# Chat session management
st.sidebar.subheader("💬 Chat Sessions")

# Create new session
if st.sidebar.button("🆕 New Chat Session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.current_session = []
    st.session_state.user_input = ""
    st.sidebar.success(f"New session started: {st.session_state.session_id[:8]}...")

# Initialize chat history for current session
def init_session_history(session_id):
    """Initialize chat history for a session"""
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "messages": []
        }

def update_session_activity(session_id):
    """Update last activity timestamp for session"""
    if session_id in st.session_state.chat_history:
        st.session_state.chat_history[session_id]["last_activity"] = datetime.now()

def add_message_to_session(session_id, message_type, content, sources=None):
    """Add a message to session history"""
    init_session_history(session_id)
    
    message = {
        "type": message_type,
        "content": content,
        "timestamp": datetime.now(),
        "sources": sources or []
    }
    
    st.session_state.chat_history[session_id]["messages"].append(message)
    update_session_activity(session_id)

def get_session_history(session_id):
    """Get chat history for a session"""
    init_session_history(session_id)
    return st.session_state.chat_history[session_id]

def delete_session(session_id):
    """Delete a chat session"""
    if session_id in st.session_state.chat_history:
        del st.session_state.chat_history[session_id]
    if st.session_state.session_id == session_id:
        st.session_state.session_id = None
        st.session_state.current_session = []
        st.session_state.user_input = ""
        st.session_state.show_chat_history = False

# Health check
if st.sidebar.button("🌐 Check System Status"):
    with st.sidebar:
        # Check vectorstore
        vectorstore_status = load_vectorstore()
        
        # Check data directory
        data_files = list_documents()
        
        status_msg = []
        if vectorstore_status.get("exists"):
            status_msg.append("✅ Vectorstore available")
        else:
            status_msg.append("❌ Vectorstore not available")
        
        if data_files.get("count", 0) > 0:
            status_msg.append(f"✅ {data_files['count']} documents found")
        else:
            status_msg.append("❌ No documents found")
        
        if st.session_state.session_id:
            status_msg.append(f"✅ Active session: {st.session_state.session_id[:8]}...")
        
        st.sidebar.info("\n".join(status_msg))

# Main chat interface
st.header("💬 Chat with your documents")

# Session info
if st.session_state.session_id:
    st.info(f"✅ Current Session: {st.session_state.session_id[:8]}...")
    if not st.session_state.vectorstore_loaded:
        st.warning("⚠️ Vectorstore not loaded. Please load vectorstore first.")
else:
    st.warning("No active session. Click 'New Chat Session' to start.")

# Show chat history toggle
if st.session_state.session_id and st.button("📜 Show Chat History"):
    st.session_state.show_chat_history = not st.session_state.show_chat_history

if st.session_state.show_chat_history and st.session_state.session_id:
    history = get_session_history(st.session_state.session_id)
    st.subheader("📜 Session Chat History")
    
    messages = history.get("messages", [])
    if messages:
        for message in messages:
            if message["type"] == "user":
                st.markdown(f"**🧑 You:** {message['content']}")
                st.caption(f"*{message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
            else:
                st.markdown(f"**🤖 Bot:** {message['content']}")
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"• {source.get('source_file', 'Unknown')} (Page {source.get('page', 'N/A')})")
                st.caption(f"*{message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
            st.divider()
    else:
        st.info("No messages in this session yet.")

# Chat input
user_input = st.text_input("Ask a question:", value=st.session_state.user_input, key="user_input_widget")
k_slider = st.slider("Number of relevant chunks (k):", min_value=1, max_value=10, value=4)

if st.button("🚀 Send") and user_input.strip():
    if not st.session_state.session_id:
        # Create new session if none exists
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.current_session = []
        init_session_history(st.session_state.session_id)
    
    if not st.session_state.vectorstore_loaded:
        # Try to load vectorstore if not loaded
        load_result = load_vectorstore()
        if not load_result.get("exists"):
            st.error("❌ No vectorstore available. Please upload and process documents first.")
            st.stop()
    
    with st.spinner("Thinking..."):
        try:
            # Execute query using RAG pipeline
            rag_pipeline = st.session_state.rag_pipeline
            result = rag_pipeline.query(user_input, k=k_slider)
            
            # Add to session history
            add_message_to_session(
                st.session_state.session_id, 
                "user", 
                user_input
            )
            
            add_message_to_session(
                st.session_state.session_id,
                "bot",
                result["answer"],
                result.get("sources", [])
            )
            
            # Add to current session display
            st.session_state.current_session.append(("You", user_input))
            st.session_state.current_session.append(("Bot", result["answer"]))
            
            # Clear the input for next question
            st.session_state.user_input = ""
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Query error: {str(e)}", exc_info=True)

# Display current conversation if any
if st.session_state.current_session:
    st.subheader("💭 Current Conversation")
    for i, (sender, msg) in enumerate(st.session_state.current_session):
        if sender == "You":
            st.markdown(f"**🧑 {sender}:** {msg}")
        else:
            st.markdown(f"**🤖 {sender}:** {msg}")
            
            # Show sources for bot messages
            if st.session_state.session_id:
                history = get_session_history(st.session_state.session_id)
                messages = history.get("messages", [])
                if messages and len(messages) > i//2:
                    bot_message = messages[i//2]
                    if bot_message.get("type") == "bot" and bot_message.get("sources"):
                        with st.expander("📚 Sources"):
                            for source in bot_message["sources"]:
                                st.write(f"• {source.get('source_file', 'Unknown')} (Page {source.get('page', 'N/A')})")

# Session management buttons
if st.session_state.session_id:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Clear Current Chat"):
            st.session_state.current_session = []
            st.session_state.user_input = ""
            st.rerun()

    with col2:
        if st.button("🗑️ Delete Session"):
            delete_session(st.session_state.session_id)
            st.success("Session deleted successfully")
            st.rerun()

# Footer
st.divider()
st.caption("RAG Chatbot - Upload documents and chat with them using AI-powered search")

# Initialize components on first run
init_components()
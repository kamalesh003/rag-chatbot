https://huggingface.co/spaces/higher5fh/rag-chatbot

## HOW TO USE?

1. **Open Streamlit App** → Launch the app in the browser as a fresh standalone session.
2. **Create New Session** → Initialize a new workspace for the user (clear old context).
3. **Upload Documents** → User uploads PDFs, text, or other files → embeddings are generated & stored in vector DB.
4. **Load / Connect Vector Store** → Set up or connect to an existing vector database (FAISS).
5. **Ask Questions** → User enters natural language queries, system retrieves relevant chunks from vector DB.
6. **Get Answers** → Display context-aware responses with optional sources/metadata.

## NOTE:
7. **Current converstation = short-term view (2–3 pairs)**
8. **Chat history = full session log.**

## WORKING OF RAG:

<img width="1915" height="825" alt="Screenshot 2025-09-05 055120" src="https://github.com/user-attachments/assets/b261fb9c-9940-4e95-9d0d-4d4507f0a49f" />

## FLOW DIAGRAM:
<img width="3840" height="2931" alt="Untitled diagram _ Mermaid Chart-2025-09-10-020132" src="https://github.com/user-attachments/assets/9f9af7b0-55bf-4fb3-95b5-8dee1ae08f41" />



## DocumentProcessor (document_processor.py)

**Purpose**: Handles document loading, processing, and vector store management.

### Key Functions:

**Initialization:**
- `__init__()` - Sets up data directory, vector store path, embeddings, and supported file types

**File Operations:**
- `get_supported_files()` - Returns list of supported files (.pdf, .txt) in data directory
- `load_document(file_path)` - Loads a single document (PDF or TXT) and adds source metadata

**Document Processing:**
- `split_documents(documents)` - Splits documents into chunks with configurable size/overlap
- `create_vectorstore(chunks)` - Creates FAISS vector store from document chunks
- `save_vectorstore(vectorstore)` - Saves vector store to disk

**Vector Store Management:**
- `load_existing_vectorstore()` - Loads existing vector store from disk
- `process_documents()` - Full pipeline: loads all files, splits, creates and saves vector store
- `add_documents_to_existing_store(file_paths)` - Adds new documents to existing vector store

## RAGPipeline (rag_pipeline.py)

**Purpose**: Handles querying and retrieval from the vector store using RAG pattern.

### Key Functions:

**Initialization:**
- `__init__()` - Sets up vector store path, embeddings, and LLM (GPT-3.5-turbo)

**Vector Store Operations:**
- `load_vectorstore()` - Loads the FAISS vector store from disk

**Retrieval & Querying:**
- `retrieve_context(query, k=4)` - Finds most relevant documents for a query
- `format_context(documents)` - Formats retrieved documents into readable context string
- `generate_answer(query, context)` - Uses LLM to generate answer based on context

**Main Pipeline:**
- `query(question, k=4)` - Complete RAG workflow: retrieval → context formatting → answer generation



## How the Streamlit App Utilizes Both Modules

### 1. **DocumentProcessor Integration**

**File Upload & Processing:**
```python
# Uses DocumentProcessor to handle document operations
processor = st.session_state.processor = DocumentProcessor()

# Get supported files from data directory
files = processor.get_supported_files()

# Load individual documents
documents = processor.load_document(file_path)

# Split documents into chunks
chunks = processor.split_documents(documents)

# Create vector store
vectorstore = processor.create_vectorstore(chunks)

# Save vector store
processor.save_vectorstore(vectorstore)

# Load existing vector store
existing_store = processor.load_existing_vectorstore()

# Add new documents to existing store
chunks_added = processor.add_documents_to_existing_store(file_paths)
```

**Key Features:**
- Handles PDF and TXT file uploads
- Processes documents into chunks
- Manages vector store creation and updates
- Provides document listing functionality

### 2. **RAGPipeline Integration**

**Query Processing:**
```python
# Uses RAGPipeline for retrieval and generation
rag_pipeline = st.session_state.rag_pipeline = RAGPipeline()

# Load vector store into RAG pipeline
rag_pipeline.load_vectorstore()

# Execute queries
result = rag_pipeline.query(user_input, k=k_slider)
```

**Key Features:**
- Handles similarity search and retrieval
- Formats context from retrieved documents
- Generates answers using OpenAI LLM
- Manages the complete RAG workflow

## Complete Workflow in the App

### 1. **Document Management (Using DocumentProcessor)**
- Users upload PDF/TXT files through the sidebar
- Files are saved to `./data` directory
- `DocumentProcessor` processes them into vector embeddings
- Vector store is created/updated in `./vectorstore/faiss_index/`

### 2. **Vector Store Management**
- **Loading**: `processor.load_existing_vectorstore()` + `rag_pipeline.load_vectorstore()`
- **Health Checks**: System status checks using both modules
- **Error Handling**: Graceful handling of missing vector stores

### 3. **Chat Interface (Using RAGPipeline)**
- User inputs questions through text input
- `RAGPipeline.query()` handles:
  - Retrieval of relevant document chunks
  - Context formatting
  - Answer generation using GPT-3.5-turbo
- Results include answer + source citations

### 4. **Session Management**
- Chat history persistence across sessions
- Source tracking and display
- Session creation/deletion











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


# Document Processing and RAG Pipeline System Documentation



## Module 1: DocumentProcessor

The `DocumentProcessor` class handles document loading, splitting, and vector store creation/management.

### Key Functions

#### 1. `__init__(data_dir: str, vectorstore_path: str)`
- **Purpose**: Initialize the document processor with data directory and vector store path
- **Parameters**:
  - `data_dir`: Path to directory containing documents (default: "./data")
  - `vectorstore_path`: Path to store/load FAISS vector store (default: "./vectorstore/faiss_index")
- **Optimization**: Uses OpenAI embeddings with configurable model via environment variable

#### 2. `get_supported_files() → List[Path]`
- **Purpose**: Retrieve all supported files from the data directory
- **Supported Formats**: PDF (.pdf) and Text (.txt) files
- **Error Handling**: Raises FileNotFoundError if data directory doesn't exist

#### 3. `load_document(file_path: Path) → List[Document]`
- **Purpose**: Load a document based on its file type
- **Format Handling**:
  - PDF files: Uses PyPDFLoader
  - Text files: Uses TextLoader with UTF-8 encoding
- **Metadata Enhancement**: Adds source_file metadata to track document origin
- **Error Handling**: Comprehensive exception handling with detailed logging

#### 4. `split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) → List[Document]`
- **Purpose**: Split documents into manageable chunks for processing
- **Default Parameters**: 
  - `chunk_size`: 1000 characters
  - `chunk_overlap`: 200 characters
- **Optimization**: Uses recursive character splitting with intelligent separators

#### 5. `create_vectorstore(chunks: List[Document]) → FAISS`
- **Purpose**: Create a FAISS vector store from document chunks
- **Embeddings**: Uses configured OpenAI embeddings

#### 6. `save_vectorstore(vectorstore: FAISS)`
- **Purpose**: Save vector store to disk
- **Directory Management**: Automatically creates parent directories if needed

#### 7. `load_existing_vectorstore() → Optional[FAISS]`
- **Purpose**: Load an existing vector store from disk
- **Security Note**: Uses `allow_dangerous_deserialization` flag (requires trust in source)

#### 8. `process_documents() → Tuple[FAISS, int]`
- **Purpose**: Complete pipeline to process all documents in data directory
- **Workflow**: 
  1. Get all supported files
  2. Load each document
  3. Split into chunks
  4. Create vector store
  5. Save to disk
- **Returns**: Vector store instance and total chunk count

#### 9. `add_documents_to_existing_store(file_paths: List[Path]) → int`
- **Purpose**: Add new documents to an existing vector store
- **Workflow**:
  1. Load existing vector store
  2. Process new documents
  3. Add chunks to existing store
  4. Save updated store
- **Returns**: Number of new chunks added

## Module 2: RAGPipeline

The `RAGPipeline` class handles querying the vector store and generating answers using LLMs.

### Key Functions

#### 1. `__init__(vectorstore_path: str)`
- **Purpose**: Initialize the RAG pipeline with vector store path
- **Components**:
  - OpenAI embeddings (text-embedding-3-small)
  - ChatOpenAI LLM (gpt-3.5-turbo, temperature=0)
  - Vector store reference

#### 2. `load_vectorstore() → bool`
- **Purpose**: Load the FAISS vector store from disk
- **Error Handling**: Returns boolean status with detailed logging

#### 3. `retrieve_context(query: str, k: int) → List[Document]`
- **Purpose**: Retrieve relevant document chunks for a query
- **Parameters**:
  - `query`: User question
  - `k`: Number of chunks to retrieve (default: 4)
- **Method**: FAISS similarity search

#### 4. `format_context(documents: List[Document]) → str`
- **Purpose**: Format retrieved documents into a readable context string
- **Format**: Includes source file and page information for citation

#### 5. `generate_answer(query: str, context: str) → str`
- **Purpose**: Generate answer using LLM with retrieved context
- **Prompt Engineering**: Includes instructions for:
  - Context-based answering
  - Handling insufficient information
  - Conciseness and accuracy
  - Source citation

#### 6. `query(question: str, k: int) → dict`
- **Purpose**: Complete RAG pipeline execution
- **Workflow**:
  1. Load vector store (if not loaded)
  2. Retrieve relevant context
  3. Format context
  4. Generate answer
  5. Extract source information
- **Returns**: Comprehensive response dictionary with:
  - Answer text
  - Source documents with metadata
  - Context string
  - Retrieved document count
  - Error information (if any)






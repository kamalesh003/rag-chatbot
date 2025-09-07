from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import logging
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline
from pydantic import BaseModel
from typing import List, Optional


# Logging Setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Pydantic Models

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4


# FastAPI App

app = FastAPI(
    title="RAG Chatbot API",
    description="Upload documents, build vectorstore, and query them using RAG pipeline",
    version="1.0.0"
)

# Initialize processor and RAG pipeline
processor = DocumentProcessor()
rag_pipeline = RAGPipeline()


# Endpoints


@app.post("/add-documents")
async def add_documents(files: List[UploadFile] = File(...)):
    """
    Add new documents to the vectorstore (PDF or TXT).
    Creates a new vectorstore if none exists.
    """
    try:
        os.makedirs("./data", exist_ok=True)
        file_paths = []

        for file in files:
            ext = file.filename.split('.')[-1].lower()
            if ext not in ["pdf", "txt"]:
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue

            file_path = Path(f"./data/{file.filename}")
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)

        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid PDF/TXT files provided")

        # Load existing store if available
        existing_store = processor.load_existing_vectorstore()
        if existing_store:
            chunks_added = processor.add_documents_to_existing_store(file_paths)
            return {"message": "Documents added to existing vectorstore", "chunks_added": chunks_added}
        else:
            vectorstore, chunk_count = processor.process_documents()
            return {"message": "New vectorstore created", "chunks_processed": chunk_count}

    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

@app.get("/list-documents")
async def list_documents():
    """
    List all PDF/TXT documents in the data directory.
    """
    try:
        files = processor.get_supported_files()
        return {"documents": [file.name for file in files], "count": len(files)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/load-vectorstore")
async def load_vectorstore():
    """
    Load the FAISS vectorstore if available.
    """
    try:
        if processor.load_existing_vectorstore():
            return {"message": "Vectorstore loaded successfully", "exists": True}
        else:
            raise HTTPException(status_code=404, detail="No vectorstore found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vectorstore: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Ask a question against the document knowledge base.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        result = rag_pipeline.query(request.question, k=request.k)
        return result
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG Chatbot API", "version": "1.0.0"}

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "add_documents": "/add-documents",
            "load_vectorstore": "/load-vectorstore",
            "list_documents": "/list-documents",
            "query": "/query"
        }
    }





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import logging
from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="API for document processing and vector storage",
    version="1.0.0"
)

# Initialize processor
processor = DocumentProcessor()



@app.post("/add-documents")
async def add_documents(files: list[UploadFile] = File(...)):
    """
    Add new documents to the vectorstore. ccreates vectorstore if it doesn't exist.
    """
    try:
        # Ensure data directory exists
        os.makedirs("./data", exist_ok=True)
        file_paths = []
        
        # Save uploaded files
        for file in files:
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in ['pdf', 'txt']:
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue
            
            file_path = Path(f"./data/{file.filename}")
            
            # Read and save file content
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            file_paths.append(file_path)
            logger.info(f"Saved file: {file.filename}")
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid PDF/TXT files provided")
        
        # Check if vectorstore exists
        existing_vectorstore = processor.load_existing_vectorstore()
        
        if existing_vectorstore:
            # Add to existing vectorstore
            chunks_added = processor.add_documents_to_existing_store(file_paths)
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Documents added to existing vectorstore",
                    "chunks_added": chunks_added,
                    "vectorstore_created": False
                }
            )
        else:
            # Create new vectorstore with uploaded files
            # First, remove any existing files in data directory to avoid duplicates
            for existing_file in processor.get_supported_files():
                if existing_file not in file_paths:
                    os.remove(existing_file)
            
            # Process the uploaded files
            vectorstore, chunk_count = processor.process_documents()
            return JSONResponse(
                status_code=201,
                content={
                    "message": "New vectorstore created with uploaded documents",
                    "chunks_processed": chunk_count,
                    "vectorstore_created": True
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

@app.get("/load-vectorstore")
async def load_vectorstore():
    """
    Check if vectorstore exists and can be loaded.
    """
    try:
        vectorstore = processor.load_existing_vectorstore()
        if not vectorstore:
            raise HTTPException(status_code=404, detail="No vectorstore found")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Vectorstore loaded successfully",
                "exists": True
            }
        )
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load vectorstore: {str(e)}")

@app.get("/list-documents")
async def list_documents():
    """
    List all supported documents in the data directory.
    """
    try:
        files = processor.get_supported_files()
        return JSONResponse(
            status_code=200,
            content={
                "documents": [str(file.name) for file in files],
                "count": len(files)
            }
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/clear-vectorstore")
async def clear_vectorstore():
    """
    Delete the existing vectorstore.
    """
    try:
        vectorstore_path = Path("./vectorstore/faiss_index")
        if vectorstore_path.exists():
            import shutil
            shutil.rmtree(vectorstore_path)
            return JSONResponse(
                status_code=200,
                content={"message": "Vectorstore deleted successfully"}
            )
        else:
            raise HTTPException(status_code=404, detail="Vectorstore not found")
    except Exception as e:
        logger.error(f"Error clearing vectorstore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear vectorstore: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "RAG Chatbot API",
            "version": "1.0.0"
        }
    )

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return JSONResponse(
        status_code=200,
        content={
            "message": "RAG Chatbot API",
            "version": "1.0.0",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "process_documents": "/process-documents",
                "add_documents": "/add-documents",
                "load_vectorstore": "/load-vectorstore",
                "list_documents": "/list-documents",
                "clear_vectorstore": "/clear-vectorstore"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
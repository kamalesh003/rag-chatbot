# document_processor.py (improved)
import os
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, data_dir: str = "./data", vectorstore_path: str = "./vectorstore/faiss_index"):
        self.data_dir = Path(data_dir)
        self.vectorstore_path = Path(vectorstore_path)
        self.embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
        self.supported_extensions = ['.pdf', '.txt']
    
    def get_supported_files(self) -> List[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {self.data_dir}")
        
        files = []
        for ext in self.supported_extensions:
            files.extend(list(self.data_dir.glob(f"*{ext}")))
        return files
    
    def load_document(self, file_path: Path) -> List[Document]:
        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = file_path.name
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)
    
    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        return FAISS.from_documents(chunks, self.embeddings)
    
    def save_vectorstore(self, vectorstore: FAISS):
        os.makedirs(self.vectorstore_path.parent, exist_ok=True)
        vectorstore.save_local(str(self.vectorstore_path))
    
    def load_existing_vectorstore(self) -> Optional[FAISS]:
        index_file = self.vectorstore_path / "index.faiss"
        if index_file.exists():
            # Warning: Only use allow_dangerous_deserialization if you trust the source
            return FAISS.load_local(
                str(self.vectorstore_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return None
    
    def process_documents(self) -> Tuple[FAISS, int]:
        files = self.get_supported_files()
        if not files:
            raise ValueError("No supported files found in data directory")
        
        all_chunks = []
        for file_path in files:
            logger.info(f"Processing {file_path.name}")
            documents = self.load_document(file_path)
            chunks = self.split_documents(documents)
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        
        vectorstore = self.create_vectorstore(all_chunks)
        self.save_vectorstore(vectorstore)
        logger.info(f"Created vectorstore with {len(all_chunks)} chunks")
        return vectorstore, len(all_chunks)
    
    def add_documents_to_existing_store(self, file_paths: List[Path]) -> int:
        vectorstore = self.load_existing_vectorstore()
        if not vectorstore:
            raise ValueError("No existing vector store found")
        
        new_chunks = []
        for file_path in file_paths:
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.warning(f"Skipping unsupported file: {file_path}")
                continue
            try:
                documents = self.load_document(file_path)
                chunks = self.split_documents(documents)
                new_chunks.extend(chunks)
                logger.info(f"Added {len(chunks)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
        
        if new_chunks:
            vectorstore.add_documents(new_chunks)
            self.save_vectorstore(vectorstore)
            logger.info(f"Added {len(new_chunks)} new chunks to vectorstore")
        
        return len(new_chunks)
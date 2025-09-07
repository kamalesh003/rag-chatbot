import os
from pathlib import Path
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import logging




load_dotenv()

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vectorstore_path: str = "./vectorstore/faiss_index/"):
        self.vectorstore_path = vectorstore_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vectorstore = None
        
    def load_vectorstore(self) -> bool:
        """Load the existing vectorstore"""
        try:
            index_file = Path(self.vectorstore_path) / "index.faiss"
            if not index_file.exists():
                logger.warning("Vectorstore index file not found")
                return False
            
            self.vectorstore = FAISS.load_local(
                str(self.vectorstore_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vectorstore loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}")
            return False
    
    def retrieve_context(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant context from vectorstore"""
        if not self.vectorstore:
            if not self.load_vectorstore():
                raise ValueError("Vectorstore not available")
        
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[Document {i+1} - {source} (Page {page})]:\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context"""
        try:
            # Define the prompt template
            prompt_template = ChatPromptTemplate.from_template(
                """You are a helpful assistant that answers questions based on the provided context.
                
                Context:
                {context}
                
                Question: {question}
                
                Instructions:
                1. Answer the question based only on the context provided
                2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided documents."
                3. Be concise and accurate
                4. Cite the source documents when relevant
                
                Answer:"""
            )
            
            # Create the RAG chain
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = rag_chain.invoke({"context": context, "question": query})
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def query(self, question: str, k: int = 4) -> dict:
        """Complete RAG pipeline: retrieve context and generate answer"""
        try:
            # Check if vectorstore is available
            if not self.vectorstore and not self.load_vectorstore():
                return {
                    "answer": "No vectorstore available. Please add documents first.",
                    "sources": [],
                    "error": "Vectorstore not found"
                }
            
            # Retrieve relevant context
            retrieved_docs = self.retrieve_context(question, k=k)
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "context": ""
                }
            
            # Format context
            context_str = self.format_context(retrieved_docs)
            
            # Generate answer
            answer = self.generate_answer(question, context_str)
            
            # Extract source information
            sources = [
                {
                    "source_file": doc.metadata.get('source_file', 'Unknown'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in retrieved_docs
            ]
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context_str,
                "retrieved_docs_count": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import uuid

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Document processing
import PyPDF2
import docx
from io import BytesIO
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    user_role: Optional[str] = "user"
    max_results: Optional[int] = 3

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]]
    query_id: str
    timestamp: str

class FeedbackRequest(BaseModel):
    query_id: str
    helpful: bool
    comments: Optional[str] = ""

class DocumentChunk(BaseModel):
    content: str
    source: str
    chunk_id: str

# Global variables for the RAG system
embedding_model = None
vector_index = None
document_chunks = []
gemini_model = None

class RAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_index = None
        self.document_chunks = []
        self.gemini_model = None
        self.feedback_log = []
        
    def initialize_models(self):
        """Initialize embedding model and Gemini client"""
        try:
            # Initialize SentenceTransformer for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
            
            # Initialize Gemini client (requires GEMINI_API_KEY environment variable)
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCuK9KFbeMxI5nzr8D8RvNpSW7cunHamig")
            if api_key and api_key != "your-actual-api-key-here":
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini client initialized successfully")
                
                # Test the API key with a simple call
                try:
                    test_response = self.gemini_model.generate_content("Test")
                    logger.info("Gemini API key verified successfully")
                except Exception as e:
                    logger.error(f"Gemini API key verification failed: {e}")
                    self.gemini_model = None
                    
            else:
                logger.warning("GEMINI_API_KEY not found or is placeholder. Using fallback response generation.")
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filename}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_content: bytes, filename: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {filename}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_content: bytes, filename: str) -> str:
        """Extract text from TXT file"""
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting text from TXT {filename}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, file_content: bytes, filename: str) -> List[DocumentChunk]:
        """Process a document and return chunks"""
        # Extract text based on file extension
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            text = self.extract_text_from_pdf(file_content, filename)
        elif file_ext == 'docx':
            text = self.extract_text_from_docx(file_content, filename)
        elif file_ext == 'txt':
            text = self.extract_text_from_txt(file_content, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if not text.strip():
            raise ValueError(f"No text extracted from {filename}")
        
        # Split into chunks
        chunks = self.chunk_text(text)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_obj = DocumentChunk(
                content=chunk,
                source=filename,
                chunk_id=f"{filename}_{i}"
            )
            document_chunks.append(chunk_obj)
        
        return document_chunks
    
    def add_documents_to_index(self, document_chunks: List[DocumentChunk]):
        """Add document chunks to the vector index"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in document_chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Initialize or update FAISS index
        if self.vector_index is None:
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.vector_index.add(embeddings.astype('float32'))
        
        # Store document chunks
        self.document_chunks.extend(document_chunks)
        
        logger.info(f"Added {len(document_chunks)} chunks to index. Total chunks: {len(self.document_chunks)}")
    
    def _rebuild_vector_index(self):
        """Rebuild the vector index with current document chunks"""
        if not self.document_chunks:
            self.vector_index = None
            return
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in self.document_chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Create new FAISS index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.vector_index.add(embeddings.astype('float32'))
        
        logger.info(f"Rebuilt vector index with {len(self.document_chunks)} chunks")
    
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k relevant document chunks"""
        if not self.embedding_model or not self.vector_index:
            raise ValueError("Models not initialized or no documents indexed")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = self.vector_index.search(query_embedding.astype('float32'), k)
        
        # Return chunks with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_chunks):
                results.append((self.document_chunks[idx], float(score)))
        
        return results
    
    def generate_response(self, query: str, retrieved_chunks: List[tuple], user_role: str = "user") -> str:
        """Generate response using retrieved chunks and LLM"""
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"Source: {chunk.source}\nContent: {chunk.content}"
            for chunk, score in retrieved_chunks
        ])
        
        # Role-based filtering (basic implementation)
        role_context = ""
        if user_role.lower() == "manager":
            role_context = "Please provide a high-level executive summary. "
        elif user_role.lower() == "developer":
            role_context = "Please focus on technical details and implementations. "
        
        if self.gemini_model:
            try:
                # Use Gemini API
                prompt = f"""
{role_context}Based on the following context, please answer the user's question. 
If the answer cannot be found in the context, please say so clearly.

Context:
{context}

Question: {query}

Please provide a helpful answer and cite the sources where relevant information was found.
"""
                
                response = self.gemini_model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                # Fallback to simple response
                return self._generate_fallback_response(query, retrieved_chunks, user_role)
        else:
            return self._generate_fallback_response(query, retrieved_chunks, user_role)
    
    def _generate_fallback_response(self, query: str, retrieved_chunks: List[tuple], user_role: str) -> str:
        """Generate a simple fallback response when Gemini is not available"""
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your question."
        
        response_parts = [f"Based on the available documents, here's what I found regarding '{query}':\n"]
        
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            response_parts.append(f"{i}. From {chunk.source}:")
            response_parts.append(f"   {chunk.content[:200]}...")
            response_parts.append("")
        
        response_parts.append("Please note: This is a basic response. For more sophisticated answers, configure the Gemini API key.")
        
        return "\n".join(response_parts)
    
    def log_feedback(self, query_id: str, helpful: bool, comments: str = ""):
        """Log user feedback"""
        feedback = {
            "query_id": query_id,
            "helpful": helpful,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        self.feedback_log.append(feedback)
        logger.info(f"Feedback logged for query {query_id}: {'Helpful' if helpful else 'Not helpful'}")

# Initialize RAG system
rag_system = RAGSystem()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    rag_system.initialize_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Chatbot API is running",
        "total_chunks": len(rag_system.document_chunks),
        "status": "healthy"
    }

@app.post("/upload-documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    try:
        results = []
        
        for file in files:
            # Read file content
            content = await file.read()
            
            # Process document
            chunks = rag_system.process_document(content, file.filename)
            
            # Add to vector index
            rag_system.add_documents_to_index(chunks)
            
            results.append({
                "filename": file.filename,
                "chunks_created": len(chunks),
                "status": "success"
            })
        
        return {
            "message": f"Successfully processed {len(files)} documents",
            "results": results,
            "total_chunks": len(rag_system.document_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection"""
    try:
        if len(rag_system.document_chunks) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No documents have been uploaded yet. Please upload documents first."
            )
        
        # Retrieve relevant chunks
        retrieved_chunks = rag_system.retrieve_relevant_chunks(
            request.query, 
            k=request.max_results
        )
        
        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found for your query."
            )
        
        # Generate response
        response_text = rag_system.generate_response(
            request.query, 
            retrieved_chunks, 
            request.user_role
        )
        
        # Prepare sources
        sources = [
            {
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "relevance_score": f"{score:.3f}"
            }
            for chunk, score in retrieved_chunks
        ]
        
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        return QueryResponse(
            response=response_text,
            sources=sources,
            query_id=query_id,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a query response"""
    try:
        rag_system.log_feedback(
            request.query_id,
            request.helpful,
            request.comments
        )
        
        return {
            "message": "Feedback submitted successfully",
            "query_id": request.query_id
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/")
async def get_stats():
    """Get system statistics"""
    return {
        "total_documents": len(set(chunk.source for chunk in rag_system.document_chunks)),
        "total_chunks": len(rag_system.document_chunks),
        "total_feedback": len(rag_system.feedback_log),
        "positive_feedback": sum(1 for f in rag_system.feedback_log if f["helpful"]),
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "FAISS",
        "llm_available": rag_system.gemini_model is not None
    }

@app.get("/documents/")
async def list_documents():
    """List all uploaded documents"""
    documents = {}
    for chunk in rag_system.document_chunks:
        if chunk.source not in documents:
            documents[chunk.source] = 0
        documents[chunk.source] += 1
    
    return {
        "documents": [
            {"filename": filename, "chunks": count}
            for filename, count in documents.items()
        ]
    }

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a specific document and its chunks"""
    try:
        # Check if document exists
        document_exists = any(chunk.source == filename for chunk in rag_system.document_chunks)
        if not document_exists:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        
        # Get chunks to remove
        chunks_to_remove = [i for i, chunk in enumerate(rag_system.document_chunks) if chunk.source == filename]
        
        if not chunks_to_remove:
            raise HTTPException(status_code=404, detail=f"No chunks found for document '{filename}'")
        
        # Remove chunks from document list
        rag_system.document_chunks = [
            chunk for chunk in rag_system.document_chunks 
            if chunk.source != filename
        ]
        
        # Rebuild vector index without the deleted document
        if rag_system.document_chunks:
            rag_system._rebuild_vector_index()
        else:
            # If no documents left, reset the index
            rag_system.vector_index = None
        
        logger.info(f"Deleted document '{filename}' with {len(chunks_to_remove)} chunks")
        
        return {
            "message": f"Successfully deleted document '{filename}'",
            "chunks_removed": len(chunks_to_remove),
            "remaining_documents": len(set(chunk.source for chunk in rag_system.document_chunks)),
            "remaining_chunks": len(rag_system.document_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/")
async def delete_all_documents():
    """Delete all documents and reset the system"""
    try:
        total_documents = len(set(chunk.source for chunk in rag_system.document_chunks))
        total_chunks = len(rag_system.document_chunks)
        
        # Clear all documents and reset index
        rag_system.document_chunks = []
        rag_system.vector_index = None
        
        logger.info(f"Deleted all documents: {total_documents} documents, {total_chunks} chunks")
        
        return {
            "message": "Successfully deleted all documents",
            "documents_removed": total_documents,
            "chunks_removed": total_chunks
        }
        
    except Exception as e:
        logger.error(f"Error deleting all documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Set Gemini API key from environment variable or .env file
    # The key is now loaded automatically by load_dotenv() at the top
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# End-to-End RAG System Implementation
# Complete modular system with all required components

import os
import json
import uuid
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
import io

# Core dependencies
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import streamlit as st
from tqdm import tqdm
import requests

# Configure Tesseract OCR
import platform
if platform.system() == "Windows":
    # Set Tesseract path for Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class RAGConfig:
    """Central configuration for the RAG system"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "./chroma_db"
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.3
    max_context_length: int = 2048
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2:7b"
    
config = RAGConfig()

# Data Models
@dataclass
class Document:
    """Document metadata structure"""
    id: str
    filename: str
    file_path: str
    file_hash: str
    total_pages: int
    processing_method: str  # 'native' or 'ocr'
    created_at: str
    metadata: Dict[str, Any]

@dataclass
class TextChunk:
    """Text chunk with metadata"""
    id: str
    document_id: str
    content: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Retrieval result with relevance scoring"""
    chunk: TextChunk
    similarity_score: float
    rank: int

@dataclass
class RAGResponse:
    """Complete RAG response structure"""
    answer: str
    citations: List[Dict[str, Any]]
    retrieval_results: List[RetrievalResult]
    processing_time: float
    metadata: Dict[str, Any]

# PDF Processing Pipeline
class PDFProcessor:
    """Main PDF processing orchestrator"""
    
    def __init__(self):
        self.native_extractor = NativeTextExtractor()
        self.ocr_extractor = OCRTextExtractor()
        
    def process_pdf(self, file_path: str) -> Document:
        """Process PDF and return Document object"""
        logger.info(f"Processing PDF: {file_path}")
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(file_path)
        
        # Try native extraction first
        try:
            pages_data = self.native_extractor.extract_text(file_path)
            processing_method = "native"
        except Exception as e:
            logger.warning(f"Native extraction failed: {e}. Trying OCR...")
            pages_data = self.ocr_extractor.extract_text(file_path)
            processing_method = "ocr"
        
        # Create document object
        document = Document(
            id=str(uuid.uuid4()),
            filename=Path(file_path).name,
            file_path=file_path,
            file_hash=file_hash,
            total_pages=len(pages_data),
            processing_method=processing_method,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata={"pages_data": pages_data}
        )
        
        logger.info(f"Successfully processed {document.filename} with {document.total_pages} pages")
        return document
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

class NativeTextExtractor:
    """Extract text from native PDFs using PyMuPDF"""
    
    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF pages"""
        pages_data = []
        
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                
                # Skip if page is mostly empty (likely scanned)
                if len(text.strip()) < 50:
                    raise ValueError(f"Page {page_num} contains insufficient text")
                
                pages_data.append({
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text),
                    "extraction_method": "native"
                })
        
        return pages_data

class OCRTextExtractor:
    """Extract text from scanned PDFs using OCR"""
    
    def __init__(self):
        # Verify tesseract is available
        self._verify_tesseract()
    
    def _verify_tesseract(self):
        """Verify that Tesseract is properly installed and accessible"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.error(f"Tesseract OCR not available: {e}")
            raise RuntimeError("Tesseract OCR is not properly installed or configured")
    
    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text using OCR"""
        pages_data = []
        
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc, 1):
                    try:
                        # Convert page to image
                        mat = fitz.Matrix(2.0, 2.0)  # Higher resolution
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("ppm")
                        
                        # Convert to PIL Image
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Perform OCR with error handling
                        text = pytesseract.image_to_string(img, config='--psm 6')
                        
                        pages_data.append({
                            "page_number": page_num,
                            "text": text,
                            "char_count": len(text),
                            "extraction_method": "ocr"
                        })
                        
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num}: {e}")
                        # Add empty page data to maintain page numbering
                        pages_data.append({
                            "page_number": page_num,
                            "text": "",
                            "char_count": 0,
                            "extraction_method": "ocr_failed"
                        })
                        
        except Exception as e:
            logger.error(f"Failed to process PDF for OCR: {e}")
            raise
        
        return pages_data

# Text Chunking and Processing
class TextChunker:
    """Advanced text chunking with overlap and sentence awareness"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, min_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        """Create chunks from document"""
        chunks = []
        chunk_index = 0
        
        for page_data in document.metadata["pages_data"]:
            page_num = page_data["page_number"]
            text = page_data["text"]
            
            page_chunks = self._create_page_chunks(
                text, document.id, page_num, chunk_index
            )
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {document.filename}")
        return chunks
    
    def _create_page_chunks(self, text: str, doc_id: str, page_num: int, start_index: int) -> List[TextChunk]:
        """Create chunks from page text"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = start_index
        
        for sentence in sentences:
            # Check if adding sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and len(current_chunk) > self.min_size:
                # Create chunk
                chunk = TextChunk(
                    id=f"{doc_id}_chunk_{chunk_index}",
                    document_id=doc_id,
                    content=current_chunk.strip(),
                    page_number=page_num,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={"sentence_count": len(current_chunk.split('.'))}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else ""
                current_chunk = overlap_text + " " + sentence
                current_start += len(current_chunk) - len(overlap_text)
                chunk_index += 1
            else:
                current_chunk += " " + sentence
        
        # Handle remaining text
        if len(current_chunk.strip()) >= self.min_size:
            chunk = TextChunk(
                id=f"{doc_id}_chunk_{chunk_index}",
                document_id=doc_id,
                content=current_chunk.strip(),
                page_number=page_num,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={"sentence_count": len(current_chunk.split('.'))}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

# Embedding and Vector Index
class EmbeddingEngine:
    """Generate and manage embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for text list"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for single query"""
        return self.model.encode([query], normalize_embeddings=True)[0]

class VectorStore:
    """ChromaDB vector storage and retrieval"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._get_or_create_collection()
        logger.info(f"Initialized ChromaDB at {persist_directory}")
    
    def _get_or_create_collection(self):
        """Get or create the main collection"""
        try:
            return self.client.get_collection("rag_documents")
        except:
            return self.client.create_collection("rag_documents")
    
    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to vector store"""
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        where_clause = {"document_id": document_filter} if document_filter else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }

# Retrieval Engine
class RetrievalEngine:
    """Advanced retrieval with reranking and filtering"""
    
    def __init__(self, vector_store: VectorStore, embedding_engine: EmbeddingEngine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
    
    def retrieve(self, query: str, top_k: int = 5, 
                document_filter: Optional[str] = None,
                similarity_threshold: float = 0.3) -> List[RetrievalResult]:
        """Retrieve relevant chunks for query"""
        logger.info(f"Retrieving chunks for query: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_engine.generate_query_embedding(query)
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding, top_k=top_k * 2, document_filter=document_filter
        )
        
        # Process results
        retrieval_results = []
        if search_results['documents']:
            for i, (doc, metadata, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold:
                    chunk = TextChunk(
                        id=search_results['ids'][0][i],
                        document_id=metadata['document_id'],
                        content=doc,
                        page_number=metadata['page_number'],
                        chunk_index=metadata['chunk_index'],
                        start_char=metadata['start_char'],
                        end_char=metadata['end_char'],
                        metadata=metadata
                    )
                    
                    result = RetrievalResult(
                        chunk=chunk,
                        similarity_score=similarity_score,
                        rank=i + 1
                    )
                    retrieval_results.append(result)
        
        # Sort by similarity score and limit to top_k
        retrieval_results.sort(key=lambda x: x.similarity_score, reverse=True)
        retrieval_results = retrieval_results[:top_k]
        
        logger.info(f"Retrieved {len(retrieval_results)} relevant chunks")
        return retrieval_results

# LLM Integration (Ollama)
class LLMEngine:
    """Local LLM integration using Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2:7b"):
        self.base_url = base_url
        self.model = model
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama connection and model availability"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json()['models']]
                if self.model not in models:
                    logger.warning(f"Model {self.model} not found. Available: {models}")
                else:
                    logger.info(f"Connected to Ollama with model {self.model}")
        except requests.exceptions.RequestException:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using local LLM"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return "Error: Unable to generate response"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return "Error: LLM service unavailable"

# Main RAG System
class RAGSystem:
    """Complete RAG system orchestrator"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            min_size=self.config.min_chunk_size
        )
        self.embedding_engine = EmbeddingEngine(self.config.embedding_model)
        self.vector_store = VectorStore(self.config.vector_db_path)
        self.retrieval_engine = RetrievalEngine(self.vector_store, self.embedding_engine)
        self.llm_engine = LLMEngine(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model
        )
        
        # Storage for documents
        self.documents_store = {}
        self._load_documents_store()
        
        logger.info("RAG System initialized successfully")
    
    def _load_documents_store(self):
        """Load documents store from file"""
        store_path = Path(self.config.vector_db_path) / "documents.json"
        if store_path.exists():
            with open(store_path, 'r') as f:
                data = json.load(f)
                self.documents_store = {
                    doc_id: Document(**doc_data) 
                    for doc_id, doc_data in data.items()
                }
    
    def _save_documents_store(self):
        """Save documents store to file"""
        store_path = Path(self.config.vector_db_path) / "documents.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(store_path, 'w') as f:
            json.dump(
                {doc_id: asdict(doc) for doc_id, doc in self.documents_store.items()},
                f, indent=2
            )
    
    def ingest_document(self, file_path: str) -> str:
        """Ingest a PDF document into the RAG system"""
        start_time = time.time()
        
        try:
            # Process PDF
            document = self.pdf_processor.process_pdf(file_path)
            
            # Check for duplicates
            existing_doc = self._find_duplicate_document(document.file_hash)
            if existing_doc:
                logger.info(f"Document already exists: {existing_doc.filename}")
                return existing_doc.id
            
            # Create chunks
            chunks = self.text_chunker.chunk_document(document)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_engine.generate_embeddings(chunk_texts)
            
            # Store in vector database
            self.vector_store.add_chunks(chunks, embeddings)
            
            # Store document metadata
            self.documents_store[document.id] = document
            self._save_documents_store()
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully ingested {document.filename} in {processing_time:.2f}s")
            
            return document.id
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            raise
    
    def _find_duplicate_document(self, file_hash: str) -> Optional[Document]:
        """Find document by hash to avoid duplicates"""
        for doc in self.documents_store.values():
            if doc.file_hash == file_hash:
                return doc
        return None
    
    def query(self, question: str, document_filter: Optional[str] = None) -> RAGResponse:
        """Main query method with full RAG pipeline"""
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieval_results = self.retrieval_engine.retrieve(
            question, 
            top_k=self.config.top_k_retrieval,
            document_filter=document_filter,
            similarity_threshold=self.config.similarity_threshold
        )
        
        # Prepare context for LLM
        context = self._prepare_context(retrieval_results)
        
        # Generate prompt
        prompt = self._create_prompt(question, context)
        
        # Generate answer
        answer = self.llm_engine.generate_response(prompt)
        
        # Create citations
        citations = self._create_citations(retrieval_results)
        
        processing_time = time.time() - start_time
        
        response = RAGResponse(
            answer=answer,
            citations=citations,
            retrieval_results=retrieval_results,
            processing_time=processing_time,
            metadata={
                "question": question,
                "context_length": len(context),
                "num_sources": len(retrieval_results)
            }
        )
        
        return response
    
    def _prepare_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        for i, result in enumerate(retrieval_results):
            chunk = result.chunk
            doc = self.documents_store.get(chunk.document_id)
            if doc:
                context_parts.append(
                    f"[Source {i+1}] From {doc.filename}, Page {chunk.page_number}:\n"
                    f"{chunk.content}\n"
                )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for LLM"""
        return f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {question}

Answer: Please provide a comprehensive answer based on the context above. Include specific references to the sources when possible."""
    
    def _create_citations(self, retrieval_results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Create citation information"""
        citations = []
        for i, result in enumerate(retrieval_results):
            chunk = result.chunk
            doc = self.documents_store.get(chunk.document_id)
            if doc:
                citations.append({
                    "source_id": i + 1,
                    "document": doc.filename,
                    "page": chunk.page_number,
                    "chunk_id": chunk.id,
                    "similarity_score": result.similarity_score,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                })
        
        return citations
    
    def get_documents(self) -> List[Document]:
        """Get list of ingested documents"""
        return list(self.documents_store.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        return {
            "total_documents": len(self.documents_store),
            "total_chunks": vector_stats["total_chunks"],
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.ollama_model
        }

if __name__ == "__main__":
    # Example usage
    rag = RAGSystem()
    
    # Example: Ingest a document
    # doc_id = rag.ingest_document("path/to/your/document.pdf")
    
    # Example: Query the system
    # response = rag.query("What is the main topic of the document?")
    # print(f"Answer: {response.answer}")
    # print(f"Citations: {response.citations}")
    
    print("RAG System initialized. Ready for document ingestion and queries.")

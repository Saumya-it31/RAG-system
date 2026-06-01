import streamlit as st
import os
import tempfile
from pathlib import Path
import time

# Import our RAG system
from rag_system import RAGSystem, RAGConfig

# Page configuration
st.set_page_config(
    page_title="RAG System Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        margin-bottom: 30px;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #2E8B57;
    }
    .source-highlight {
        background-color: #FFE5B4;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    return RAGSystem()

def main():
    st.markdown("<h1 class='main-header'>📚 RAG System Demo</h1>", unsafe_allow_html=True)
    st.markdown("### Retrieval-Augmented Generation with Local LLM")
    
    # Initialize RAG system
    try:
        rag = initialize_rag_system()
        st.success("✅ RAG System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {e}")
        st.stop()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Process with RAG system
                        with st.spinner("Processing document... This may take a few minutes for scanned PDFs."):
                            doc_id = rag.ingest_document(tmp_path)
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                        st.success(f"✅ Document processed successfully!")
                        st.success(f"Document ID: {doc_id[:8]}...")
                        
                        # Refresh the app to show updated document list
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "tesseract" in error_msg.lower():
                            st.error("❌ Tesseract OCR Error: Please ensure Tesseract is properly installed and in your PATH.")
                            st.info("💡 Try restarting the application or check the installation guide.")
                        else:
                            st.error(f"❌ Error processing document: {error_msg}")
                        
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        # Document list
        st.subheader("📄 Ingested Documents")
        documents = rag.get_documents()
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc.filename}"):
                    st.write(f"**Pages:** {doc.total_pages}")
                    st.write(f"**Method:** {doc.processing_method}")
                    st.write(f"**Created:** {doc.created_at}")
                    st.write(f"**ID:** {doc.id[:8]}...")
        else:
            st.info("No documents uploaded yet")
        
        # System stats
        st.subheader("📊 System Statistics")
        stats = rag.get_stats()
        st.metric("Documents", stats['total_documents'])
        st.metric("Text Chunks", stats['total_chunks'])
        st.write(f"**Embedding Model:** {stats['embedding_model']}")
        st.write(f"**LLM Model:** {stats['llm_model']}")
    
    # Main query interface
    st.header("🔍 Ask Questions")
    
    # Document filter
    documents = rag.get_documents()
    doc_options = ["All Documents"] + [doc.filename for doc in documents]
    selected_doc = st.selectbox("Filter by Document", doc_options)
    
    # Query input
    question = st.text_area(
        "Enter your question:",
        placeholder="Ask anything about your uploaded documents...",
        height=100
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.1)
        with col2:
            max_tokens = st.slider("Max response tokens", 100, 2000, 1000, 100)
    
    if st.button("🚀 Ask Question", type="primary", disabled=not question or not documents):
        if not documents:
            st.warning("Please upload at least one document first!")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    # Determine document filter
                    document_filter = None
                    if selected_doc != "All Documents":
                        for doc in documents:
                            if doc.filename == selected_doc:
                                document_filter = doc.id
                                break
                    
                    # Update config temporarily
                    rag.config.top_k_retrieval = top_k
                    rag.config.similarity_threshold = similarity_threshold
                    
                    # Query the system
                    start_time = time.time()
                    response = rag.query(question, document_filter=document_filter)
                    
                    # Display results
                    st.header("📝 Answer")
                    st.write(response.answer)
                    
                    # Processing metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processing Time", f"{response.processing_time:.2f}s")
                    with col2:
                        st.metric("Sources Found", len(response.retrieval_results))
                    with col3:
                        st.metric("Context Length", response.metadata.get('context_length', 0))
                    
                    # Citations
                    if response.citations:
                        st.header("📚 Sources & Citations")
                        
                        for citation in response.citations:
                            with st.expander(f"📄 Source {citation['source_id']}: {citation['document']} (Page {citation['page']})"):
                                st.markdown(f"**Relevance Score:** {citation['similarity_score']:.3f}")
                                st.markdown(f"**Preview:** {citation['content_preview']}")
                                
                                # Show full chunk content
                                for result in response.retrieval_results:
                                    if result.rank == citation['source_id']:
                                        st.markdown("**Full Content:**")
                                        st.text_area(
                                            "Chunk Content", 
                                            result.chunk.content, 
                                            height=200,
                                            key=f"chunk_{citation['source_id']}"
                                        )
                                        break
                    
                    # Debug information
                    with st.expander("🔧 Debug Information"):
                        st.json({
                            "question": question,
                            "document_filter": document_filter,
                            "top_k": top_k,
                            "similarity_threshold": similarity_threshold,
                            "retrieval_results_count": len(response.retrieval_results),
                            "processing_time": response.processing_time
                        })
                        
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    st.exception(e)
    
    # Help section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### How to Use This RAG System
        
        1. **Upload Documents**: Use the sidebar to upload PDF documents to your knowledge base
        2. **Ask Questions**: Type your question in the text area above
        3. **Review Answers**: Get AI-generated answers with source citations
        4. **Explore Sources**: Click on citations to see the exact text that supports the answer
        
        ### Features
        - 📄 **Mixed PDF Support**: Handles both native and scanned PDFs
        - 🔍 **Semantic Search**: Uses advanced embeddings for relevant retrieval
        - 🤖 **Local LLM**: Powered by Ollama (completely offline)
        - 📚 **Source Citations**: Every answer includes traceable sources
        - ⚡ **Real-time Processing**: Fast ingestion and query processing
        
        ### Tips
        - Ask specific questions for better results
        - Use the document filter to search within specific documents
        - Adjust similarity threshold for more/fewer sources
        - Check the debug section for system insights
        """)

if __name__ == "__main__":
    main()

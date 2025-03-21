import gradio as gr
import os
import re
import json
import time
import logging
import hashlib
from functools import lru_cache
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Create directories for storage
os.makedirs("cache", exist_ok=True)
os.makedirs("user_data", exist_ok=True)

#-------------------------- DOCUMENT PROCESSING----------------------------
def get_document_text(file_list: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts text from various document types
    Returns both the full text and metadata about each document
    """
    full_text = ""
    doc_metadata = []
    
    for file in file_list:
        file_path = file.name
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        try:
            # Extract content based on file type
            if file_ext == '.pdf':
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
                doc_text = ""
                
                for i, page in enumerate(pdf_reader.pages):
                    content = page.extract_text()
                    if content:
                        doc_text += content
                    # Add page markers to help with context
                    if i < page_count - 1:
                        doc_text += f"\n\n--- Page {i+1} ---\n\n"
                
            elif file_ext == '.docx':
                doc_text = docx2txt.process(file)
                
            elif file_ext == '.txt':
                doc_text = file.read().decode('utf-8')
                
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                continue
                
            # Add document header
            doc_header = f"--- Document: {file_name} ---\n\n"
            full_text += doc_header + doc_text + "\n\n"
            
            # Calculate hash for caching purposes
            file.seek(0)
            file_hash = hashlib.md5(file.read()).hexdigest()
            
            # Store metadata
            doc_metadata.append({
                "filename": file_name,
                "filetype": file_ext,
                "hash": file_hash,
                "timestamp": datetime.now().isoformat(),
                "char_count": len(doc_text)
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            raise RuntimeError(f"Error processing file {file_name}: {str(e)}")
    
    return full_text, doc_metadata

def get_optimized_text_chunks(text: str) -> List[str]:
    """
    Improved text chunking with semantic boundaries and adaptive sizing
    """
    if not text.strip():
        return []
    
    # First try to split on semantic boundaries
    semantic_patterns = [
        r'(?<=\n\n--- Document: .*? ---\n\n)',  # Document boundaries
        r'(?<=\n\n--- Page \d+ ---\n\n)',       # Page boundaries
        r'(?<=\n## .*?\n)',                     # Markdown h2 headers
        r'(?<=\n# .*?\n)',                      # Markdown h1 headers
        r'(?<=\n\n)',                           # Double newlines
        r'(?<=\.)\s+(?=[A-Z])'                  # Sentences
    ]
    
    # Try each pattern in order of preference
    chunks = [text]
    for pattern in semantic_patterns:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) > 8000:  # Only split large chunks
                splits = re.split(pattern, chunk)
                new_chunks.extend([s for s in splits if s.strip()])
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
    
    # Apply final size limits with RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,           # Smaller chunks for better retrieval
        chunk_overlap=1000,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 6000:
            final_chunks.extend(text_splitter.split_text(chunk))
        else:
            final_chunks.append(chunk)
    
    # Add index numbers for reference
    for i, chunk in enumerate(final_chunks):
        final_chunks[i] = f"[Chunk {i+1} of {len(final_chunks)}]\n{chunk}"
    
    return final_chunks

def get_vector_store(text_chunks: List[str], doc_metadata: List[Dict[str, Any]]) -> FAISS:
    """
    Creates a FAISS vector store from text chunks with metadata
    """
    if not text_chunks:
        raise ValueError("No text chunks provided for embeddings.")
    
    # Save processed chunks for debugging/reference
    with open("cache/last_processed_chunks.json", "w") as f:
        json.dump({"chunks": text_chunks, "metadata": doc_metadata}, f, indent=2)
    
    logger.info(f"Creating embeddings for {len(text_chunks)} chunks")
    
    # Create metadata for each chunk
    metadatas = []
    for chunk in text_chunks:
        # Extract chunk number from the chunk header
        chunk_num = "Unknown"
        match = re.search(r'\[Chunk (\d+) of \d+\]', chunk)
        if match:
            chunk_num = match.group(1)
            
        # Extract document name if present
        doc_name = "Unknown"
        match = re.search(r'--- Document: (.*?) ---', chunk)
        if match:
            doc_name = match.group(1)
            
        metadatas.append({
            "chunk_id": chunk_num,
            "document": doc_name,
            "chunk_length": len(chunk),
            "timestamp": datetime.now().isoformat()
        })
    
    # Create embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # Save the vector store for persistence
        vector_store.save_local("cache/faiss_index")
        
        # Log success
        logger.info(f"Successfully created vector store with {len(text_chunks)} chunks")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {str(e)}")
        raise RuntimeError(f"Error creating embeddings: {str(e)}")

#-------------------------- RAG FUNCTIONALITY --------------------------
def get_conversational_chain():
    """Creates an LLM chain for answering questions with enhanced prompt"""
    prompt_template = """
    You are a knowledgeable job and career advisor specializing in providing information about job roles,
    skills, certifications, and career development based on the provided documents.
    
    When answering, follow these guidelines:
    1. Base your answers only on the information in the CONTEXT provided below
    2. If the information isn't in the context, say "I don't find specific information about this in the uploaded documents" instead of making up an answer
    3. Structure your answers clearly with bullet points or sections when appropriate
    4. Include specific details from the documents when relevant
    5. If multiple documents have different information, acknowledge this and present both perspectives
    
    CONTEXT: {context}
    
    QUESTION: {question}
    
    YOUR ANSWER:
    """
    
    # Use model with lower temperature for more factual responses
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest", 
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048
    )
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_answer(answer: str) -> str:
    """Cleans up and formats the LLM response"""
    # Remove any template artifacts
    answer = re.sub(r'YOUR ANSWER:', '', answer, flags=re.IGNORECASE)
    answer = answer.strip()
    
    # Add source attribution if we detect source references
    if re.search(r'\[Chunk \d+\]|\[Document:', answer):
        answer += "\n\n(Sources: Based on information from your uploaded documents)"
    
    return answer

def user_query(user_question: str, vector_store: Any, retry_count: int = 0) -> str:
    """Enhanced query processing with retry logic"""
    MAX_RETRIES = 2
    
    if not vector_store:
        return "⚠️ Please upload documents first and click 'Process Documents'."
    
    if not user_question:
        return "⚠️ Please enter a question."
    
    try:
        # Log the query
        logger.info(f"Processing query: {user_question}")
        
        # Get start time for performance monitoring
        start_time = time.time()
        
        # Search for relevant documents - increased k for better context
        docs = vector_store.similarity_search(user_question, k=4)
        
        # Log retrieved chunks for debugging
        chunk_ids = [doc.metadata.get('chunk_id', 'Unknown') for doc in docs]
        logger.info(f"Retrieved chunks: {chunk_ids}")
        
        # Get the conversational chain
        chain = get_conversational_chain()
        
        # Generate an answer
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        # Process the answer
        answer = process_answer(response['output_text'])
        
        # Log completion time
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed in {elapsed_time:.2f} seconds")
        
        return answer
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Handle rate limiting with retries
        if ("rate limit" in error_msg or "quota" in error_msg) and retry_count < MAX_RETRIES:
            logger.warning(f"Rate limit hit, retrying ({retry_count+1}/{MAX_RETRIES})...")
            time.sleep(2 ** retry_count)  # Exponential backoff
            return user_query(user_question, vector_store, retry_count + 1)
        
        # Log the error
        logger.error(f"Query error: {str(e)}")
        
        # User-friendly error messages
        if "rate limit" in error_msg:
            return "❌ The service is currently busy. Please try again in a moment."
        elif "api key" in error_msg:
            return "❌ API authentication error. Please contact the administrator."
        else:
            return f"❌ Error processing your question: {str(e)}"

#-------------------------- DOCUMENT MANAGEMENT --------------------------
def list_saved_documents() -> List[Dict[str, Any]]:
    """Returns a list of previously processed documents from metadata storage"""
    metadata_path = "user_data/document_metadata.json"
    
    if not os.path.exists(metadata_path):
        return []
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading document metadata: {str(e)}")
        return []

def save_document_metadata(metadata: List[Dict[str, Any]]) -> bool:
    """Saves document metadata to persistent storage"""
    metadata_path = "user_data/document_metadata.json"
    
    # Merge with existing metadata if any
    existing = list_saved_documents()
    
    # Check for duplicates by hash and update or add
    for new_doc in metadata:
        found = False
        for i, existing_doc in enumerate(existing):
            if existing_doc.get('hash') == new_doc.get('hash'):
                existing[i] = new_doc  # Update with new metadata
                found = True
                break
        
        if not found:
            existing.append(new_doc)
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(existing, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving document metadata: {str(e)}")
        return False

#-------------------------- DOCUMENT PROCESSING CACHE --------------------------
def get_cache_key(files: List[Any]) -> str:
    """Creates a cache key based on file hashes"""
    if not files:
        return ""
    
    # Create a hash of all file contents
    hasher = hashlib.md5()
    for file in files:
        file.seek(0)
        hasher.update(file.read())
        file.seek(0)
    
    return hasher.hexdigest()

def check_processing_cache(cache_key: str) -> Optional[FAISS]:
    """Checks if we've already processed these files and returns cached vector store"""
    if not cache_key:
        return None
        
    cache_file = f"cache/processed_{cache_key}.json"
    
    if os.path.exists(cache_file) and os.path.exists("cache/faiss_index"):
        try:
            # Load the cached vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local("cache/faiss_index", embeddings)
            logger.info(f"Loaded vector store from cache: {cache_key}")
            return vector_store
        except Exception as e:
            logger.warning(f"Cache hit but failed to load vector store: {str(e)}")
            return None
    
    return None

def save_processing_cache(cache_key: str, metadata: List[Dict[str, Any]]) -> None:
    """Saves cache information for future use"""
    if not cache_key:
        return
        
    cache_file = f"cache/processed_{cache_key}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save processing cache: {str(e)}")

#-------------------------- QUERY HISTORY --------------------------
def save_query_history(question: str, answer: str) -> int:
    """Saves a query and answer to history and returns the history ID"""
    history_file = "user_data/query_history.json"
    
    try:
        # Load existing history
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Generate ID
        history_id = len(history) + 1
        
        # Add new entry
        history.append({
            "id": history_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        return history_id
        
    except Exception as e:
        logger.error(f"Error saving query history: {str(e)}")
        return -1

def get_query_history() -> List[Dict[str, Any]]:
    """Returns the query history"""
    history_file = "user_data/query_history.json"
    
    if not os.path.exists(history_file):
        return []
        
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading query history: {str(e)}")
        return []

#-------------------------- GRADIO INTERFACE --------------------------
def process_uploaded_files(files, existing_vector_store=None):
    """Enhanced file processing with caching"""
    if not files:
        return existing_vector_store, "⚠️ No files uploaded. Please upload at least one document file."
    
    try:
        # Check for cached processing
        cache_key = get_cache_key(files)
        cached_store = check_processing_cache(cache_key)
        
        if cached_store:
            logger.info(f"Using cached vector store for {len(files)} files")
            return cached_store, "✅ Using cached document processing. Documents ready for querying!"
        
        # If no cache hit, process the files
        logger.info(f"Processing {len(files)} files")
        
        # Extract text from documents
        raw_text, doc_metadata = get_document_text(files)
        
        # Save the document metadata
        save_document_metadata(doc_metadata)
        
        # Split text into chunks
        text_chunks = get_optimized_text_chunks(raw_text)
        
        if not text_chunks:
            return None, "⚠️ No text content could be extracted from the uploaded files."
        
        # Create vector store
        vector_store = get_vector_store(text_chunks, doc_metadata)
        
        # Save to cache
        save_processing_cache(cache_key, doc_metadata)
        
        return vector_store, f"✅ Successfully processed {len(files)} documents with {len(text_chunks)} content chunks. Ready for queries!"
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return None, f"❌ Error processing documents: {str(e)}"

def handle_query(question, vector_store_state):
    """Enhanced query handler with history"""
    if vector_store_state is None:
        return "⚠️ Please upload documents first and click 'Process Documents'."
    
    try:
        # Process the query
        answer = user_query(question, vector_store_state)
        
        # Save to history
        save_query_history(question, answer)
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in query handling: {str(e)}")
        return f"❌ Error processing your question: {str(e)}"

def get_document_stats(vector_store_state):
    """Returns statistics about currently loaded documents"""
    if vector_store_state is None:
        return "No documents loaded."
    
    try:
        # Get metadata
        doc_metadata = list_saved_documents()
        
        if not doc_metadata:
            return "Documents loaded but no metadata available."
        
        # Calculate stats
        doc_count = len(doc_metadata)
        total_size = sum(doc.get('char_count', 0) for doc in doc_metadata)
        doc_types = set(doc.get('filetype', '') for doc in doc_metadata)
        
        stats = f"📊 Document Stats:\n" \
               f"• Documents: {doc_count}\n" \
               f"• File types: {', '.join(doc_types)}\n" \
               f"• Total content size: {total_size:,} characters\n" \
               f"• Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        return "Unable to retrieve document statistics."

def load_query_history_to_ui():
    """Loads and formats query history for display"""
    history = get_query_history()
    
    if not history:
        return "No query history available."
    
    # Only show the last 5 items, newest first
    recent_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
    
    formatted_history = "### Recent Queries\n\n"
    
    for item in recent_history:
        q = item.get('question', '')
        timestamp = datetime.fromisoformat(item.get('timestamp', '')).strftime('%m/%d %H:%M')
        formatted_history += f"**{timestamp}**: {q}\n\n"
    
    return formatted_history

# Custom CSS for improved styling
custom_css = """
:root {
    --primary-color: #2563eb;
    --secondary-color: #1e40af;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --background-color: #f9fafb;
    --card-background: #ffffff;
    --text-color: #1f2937;
    --border-radius: 8px;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --transition: all 0.3s ease;
}

/* Dark mode support with media query */
@media (prefers-color-scheme: dark) {
    .darkmode-enabled {
        --background-color: #111827;
        --card-background: #1f2937;
        --text-color: #f3f4f6;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
    }
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    transition: var(--transition);
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

.main-header {
    color: var(--primary-color);
    font-weight: 700;
    font-size: 2.25rem;
    margin-bottom: 0.5rem;
    text-align: center;
}

.sub-header {
    font-size: 1.1rem;
    color: #4b5563;
    margin-bottom: 2rem;
    text-align: center;
}

.container {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: var(--transition);
}

button.primary {
    background-color: var(--primary-color) !important;
    color: white !important;
    border-radius: var(--border-radius) !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
}

button.primary:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-1px);
}

button.secondary {
    background-color: #f3f4f6 !important;
    color: var(--text-color) !important;
    border: 1px solid #e5e7eb !important;
    border-radius: var(--border-radius) !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
}

button.secondary:hover {
    background-color: #e5e7eb !important;
    transform: translateY(-1px);
}

.footer {
    text-align: center;
    font-size: 0.875rem;
    color: #6b7280;
    margin-top: 2rem;
}

.status-success {
    color: var(--success-color);
    font-weight: 600;
}

.status-warning {
    color: var(--warning-color);
    font-weight: 600;
}

.status-error {
    color: var(--error-color);
    font-weight: 600;
}

.icon {
    vertical-align: middle;
    margin-right: 0.5rem;
}

/* Animation for processing */
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

.processing {
  animation: pulse 1.5s infinite;
}

/* Responsive improvements */
@media (max-width: 768px) {
    .main-header {
        font-size: 1.75rem;
    }
    
    .sub-header {
        font-size: 0.9rem;
    }
    
    .container {
        padding: 1rem;
    }
}

/* Chat-like message display */
.message-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 1rem;
    border-radius: var(--border-radius);
    background-color: #f3f4f6;
    margin-bottom: 1rem;
}

.message {
    padding: 0.75rem;
    border-radius: var(--border-radius);
    margin-bottom: 0.75rem;
}

.message-user {
    background-color: #e5e7eb;
    margin-left: 2rem;
    margin-right: 0.5rem;
}

.message-assistant {
    background-color: #dbeafe;
    margin-right: 2rem;
    margin-left: 0.5rem;
}
"""

# Create enhanced Gradio interface
with gr.Blocks(css=custom_css, title="Job Role Resources Assistant") as demo:
    gr.HTML("""
        <h1 class="main-header">🧠 Advanced Job Role Resources Assistant</h1>
        <p class="sub-header">Upload documents containing job role resources, then ask questions to get personalized guidance</p>
    """)
    
    # State variables for the application
    vector_store_state = gr.State(None)
    conversation_state = gr.State([])
    is_dark_mode = gr.State(False)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("📚 Documents & Questions"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="container"):
                        gr.Markdown("### 📁 Document Management")
                        
                        file_input = gr.File(
                            label="Upload Documents",
                            file_types=[".pdf", ".docx", ".txt"],
                            file_count="multiple"
                        )
                        
                        with gr.Row():
                            process_button = gr.Button("Process Documents", elem_classes="primary")
                            clear_button = gr.Button("Clear All", elem_classes="secondary")
                        
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        doc_stats = gr.Textbox(
                            label="Document Statistics",
                            interactive=False,
                            value="No documents loaded."
                        )
                        
                        gr.Markdown("""
                        ### 💡 Instructions
                        1. Upload one or more document files (.pdf, .docx, .txt)
                        2. Click "Process Documents" and wait for confirmation
                        3. Ask questions about job roles in the panel on the right
                        4. Previous documents will be cached for faster loading
                        """)
                        
                        # Add recent query history display
                        history_display = gr.Markdown(
                            value="No recent queries.",
                            label="Recent Queries"
                        )
                
                with gr.Column(scale=2):
                    with gr.Group(elem_classes="container"):
                        gr.Markdown("### 🔍 Ask Questions")
                        
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="What resources are available for a Data Scientist role?",
                            elem_id="question-input"
                        )
                        
                        with gr.Row():
                            query_button = gr.Button("Get Answer", elem_classes="primary")
                            clear_question = gr.Button("Clear", elem_classes="secondary")
                        
                        gr.HTML("""
                            <div style="text-align: center; margin: 10px 0;">
                                <span style="color: #6b7280; font-size: 0.9rem;">Examples:</span>
                            </div>
                        """)
                        
                        with gr.Row():
                            example1 = gr.Button("What skills are needed for a Machine Learning Engineer?")
                            example2 = gr.Button("How to prepare for a Data Scientist interview?")
                        
                        with gr.Row():
                            example3 = gr.Button("What certifications are valuable for a Cloud Engineer?")
                            example4 = gr.Button("Compare Data Analyst vs Business Analyst roles")
                        
                        answer_output = gr.Textbox(
                            label="Answer",
                            interactive=False,
                            lines=15
                        )
                        
                        with gr.Row():
                            feedback_positive = gr.Button("👍 Helpful", elem_classes="secondary")
                            feedback_negative = gr.Button("👎 Not Helpful", elem_classes="secondary")
                            explain_more = gr.Button("🔍 Explain Further", elem_classes="secondary")
        
        with gr.TabItem("📊 Analytics"):
            gr.Markdown("### Document Analysis")
            
            with gr.Row():
                with gr.Column():
                    summarize_button = gr.Button("Generate Document Summary", elem_classes="primary")
                    summary_output = gr.Textbox(
                        label="Document Summary",
                        interactive=False,
                        lines=10
                    )
                
                with gr.Column():
                    extract_button = gr.Button("Extract Key Skills & Requirements", elem_classes="primary")
                    skills_output = gr.Textbox(
                        label="Key Skills & Requirements",
                        interactive=False,
                        lines=10
                    )
            
            with gr.Group(elem_classes="container"):
                gr.Markdown("### Content Analytics")
                
                with gr.Row():
                    topics_button = gr.Button("Analyze Document Topics", elem_classes="primary")
                    visualize_button = gr.Button("Generate Skills Visualization", elem_classes="primary")
                
                analytics_output = gr.Textbox(
                    label="Analytics Results",
                    interactive=False,
                    lines=10
                )
        
        with gr.TabItem("⚙️ Settings & Help"):
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="container"):
                        gr.Markdown("### Application Settings")
                        
                        with gr.Row():
                            dark_mode = gr.Checkbox(label="Dark Mode", value=False)
                            verbose_mode = gr.Checkbox(label="Verbose Mode (Show Processing Details)", value=False)
                        
                        with gr.Row():
                            chunk_size = gr.Slider(
                                minimum=1000, 
                                maximum=10000, 
                                value=6000, 
                                step=500, 
                                label="Document Chunk Size"
                            )
                            
                            context_size = gr.Slider(
                                minimum=1, 
                                maximum=10, 
                                value=4, 
                                step=1, 
                                label="Context Chunks Per Query"
                            )
                        
                        reset_button = gr.Button("Reset to Defaults", elem_classes="secondary")
                
                with gr.Column():
                    with gr.Group(elem_classes="container"):
                        gr.Markdown("### Help & Information")
                        
                        gr.Markdown("""
                        #### About this Application
                        
                        This application helps you analyze job role resources and extract relevant information about different positions, required skills, certifications, and career paths.
                        
                        #### Supported File Types
                        - PDF Documents (.pdf)
                        - Word Documents (.docx)
                        - Text Files (.txt)
                        
                        #### Example Questions
                        - What skills are required for a Data Scientist role?
                        - How do I prepare for a Cloud Architect certification?
                        - What's the career progression path for a Frontend Developer?
                        - Compare the responsibilities of a Project Manager and a Product Manager
                        - What tools should I learn for a DevOps Engineer position?
                        
                        #### Troubleshooting
                        - If processing fails, try uploading smaller documents
                        - Some PDFs with complex formatting may not process correctly
                        - For complex questions, try breaking them down into simpler ones
                        """)
    
    # Add footer
    gr.HTML("""
        <div class="footer">
            <p>Job Role Resources Assistant © 2024 | Built with Gradio, LangChain, and Google AI</p>
        </div>
    """)
    
    #-------------------------- EVENT HANDLERS --------------------------
    
    # Document processing
    def handle_process_documents(files):
        if not files:
            return None, "⚠️ No files uploaded. Please upload at least one document."
        
        try:
            return process_uploaded_files(files)
        except Exception as e:
            logger.error(f"Error in handle_process_documents: {str(e)}")
            return None, f"❌ Error processing documents: {str(e)}"
    
    # Clear documents
    def handle_clear_documents():
        # Remove cache files
        try:
            for file in os.listdir("cache"):
                if file.startswith("processed_") or file == "faiss_index":
                    file_path = os.path.join("cache", file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
            
            return None, "✅ All documents and caches cleared.", "No documents loaded.", "No recent queries."
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return None, f"❌ Error clearing documents: {str(e)}", "Error retrieving stats.", "Error retrieving history."
    
    # Handle questions
    def handle_submit_question(question, vector_store):
        if not question.strip():
            return "⚠️ Please enter a question."
        
        if vector_store is None:
            return "⚠️ Please upload and process documents first."
        
        try:
            # Process the query
            answer = user_query(question, vector_store)
            
            # Save to history
            save_query_history(question, answer)
            
            # Update history display
            updated_history = load_query_history_to_ui()
            
            return answer, updated_history
        except Exception as e:
            logger.error(f"Error handling question: {str(e)}")
            return f"❌ Error processing your question: {str(e)}", "Error updating history."
    
    # Example button handlers
    def handle_example(example_text, vector_store):
        if vector_store is None:
            return example_text, "⚠️ Please upload and process documents first.", "No recent queries."
        
        try:
            # Process the query
            answer = user_query(example_text, vector_store)
            
            # Save to history
            save_query_history(example_text, answer)
            
            # Update history display
            updated_history = load_query_history_to_ui()
            
            return example_text, answer, updated_history
        except Exception as e:
            logger.error(f"Error handling example: {str(e)}")
            return example_text, f"❌ Error processing example: {str(e)}", "Error updating history."
    
    # Document summary handler
    def generate_document_summary(vector_store):
        if vector_store is None:
            return "⚠️ Please upload and process documents first."
        
        try:
            # Create a query to generate a summary
            summary_query = "Provide a comprehensive summary of all the job roles, skills, and requirements mentioned in these documents."
            
            # Process the query with expanded context
            docs = vector_store.similarity_search(summary_query, k=6)  # More context for summaries
            
            # Get the LLM chain with a summary-specific prompt
            summary_prompt = """
            You are a job and career advisor assistant. Generate a concise yet comprehensive summary of the
            job roles, skills, and requirements mentioned in the provided documents.
            
            Focus on:
            1. Key job roles mentioned
            2. Common skills across roles
            3. Education and certification requirements
            4. Career progression paths if mentioned
            
            Format the summary with clear sections and bullet points.
            
            CONTEXT: {context}
            
            YOUR SUMMARY:
            """
            
            # Use model with higher temperature for more creative summary
            model = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-pro-latest", 
                temperature=0.4,
                max_output_tokens=2048
            )
            
            prompt = PromptTemplate(
                template=summary_prompt, 
                input_variables=["context"]
            )
            
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            
            # Generate the summary
            response = chain(
                {"input_documents": docs, "question": summary_query},
                return_only_outputs=True
            )
            
            # Process the response
            summary = response['output_text'].replace("YOUR SUMMARY:", "").strip()
            
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"❌ Error generating document summary: {str(e)}"
    
    # Extract skills handler
    def extract_key_skills(vector_store):
        if vector_store is None:
            return "⚠️ Please upload and process documents first."
        
        try:
            # Create a query to extract skills
            skills_query = "Extract and organize all skills, technical requirements, and certifications mentioned in these documents."
            
            # Process the query
            docs = vector_store.similarity_search(skills_query, k=5)
            
            # Get the LLM chain with a skills-specific prompt
            skills_prompt = """
            You are a job skills analyzer. Extract and organize all the skills, technical requirements, 
            and certifications mentioned in the provided documents.
            
            Format your output as follows:
            
            ### Technical Skills
            - List technical skills with proficiency levels when mentioned
            
            ### Soft Skills
            - List soft/interpersonal skills
            
            ### Certifications & Qualifications
            - List all certifications, degrees, and qualifications
            
            ### Tools & Technologies
            - List all software, platforms, and technologies
            
            Only include items that are explicitly mentioned in the documents. Organize similar skills together.
            
            CONTEXT: {context}
            
            YOUR ANALYSIS:
            """
            
            # Use model with lower temperature for more factual extraction
            model = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-pro-latest", 
                temperature=0.1,
                max_output_tokens=2048
            )
            
            prompt = PromptTemplate(
                template=skills_prompt, 
                input_variables=["context"]
            )
            
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            
            # Generate the skills analysis
            response = chain(
                {"input_documents": docs, "question": skills_query},
                return_only_outputs=True
            )
            
            # Process the response
            skills = response['output_text'].replace("YOUR ANALYSIS:", "").strip()
            
            return skills
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return f"❌ Error extracting key skills: {str(e)}"
    
    # Analyze document topics
    def analyze_document_topics(vector_store):
        if vector_store is None:
            return "⚠️ Please upload and process documents first."
        
        try:
            # Create a query to analyze topics
            topics_query = "Identify and analyze the main topics and themes across all the uploaded documents."
            
            # Process the query
            docs = vector_store.similarity_search(topics_query, k=6)
            
            # Get the LLM chain with a topics-specific prompt
            topics_prompt = """
            You are a content analyst specialized in job market documentation. Analyze the provided documents 
            and identify the main topics, themes, and trends mentioned across them.
            
            For each major topic identified:
            1. Provide a brief description of what the topic covers
            2. Mention which documents or sections contain this topic
            3. Highlight any contradictions or notable differences in how topics are presented
            4. Note any emerging trends or shifts in requirements
            
            Format your analysis with clear sections and bullet points.
            
            CONTEXT: {context}
            
            YOUR TOPIC ANALYSIS:
            """
            
            # Use model with balanced temperature
            model = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-pro-latest", 
                temperature=0.3,
                max_output_tokens=2048
            )
            
            prompt = PromptTemplate(
                template=topics_prompt, 
                input_variables=["context"]
            )
            
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            
            # Generate the topic analysis
            response = chain(
                {"input_documents": docs, "question": topics_query},
                return_only_outputs=True
            )
            
            # Process the response
            analysis = response['output_text'].replace("YOUR TOPIC ANALYSIS:", "").strip()
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing topics: {str(e)}")
            return f"❌ Error analyzing document topics: {str(e)}"
    
    # Generate skills visualization (text-based for simplicity)
    def generate_skills_visualization(vector_store):
        if vector_store is None:
            return "⚠️ Please upload and process documents first."
        
        try:
            # Create a query to generate skills visualization data
            viz_query = "What are the most frequently mentioned skills across all job roles in these documents, and how are they categorized?"
            
            # Process the query
            docs = vector_store.similarity_search(viz_query, k=5)
            
            # Get the LLM chain with a visualization-specific prompt
            viz_prompt = """
            You are a data visualization specialist focusing on job market skills. Based on the provided documents,
            create a text-based visualization of the most important skills mentioned.
            
            Format your response as a skills heat map using text formatting:
            1. Group skills into categories (Technical, Soft Skills, Domain Knowledge, etc.)
            2. For each skill, indicate its relative importance/frequency using symbols:
               - ███ High importance/frequency
               - ██ Medium importance/frequency
               - █ Lower importance/frequency
            3. Include a simple count or percentage if mentioned in the documents
            
            Example format:
            # TECHNICAL SKILLS
            Python Programming  ███ (85%)
            SQL                 ██ (67%)
            JavaScript          █ (35%)
            
            # SOFT SKILLS
            Communication       ███ (90%)
            Problem Solving     ██ (70%)
            
            Only include skills that are explicitly mentioned in the documents.
            
            CONTEXT: {context}
            
            YOUR SKILLS VISUALIZATION:
            """
            
            # Use model with balanced temperature
            model = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-pro-latest", 
                temperature=0.2,
                max_output_tokens=2048
            )
            
            prompt = PromptTemplate(
                template=viz_prompt, 
                input_variables=["context"]
            )
            
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            
            # Generate the visualization
            response = chain(
                {"input_documents": docs, "question": viz_query},
                return_only_outputs=True
            )
            
            # Process the response
            visualization = response['output_text'].replace("YOUR SKILLS VISUALIZATION:", "").strip()
            
            return visualization
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return f"❌ Error generating skills visualization: {str(e)}"
    
    # Clear question
    def clear_question_field():
        return ""
    
    # Explain further handler
    def explain_further(current_answer, vector_store):
        if not current_answer or "⚠️" in current_answer or "❌" in current_answer:
            return "Please get an answer first before requesting further explanation."
        
        if vector_store is None:
            return "⚠️ Please upload and process documents first."
        
        try:
            # Create a query to explain further
            explain_query = f"Please explain the following in more detail, providing additional context and examples if available: {current_answer}"
            
            # Process the query
            docs = vector_store.similarity_search(explain_query, k=5)
            
            # Get the LLM chain with an explanation-specific prompt
            explain_prompt = """
            You are a job and career advisor assistant. The user has requested more details about a previous answer.
            
            Previous answer: {question}
            
            Provide a more detailed explanation of the concepts mentioned in the previous answer.
            Include:
            1. More specific examples
            2. Additional context
            3. Practical applications or implications
            4. Any nuances or exceptions worth noting
            
            Base your expanded explanation ONLY on information found in the documents.
            
            CONTEXT: {context}
            
            YOUR EXPANDED EXPLANATION:
            """
            
            model = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-pro-latest", 
                temperature=0.3,
                max_output_tokens=2048
            )
            
            prompt = PromptTemplate(
                template=explain_prompt, 
                input_variables=["context", "question"]
            )
            
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            
            # Generate the explanation
            response = chain(
                {"input_documents": docs, "question": current_answer},
                return_only_outputs=True
            )
            
            # Process the response
            explanation = response['output_text'].replace("YOUR EXPANDED EXPLANATION:", "").strip()
            
            # Format the final output
            final_output = "### Further Explanation\n\n" + explanation
            
            return final_output
        except Exception as e:
            logger.error(f"Error generating further explanation: {str(e)}")
            return f"❌ Error generating further explanation: {str(e)}"
    
    # Feedback handlers
    def record_positive_feedback(question, answer):
        try:
            feedback_file = "user_data/feedback.json"
            
            # Load existing feedback
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback = json.load(f)
            else:
                feedback = []
            
            # Add new feedback
            feedback.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "rating": "positive",
                "details": None
            })
            
            # Save updated feedback
            with open(feedback_file, 'w') as f:
                json.dump(feedback, f, indent=2)
                
            return "Thank you for your positive feedback!"
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return "Error recording feedback."
    
    def record_negative_feedback(question, answer):
        try:
            feedback_file = "user_data/feedback.json"
            
            # Load existing feedback
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback = json.load(f)
            else:
                feedback = []
            
            # Add new feedback
            feedback.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "rating": "negative",
                "details": None
            })
            
            # Save updated feedback
            with open(feedback_file, 'w') as f:
                json.dump(feedback, f, indent=2)
                
            return "Thank you for your feedback. We'll work to improve our responses."
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return "Error recording feedback."
    
    # Dark mode toggle
    def toggle_dark_mode(is_dark):
        # This would normally update CSS classes, but in Gradio it's limited
        # For a real implementation, you'd need JavaScript
        return not is_dark
    
    # Reset settings
    def reset_settings():
        return False, False, 6000, 4
    
    # Update document stats periodically
    def update_doc_stats(vector_store):
        if vector_store is None:
            return "No documents loaded."
        return get_document_stats(vector_store)
    
    # Connect event handlers
    process_button.click(
        handle_process_documents,
        inputs=[file_input],
        outputs=[vector_store_state, status_output]
    ).then(
        update_doc_stats,
        inputs=[vector_store_state],
        outputs=[doc_stats]
    ).then(
        load_query_history_to_ui,
        outputs=[history_display]
    )
    
    clear_button.click(
        handle_clear_documents,
        outputs=[vector_store_state, status_output, doc_stats, history_display]
    )
    
    query_button.click(
        handle_submit_question,
        inputs=[question_input, vector_store_state],
        outputs=[answer_output, history_display]
    )
    
    clear_question.click(
        clear_question_field,
        outputs=[question_input]
    )
    
    # Example button handlers
    example1.click(
        handle_example,
        inputs=[example1, vector_store_state],
        outputs=[question_input, answer_output, history_display]
    )
    
    example2.click(
        handle_example,
        inputs=[example2, vector_store_state],
        outputs=[question_input, answer_output, history_display]
    )
    
    example3.click(
        handle_example,
        inputs=[example3, vector_store_state],
        outputs=[question_input, answer_output, history_display]
    )
    
    example4.click(
        handle_example,
        inputs=[example4, vector_store_state],
        outputs=[question_input, answer_output, history_display]
    )
    
    # Analytics tab handlers
    summarize_button.click(
        generate_document_summary,
        inputs=[vector_store_state],
        outputs=[summary_output]
    )
    
    extract_button.click(
        extract_key_skills,
        inputs=[vector_store_state],
        outputs=[skills_output]
    )
    
    topics_button.click(
        analyze_document_topics,
        inputs=[vector_store_state],
        outputs=[analytics_output]
    )
    
    visualize_button.click(
        generate_skills_visualization,
        inputs=[vector_store_state],
        outputs=[analytics_output]
    )
    
    # Feedback handlers
    feedback_positive.click(
        record_positive_feedback,
        inputs=[question_input, answer_output],
        outputs=[status_output]
    )
    
    feedback_negative.click(
        record_negative_feedback,
        inputs=[question_input, answer_output],
        outputs=[status_output]
    )
    
    explain_more.click(
        explain_further,
        inputs=[answer_output, vector_store_state],
        outputs=[answer_output]
    )
    
    # Settings tab handlers
    dark_mode.change(
        toggle_dark_mode,
        inputs=[is_dark_mode],
        outputs=[is_dark_mode]
    )
    
    reset_button.click(
        reset_settings,
        outputs=[dark_mode, verbose_mode, chunk_size, context_size]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(share=True)
 
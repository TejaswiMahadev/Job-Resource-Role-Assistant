import gradio as gr
import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

#-------------------------- PDF PROCESSING----------------------------
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks"""
    if not text_chunks:
        raise ValueError("No text chunks provided for embeddings.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error creating FAISS vector store: {e}")

#-------------------------- RAG FUNCTIONALITY --------------------------
def get_conversational_chain():
    """Creates an LLM chain for answering questions"""
    prompt_template = """
    You are a helpful assistant that provides information about job role resources and preparation guidelines. 
    Use the following context to answer the question. If you don't know the answer, just say that you don't know, 
    don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {question}
    
    YOUR ANSWER:
    """
    
    # Use a model from the available list
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_answer(answer):
    """Cleans up the LLM response"""
    # Remove any special tokens or formatting that might be in the response
    answer = re.sub(r'YOUR ANSWER:', '', answer, flags=re.IGNORECASE)
    return answer.strip()

def user_query(user_question, vector_store):
    """Retrieves relevant documents and generates an answer"""
    if not vector_store:
        return "⚠️ Please upload a PDF document first and click 'Process PDF'."
    
    if not user_question:
        return "⚠️ Please enter a question."
    
    try:
        # Search for relevant documents
        docs = vector_store.similarity_search(user_question, k=3)
        
        # Get the conversational chain
        chain = get_conversational_chain()
        
        # Generate an answer
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        # Process the answer
        answer = process_answer(response['output_text'])
        return answer
    
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

#-------------------------- GRADIO INTERFACE --------------------------
def process_uploaded_files(pdf_files):
    """Processes uploaded PDF files and creates a vector store"""
    if not pdf_files:
        return None, "⚠️ No PDF files uploaded. Please upload at least one PDF file."
    
    try:
        # Extract text from PDFs
        raw_text = get_pdf_text(pdf_files)
        
        # Split text into chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Create vector store
        vector_store = get_vector_store(text_chunks)
        
        return vector_store, "✅ PDF processed successfully! You can now ask questions about job role resources."
    
    except Exception as e:
        return None, f"❌ Error processing PDF: {str(e)}"

def handle_query(question, vector_store_state):
    """Handles user queries"""
    if vector_store_state is None:
        return "⚠️ Please upload a PDF document first and click 'Process PDF'."
    
    return user_query(question, vector_store_state)

# Create Gradio interface with custom styling
with gr.Blocks( title="Job Role Resources Assistant") as demo:
    gr.HTML("""
        <h1 class="main-header">🧠 Job Role Resources Assistant</h1>
        <p class="sub-header">Upload PDF documents containing job role resources, then ask questions to get personalized guidance</p>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="container"):
                gr.Markdown("### 📁 Upload Documents")
                file_input = gr.File(
                    label="Upload PDF Documents",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                process_button = gr.Button("Process PDFs", elem_classes="primary")
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                gr.Markdown("""
                ### 💡 Instructions
                1. Upload one or more PDF documents
                2. Click "Process PDFs" and wait for confirmation
                3. Ask questions about job roles in the panel on the right
                """)
        
        with gr.Column(scale=2):
            with gr.Group(elem_classes="container"):
                gr.Markdown("### 🔍 Ask Questions")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What resources are available for a Data Scientist role?",
                    elem_id="question-input"
                )
                
                query_button = gr.Button("Get Answer", elem_classes="primary")
                
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
                    lines=12
                )
    
    # Add footer
    gr.HTML("""
        <div class="footer">
            <p>Powered by Gemini 1.5 Pro and Langchain | Created with Gradio</p>
        </div>
    """)
    
    # State for vector store
    vector_store_state = gr.State(None)
    
    # Set up event handlers
    process_button.click(
        fn=process_uploaded_files,
        inputs=[file_input],
        outputs=[vector_store_state, status_output]
    )
    
    query_button.click(
        fn=handle_query,
        inputs=[question_input, vector_store_state],
        outputs=answer_output
    )
    
    # Allow pressing Enter to submit questions
    question_input.submit(
        fn=handle_query,
        inputs=[question_input, vector_store_state],
        outputs=answer_output
    )
    
    # Example question handlers
    for example_button, example_text in [
        (example1, "What skills are needed for a Machine Learning Engineer?"),
        (example2, "How to prepare for a Data Scientist interview?"),
        (example3, "What certifications are valuable for a Cloud Engineer?"),
        (example4, "Compare Data Analyst vs Business Analyst roles")
    ]:
        example_button.click(
            lambda text: text,
            [example_button],
            [question_input],
            queue=False
        ).then(
            fn=handle_query,
            inputs=[question_input, vector_store_state],
            outputs=answer_output
        )

# Launch the demo
if __name__ == "__main__":
    demo.launch(share=True)
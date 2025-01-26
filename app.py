import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_postgres.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
import boto3
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define PostgreSQL connection URL
connection_url = (
    f"postgresql+psycopg2://{os.getenv('PGVECTOR_USER')}:{os.getenv('PGVECTOR_PASSWORD')}"
    f"@{os.getenv('PGVECTOR_HOST')}:{os.getenv('PGVECTOR_PORT')}/{os.getenv('PGVECTOR_DATABASE')}"
)

# Define Bedrock client
BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name="eu-central-1")

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
            logger.error(f"Failed to read PDF {pdf.name}: {e}")
    return text

# Split text into chunks
def get_text_chunks(text):
    """Split long text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Store embeddings in PostgreSQL database
def store_embeddings_in_db(text_chunks):
    """Store text embeddings in the PGVector database."""
    try:
        # Initialize Bedrock Embeddings
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", 
            client=BEDROCK_CLIENT
        )
        
        # Use PGVector to store embeddings
        vectorstore = PGVector.from_texts(
            texts=text_chunks, 
            embedding=embeddings,  # Pass the embedding object
            connection=connection_url
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error storing embeddings in the database: {e}")
        logger.error(f"Database error: {e}")
        return None

# Streamlit app
def main():
    st.title("Document Upload and Embedding Storage")

    # Upload PDF files
    pdf_docs = st.file_uploader("Upload your PDFs here", type="pdf", accept_multiple_files=True)

    # Button to process and store embeddings
    if st.button("Process and Store Embeddings"):
        if pdf_docs:
            with st.spinner("Processing your documents..."):
                try:
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)

                    # Check if text extraction was successful
                    if not raw_text.strip():
                        st.warning("No text could be extracted from the uploaded PDFs.")
                        return

                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Store embeddings in the database
                    vectorstore = store_embeddings_in_db(text_chunks)

                    if vectorstore:
                        st.success("Embeddings stored successfully!", icon="✅")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Processing error: {e}")
        else:
            st.warning("Please upload PDF documents before processing.")

if __name__ == "__main__":
    main()

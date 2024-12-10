import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter  # Used to split large text into smaller chunks
from langchain_openai import OpenAIEmbeddings  # Embedding model for vectorization
from langchain_community.vectorstores import FAISS  # FAISS vector store for fast search and retrieval
from langchain.chains import RetrievalQA  # Used to create the question-answering pipeline
from langchain_openai import OpenAI  # Language model for answering user questions

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function that processes a PDF, stores vectorized text, 
    and allows the user to ask interactive questions about the document.
    """
    # Check if PDF filename is provided as a command-line argument
    if len(sys.argv) != 2:
        print("\nUsage: python script.py <pdf_name>\n")
        sys.exit(1)
    
    # Get the PDF file name from the command-line argument
    pdf_name = sys.argv[1]

    # Get the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit(1)  # Exit if no API key is found

    # Extract text from the PDF file
    text = extract_data(pdf_name)
    if not text.strip():
        sys.exit(1)  # Exit if no text is extracted from the PDF

    # Split the text into smaller chunks for processing
    docs = split_text(text)
    if not docs:
        sys.exit(1)  # Exit if the document could not be split into chunks

    # Create a FAISS vector store from the document chunks
    docstorage = vectorize_and_store(docs, api_key)
    
    # Allow the user to ask questions interactively
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        
        # If the user types 'exit', end the program
        if question.lower().strip() == 'exit':
            break

        # Skip if the question is empty
        if not question.strip():
            continue

        # Get the answer from the vectorized document and language model
        response = answer_question(question, api_key, docstorage)
        print("\n--- Response ---\n", response)


def extract_data(pdf_name: str) -> str:
    """
    Extracts all the text from a PDF file.
    Args:
        pdf_name (str): The name or path to the PDF file.
    Returns:
        str: The full extracted text from the PDF.
    """
    try:
        # Load the PDF using the PyPDFLoader
        loader = PyPDFLoader(pdf_name)
        data = loader.load()
    except Exception:
        sys.exit(1)  # Exit if there's an error loading the PDF
    
    if not data:
        sys.exit(1)  # Exit if no data is extracted from the PDF
    
    policy_text_parts = []
    for doc in data:
        # If the document part is a dictionary and has 'text', extract it
        if isinstance(doc, dict) and 'text' in doc:
            policy_text_parts.append(doc['text'])
        # If the document part is already a string, add it as is
        elif isinstance(doc, str):
            policy_text_parts.append(doc)
        # If the document part is neither a string nor a dictionary, convert it to a string
        else:
            policy_text_parts.append(str(doc))
    
    # Combine all the parts of the extracted text into one single string
    return ''.join(policy_text_parts)


def split_text(text: str):
    """
    Splits the extracted text into smaller chunks for easier processing.
    Args:
        text (str): The full extracted text from the PDF.
    Returns:
        List[str]: A list of smaller chunks of the text.
    """
    # Create a text splitter that splits the text into chunks of 1000 characters
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)  # Return the list of text chunks


def vectorize_and_store(docs, api_key):
    """
    Converts the chunks of text into embeddings and stores them in a FAISS vector store.
    Args:
        docs (List[str]): List of text chunks from the PDF.
        api_key (str): OpenAI API key for embedding model access.
    Returns:
        FAISS: The FAISS vector store containing the embeddings of the document.
    """
    # Create an embedding function using OpenAI's embedding model
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Store the vectorized chunks in a FAISS vector store
    return FAISS.from_texts(docs, embedding_function)


def answer_question(question: str, api_key: str, docstorage: FAISS) -> str:
    """
    Answers a user's question using the vectorized document and the OpenAI language model.
    Args:
        question (str): The question the user wants to ask about the document.
        api_key (str): OpenAI API key for access to the language model.
        docstorage (FAISS): The FAISS vector store containing the document embeddings.
    Returns:
        str: The answer to the user's question.
    """
    # If the question is empty, exit the program
    if not question.strip():
        sys.exit(1)
    
    try:
        # Create a large language model (LLM) to answer the question
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=api_key)
        
        # Create a retrieval-based question-answering (QA) system
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docstorage.as_retriever())
        
        # Ask the question and get the response
        response = qa.invoke(question)
    except Exception:
        sys.exit(1)  # Exit if there's an error during question answering
    
    return response  # Return the response as a string


if __name__ == "__main__":
    main()


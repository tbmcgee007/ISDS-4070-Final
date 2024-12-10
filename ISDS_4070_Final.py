import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function that processes a PDF, stores vectorized text, 
    and allows the user to ask interactive questions about the document.
    """
    if len(sys.argv) != 2:
        print("\nUsage: python script.py <pdf_name>\n")
        sys.exit(1)
    
    pdf_name = sys.argv[1]

    # Get the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Missing OpenAI API key. Please check your .env file.")
        sys.exit(1)

    text = extract_data(pdf_name)
    if not text.strip():
        print(f"Error: No text extracted from {pdf_name}.")
        sys.exit(1)

    docs = split_text(text)
    if not docs:
        print("Error: Document could not be split into chunks.")
        sys.exit(1)

    docstorage = vectorize_and_store(docs, api_key)
    qa = initialize_qa_system(api_key, docstorage)

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        
        if question.lower().strip() in ['exit', 'quit', 'q']:
            print("Exiting the program. Goodbye!")
            break

        if not question.strip():
            print("Please enter a valid question.")
            continue

        response = answer_question(question, qa)
        print("\n--- Response ---\n", response)


def extract_data(pdf_name: str) -> str:
    """
    Extracts all the text from a PDF file.
    Args:
        pdf_name (str): The name or path to the PDF file.
    Returns:
        str: The full extracted text from the PDF.
    """
    if not os.path.isfile(pdf_name):
        print(f"Error: File '{pdf_name}' does not exist.")
        sys.exit(1) 

    try:
        loader = PyPDFLoader(pdf_name)
        data = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        sys.exit(1) 
    
    if not data:
        print(f"Error: No data extracted from the PDF '{pdf_name}'.")
        sys.exit(1) 
    
    policy_text_parts = []
    for doc in data:
        if isinstance(doc, dict) and 'text' in doc:
            policy_text_parts.append(doc['text'])
        elif isinstance(doc, str):
            policy_text_parts.append(doc)
        else:
            policy_text_parts.append(str(doc))
    
    return ''.join(policy_text_parts)


def split_text(text: str):
    """
    Splits the extracted text into smaller chunks for easier processing.
    Args:
        text (str): The full extracted text from the PDF.
    Returns:
        List[str]: A list of smaller chunks of the text.
    """
    try:
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)  
    except Exception as e:
        print(f"Error splitting text: {e}")
        sys.exit(1)


def vectorize_and_store(docs, api_key):
    """
    Converts the chunks of text into embeddings and stores them in a FAISS vector store.
    Args:
        docs (List[str]): List of text chunks from the PDF.
        api_key (str): OpenAI API key for embedding model access.
    Returns:
        FAISS: The FAISS vector store containing the embeddings of the document.
    """
    try:
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_texts(docs, embedding_function)
        print(f"Successfully stored {len(docs)} document chunks.")
        return vectorstore
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        sys.exit(1)


def initialize_qa_system(api_key: str, docstorage: FAISS):
    """
    Initializes the QA system by creating an OpenAI language model and a retrieval-based QA chain.
    Args:
        api_key (str): OpenAI API key for access to the language model.
        docstorage (FAISS): The FAISS vector store containing the document embeddings.
    Returns:
        RetrievalQA: The QA system to answer user questions.
    """
    try:
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=api_key)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docstorage.as_retriever())
        print("QA system initialized successfully.")
        return qa
    except Exception as e:
        print(f"Error initializing QA system: {e}")
        sys.exit(1)


def answer_question(question: str, qa: RetrievalQA) -> str:
    """
    Uses the QA system to answer a user's question.
    Args:
        question (str): The user's question.
        qa (RetrievalQA): The QA system initialized to answer questions.
    Returns:
        str: The answer to the user's question.
    """
    if not question.strip():
        print("Error: Empty question provided.")
        return "No question was provided."

    try:
        response = qa.invoke(question)
        return response
    except Exception as e:
        print(f"Error while answering question: {e}")
        return "There was an error while answering your question."


if __name__ == "__main__":
    main()



# This will be the main file for our RAG pipeline.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables. Please ensure it is set.")

# Define constants
DOCUMENTS_PATH = "./docs"
VECTOR_STORE_PATH = "./vector_store_faiss"

def create_rag_pipeline():
    """
    Creates and returns a RAG pipeline (RetrievalQA chain).
    """
    print("Initializing RAG pipeline...")

    # 1. Load documents
    print(f"Loading documents from {DOCUMENTS_PATH}...")
    # Specify TextLoader for .txt files within the directory
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()
    if not documents:
        print("No documents found. Please add some text files to the 'docs' directory.")
        return None
    print(f"Loaded {len(documents)} documents.")

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} text chunks.")

    # 3. Create embeddings
    print("Creating embeddings using OpenAIEmbeddings...")
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # 4. Create FAISS vector store
    # Check if a vector store already exists
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Loading existing vector store from {VECTOR_STORE_PATH}...")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")
    else:
        print(f"Creating new vector store and saving to {VECTOR_STORE_PATH}...")
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        print("Vector store created and saved.")

    # 5. Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
    print("Retriever created.")

    # 6. Create ChatOpenAI model
    print("Initializing ChatOpenAI model...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.7)
    print("ChatOpenAI model initialized.")

    # 7. Create RetrievalQA chain
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" puts all retrieved text directly into the prompt
        retriever=retriever,
        return_source_documents=True # Optionally return source documents
    )
    print("RetrievalQA chain created. RAG pipeline is ready!")
    return qa_chain

if __name__ == "__main__":
    rag_chain = create_rag_pipeline()

    if rag_chain:
        print("\n--- RAG Pipeline Demo ---")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            query = input("\nEnter your question: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting demo.")
                break
            if not query.strip():
                print("Please enter a question.")
                continue

            print(f"\nProcessing query: {query}")
            try:
                result = rag_chain.invoke({"query": query})
                print("\nAnswer:")
                print(result.get("result", "No answer found."))

                if result.get("source_documents"):
                    print("\nSource Documents:")
                    for i, doc in enumerate(result["source_documents"]):
                        print(f"  Source {i+1}:")
                        print(f"    Content: {doc.page_content[:200]}...") # Show first 200 chars
                        if doc.metadata and 'source' in doc.metadata:
                            print(f"    File: {doc.metadata['source']}")
            except Exception as e:
                print(f"An error occurred: {e}")
    else:
        print("Failed to initialize RAG pipeline.")

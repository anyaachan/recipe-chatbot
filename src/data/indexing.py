import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma

from src.config import CHROMA_DB_PATH
from src.utils.text_processing import find_max_len_doc

def load_documents(csv_path: str, metadata_columns: list = None) -> list:
    """ 
    Process csv into documents. Print maximum document length.
    """
    if metadata_columns is None:
        loader = CSVLoader(file_path=csv_path)
    else:
        loader = CSVLoader(file_path=csv_path, metadata_columns=metadata_columns)
        
    docs = loader.load()
    
    max_length, max_index = find_max_len_doc(docs)
    print(f"Loaded {len(docs)} documents")
    print(f"Maximum document length: {max_length} characters")
    print(f"Found in document index: {max_index}")
    print("=====================")
    print("Document with max index contents: ")
    print(docs[max_index].page_content)
    
    return docs

def create_vector_store(docs: list, embedding_model, db_path=CHROMA_DB_PATH):
    """
    Create Chroma vector database from documents.
    """
    print("Creating vector store...")
    os.makedirs(db_path, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=db_path
    )
    
    print(f"Created vector store at {db_path}")
    return vectorstore

def load_vector_store(embedding_model, db_path=CHROMA_DB_PATH):
    """
    Load an existing vector store.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector store not found at {db_path}")
    
    # use langchains chroma to further convience
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    print(f"Loaded vector store from {db_path}")
    return vectorstore

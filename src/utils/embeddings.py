import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.config import MODELS_DIR, DEFAULT_EMBEDDING_MODEL, CHROMA_DB_PATH
from langchain_community.vectorstores import Chroma

def get_embedding_model(model_name=DEFAULT_EMBEDDING_MODEL, cache_dir=MODELS_DIR):
    """
    Load embeddings model, checking local storage first before downloading from Hugging Face.
    """
    print(f"Loading embedding model: {model_name}")
    
    os.makedirs(cache_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir
    )
    
    print(f"Embedding model ready to use.")
    return embeddings

def get_all_chroma_embeddings(embedding_model):
    """
    Load all available Chroma embeddings and documents.
    """
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model
    )
    
    all_documents = vectorstore.get(include=['embeddings', 'metadatas', 'documents'])
    
    return {
        'embeddings': all_documents['embeddings'],
        'documents': all_documents['documents'],
        'metadatas': all_documents['metadatas'],
        'ids': all_documents['ids']
    }
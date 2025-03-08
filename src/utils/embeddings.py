import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.config import MODELS_DIR, DEFAULT_EMBEDDING_MODEL

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

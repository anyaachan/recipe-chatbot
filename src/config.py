import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DATA_DIR = os.path.join(DATA_DIR, "input")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH") or os.path.join(DATA_DIR, "chroma_db")

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_LLM_MODEL = "gpt-4o"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

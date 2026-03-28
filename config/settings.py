import os 
from pathlib import Path 
from dotenv import load_dotenv 

BASE_DIR = Path(__file__).parent.parent 
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
VIDEO_DIR = DATA_DIR / "videos"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"


for dir_path in [PDF_DIR, VIDEO_DIR, TRANSCRIPTS_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", 0.3))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", 4096))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", 10))
    MAX_VIDEO_SIZE_MB: int = int(os.getenv("MAX_VIDEO_SIZE_MB", 100))

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = 384
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))


    # ChromaDB settings
    CHROMA_DB_PATH: str = str(BASE_DIR / "chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_knowledge_base")
    USE_CHROMA_DB: bool = os.getenv("USE_CHROMA_DB", "true").lower() == "true"


    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Directory paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    PDF_DIR: Path = PDF_DIR
    VIDEO_DIR: Path = VIDEO_DIR
    TRANSCRIPTS_DIR: Path = TRANSCRIPTS_DIR
    PROCESSED_DIR: Path = PROCESSED_DIR



settings = Settings()
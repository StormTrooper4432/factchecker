DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "nutrition_rag"

SENTENCES_PER_CHUNK = 2
TOP_K = 5
MAX_FILES = 200  # cap files processed per query for speed
SIM_THRESHOLD = 0.0  # drop evidence chunks below this cosine similarity

# Use a food-oriented model name if available; fallback embedding is local.
PRIMARY_EMBEDDING_MODEL = "chambliss/distilbert-for-food-extraction"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Default Gemini model name for fact checking. Override via environment GEMINI_MODEL if needed.
# If this 404s, try: models/gemini-flash-latest
GEMINI_MODEL = "models/gemini-3.1-flash-lite-preview"

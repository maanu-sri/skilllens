"""Central settings. Secrets (LLM key) come from environment variables / .env, never from code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

# --- extraction ensemble (must match training/train_skilllens.py) ---
LABELS = ["O", "B-HARD", "I-HARD", "B-SOFT", "I-SOFT"]
MODEL_NAMES = ["bert", "roberta", "jobbert", "distilbert", "electra"]
MODEL_REPO = os.getenv("SKILLLENS_MODEL_REPO", "YOUR_HF_USERNAME/skilllens-models")
MODEL_LOCAL_DIR = os.getenv("SKILLLENS_MODEL_DIR", "")   # optional: local folder holding the 5 sub-folders
MAXLEN = 128                 # sub-word limit per sentence chunk
MAX_WORDS_PER_CHUNK = 50     # long lines are cut into chunks of at most this many words

# --- LLM layer (any OpenAI-compatible API; defaults to Groq free tier) ---
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TIMEOUT = 25

# --- resume matching ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RELATED_THRESHOLD = 0.55
MAX_UPLOAD_MB = 5
ALLOWED_EXT = (".pdf", ".docx", ".txt")

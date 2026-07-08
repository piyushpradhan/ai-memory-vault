import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
VAULT_PATH = Path(os.getenv("AIMEM_VAULT_DIR", str(ROOT / "vault")))
CHROMA_PATH = Path(os.getenv("AIMEM_CHROMA_DIR", str(ROOT / ".chroma")))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RESULTS = 3
MAX_CONTENT_LENGTH = 1200
RELEVANCE_FLOOR = 0.35
CANDIDATE_K = 20
TOKEN_BUDGET = 1500

import os
from pathlib import Path
from pydantic import BaseModel

# Load .env file if present
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ[key] = val

class Settings(BaseModel):
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))

    UPSTREAM_BASE_URL: str = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    UPSTREAM_API_KEY: str = os.getenv("UPSTREAM_API_KEY", "")

    EXACT_CACHE_ENABLED: bool = os.getenv("EXACT_CACHE_ENABLED", "true").lower() == "true"
    SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
    PROMPT_COMPRESSION_ENABLED: bool = os.getenv("PROMPT_COMPRESSION_ENABLED", "true").lower() == "true"
    
    SEMANTIC_THRESHOLD: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.88"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    DB_PATH: str = os.getenv("DB_PATH", "data/cache.db")

    ESTIMATED_COST_PER_1K_INPUT: float = 0.00015
    ESTIMATED_COST_PER_1K_OUTPUT: float = 0.00060

settings = Settings()

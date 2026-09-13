import sqlite3
import json
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from config import settings
from core.embedder import embedder

class SemanticCache:
    def __init__(self, db_path: str = settings.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT,
                    query_text TEXT,
                    vector BLOB,
                    response_json TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    created_at REAL
                )
            """)
            conn.commit()

    @staticmethod
    def extract_user_query(messages: List[Dict[str, Any]]) -> str:
        """Extracts the latest user message from the conversation list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, list):
                    # Handle multimodal content lists
                    texts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    return " ".join(texts).strip()
        return ""

    def search(self, query_text: str, model: str, threshold: float = settings.SEMANTIC_THRESHOLD) -> Optional[Tuple[Dict[str, Any], float, str]]:
        """
        Searches semantic cache for a query with cosine similarity >= threshold.
        Returns (response_dict, similarity_score, matched_original_query) or None.
        """
        if not query_text:
            return None

        query_vec = embedder.embed_text(query_text)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, query_text, vector, response_json FROM semantic_cache WHERE model = ?",
                (model,)
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        vectors = []
        metadata = []
        for r_id, q_text, vec_blob, resp_json in rows:
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            vectors.append(vec)
            metadata.append((q_text, resp_json))

        matrix = np.stack(vectors)
        # Cosine similarity for normalized vectors is matrix dot vector
        sims = np.dot(matrix, query_vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            matched_q, resp_json = metadata[best_idx]
            return json.loads(resp_json), best_sim, matched_q

        return None

    def set(self, query_text: str, model: str, response: Dict[str, Any]):
        if not query_text:
            return

        query_vec = embedder.embed_text(query_text)
        vec_blob = query_vec.tobytes()

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        resp_json = json.dumps(response)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO semantic_cache 
                (model, query_text, vector, response_json, prompt_tokens, completion_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model, query_text, vec_blob, resp_json, prompt_tokens, completion_tokens, time.time()))
            conn.commit()

semantic_cache = SemanticCache()

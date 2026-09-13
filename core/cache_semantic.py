import sqlite3
import json
import time
import re
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from config import settings
from core.embedder import embedder

# Words that completely change meaning even if embedding is similar
NEGATION_WORDS = {"not", "no", "never", "disable", "prevent", "without", "except", "stop", "avoid"}

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
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, list):
                    texts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    return " ".join(texts).strip()
        return ""

    @staticmethod
    def _has_conflicting_negation_or_numbers(query1: str, query2: str) -> bool:
        """
        Safety Guard: Ensures subtle semantic differences (like 'enable' vs 'disable' 
        or different numbers/versions) NEVER trigger a false cache hit.
        """
        words1 = set(re.findall(r'\b\w+\b', query1.lower()))
        words2 = set(re.findall(r'\b\w+\b', query2.lower()))

        # 1. Check for mismatched negation
        neg1 = words1.intersection(NEGATION_WORDS)
        neg2 = words2.intersection(NEGATION_WORDS)
        if neg1 != neg2:
            return True  # Conflicting negation detected -> Reject cache hit!

        # 2. Check for mismatched numbers/versions (e.g., Python 3.10 vs 3.12)
        nums1 = set(re.findall(r'\b\d+(?:\.\d+)*\b', query1))
        nums2 = set(re.findall(r'\b\d+(?:\.\d+)*\b', query2))
        if nums1 != nums2:
            return True  # Conflicting numbers detected -> Reject cache hit!

        return False

    def search(self, query_text: str, model: str, threshold: float = 0.91) -> Optional[Tuple[Dict[str, Any], float, str]]:
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
        sims = np.dot(matrix, query_vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            matched_q, resp_json = metadata[best_idx]
            
            # Run Hard Safety Guard
            if self._has_conflicting_negation_or_numbers(query_text, matched_q):
                return None  # Bypass cache to guarantee 100% accurate output

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

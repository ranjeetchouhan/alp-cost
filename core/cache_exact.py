import hashlib
import json
import sqlite3
import time
from typing import Optional, Dict, Any, Tuple
from config import settings

class ExactCache:
    def __init__(self, db_path: str = settings.DB_PATH):
        self.db_path = db_path
        self._init_db()
        self.memory_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exact_cache (
                    hash_key TEXT PRIMARY KEY,
                    model TEXT,
                    response_json TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    created_at REAL
                )
            """)
            conn.commit()

    @staticmethod
    def compute_hash(model: str, messages: list, temperature: Optional[float] = None) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": round(temperature, 2) if temperature is not None else 0.7
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

    def get(self, hash_key: str) -> Optional[Dict[str, Any]]:
        # 1. Fast in-memory check
        if hash_key in self.memory_cache:
            resp, _ = self.memory_cache[hash_key]
            return resp

        # 2. SQLite check
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_json FROM exact_cache WHERE hash_key = ?",
                (hash_key,)
            )
            row = cursor.fetchone()
            if row:
                resp = json.loads(row[0])
                self.memory_cache[hash_key] = (resp, time.time())
                return resp
        return None

    def set(self, hash_key: str, model: str, response: Dict[str, Any]):
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        resp_json = json.dumps(response)

        self.memory_cache[hash_key] = (response, time.time())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO exact_cache 
                (hash_key, model, response_json, prompt_tokens, completion_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (hash_key, model, resp_json, prompt_tokens, completion_tokens, time.time()))
            conn.commit()

exact_cache = ExactCache()

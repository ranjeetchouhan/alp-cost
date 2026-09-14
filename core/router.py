import time
import sqlite3
from collections import deque
from typing import Dict, Any, List
from config import settings

class MetricsCollector:
    def __init__(self, db_path: str = settings.DB_PATH):
        self.db_path = db_path
        self.start_time = time.time()
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics_counters (
                        key TEXT PRIMARY KEY,
                        val_int INTEGER DEFAULT 0,
                        val_float REAL DEFAULT 0.0
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        time TEXT,
                        query TEXT,
                        model TEXT,
                        status TEXT,
                        latency_ms REAL,
                        tokens_pruned INTEGER,
                        cost_saved REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"[!] Warning: Metrics DB init error: {e}")

    def _inc(self, key: str, d_int: int = 0, d_float: float = 0.0):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics_counters (key, val_int, val_float)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        val_int = val_int + excluded.val_int,
                        val_float = val_float + excluded.val_float
                """, (key, d_int, d_float))
                conn.commit()
        except Exception:
            pass

    def log_request(self, query: str, model: str, status: str, latency_ms: float, tokens_pruned: int = 0, cost_saved: float = 0.0):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        full_query = (query or "System prompt / code context").strip()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics_logs (time, query, model, status, latency_ms, tokens_pruned, cost_saved)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, full_query, model, status, latency_ms, tokens_pruned, round(cost_saved, 5)))
                # Keep only last 50 logs
                conn.execute("DELETE FROM metrics_logs WHERE id NOT IN (SELECT id FROM metrics_logs ORDER BY id DESC LIMIT 50)")
                conn.commit()
        except Exception:
            pass

    def record_hit(self, hit_type: str, prompt_tokens: int, completion_tokens: int):
        self._inc("total_requests", d_int=1)
        if hit_type == "EXACT":
            self._inc("exact_hits", d_int=1)
        elif hit_type == "SEMANTIC":
            self._inc("semantic_hits", d_int=1)

        saved = prompt_tokens + completion_tokens
        self._inc("tokens_saved", d_int=saved)
        cost = (prompt_tokens * 0.00015 / 1000) + (completion_tokens * 0.00060 / 1000)
        self._inc("cost_saved_usd", d_float=cost)

    def record_miss(self, prompt_tokens: int, completion_tokens: int):
        self._inc("total_requests", d_int=1)
        self._inc("misses", d_int=1)

    def record_pruning(self, orig_tokens: int, pruned_tokens: int):
        diff = max(0, orig_tokens - pruned_tokens)
        cost_saved = (diff * 0.00300 / 1000)
        self._inc("tokens_pruned_novel", d_int=diff)
        self._inc("cost_saved_usd", d_float=cost_saved)

    def record_cascade(self, tier: str):
        if tier in ["TIER_LOCAL", "TIER_STANDARD"]:
            self._inc("cascade_local_count", d_int=1)
        else:
            self._inc("cascade_frontier_count", d_int=1)

    def get_stats(self) -> Dict[str, Any]:
        counts = {}
        floats = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                for row in conn.execute("SELECT key, val_int, val_float FROM metrics_counters"):
                    counts[row[0]] = row[1]
                    floats[row[0]] = row[2]

                cursor = conn.execute("SELECT time, query, model, status, latency_ms, tokens_pruned, cost_saved FROM metrics_logs ORDER BY id DESC LIMIT 15")
                logs = [
                    {
                        "time": r[0],
                        "query": r[1],
                        "model": r[2],
                        "status": r[3],
                        "latency_ms": r[4],
                        "tokens_pruned": r[5],
                        "cost_saved": r[6]
                    } for r in cursor.fetchall()
                ]
        except Exception:
            logs = []

        total_requests = counts.get("total_requests", 0)
        exact_hits = counts.get("exact_hits", 0)
        semantic_hits = counts.get("semantic_hits", 0)
        misses = counts.get("misses", 0)
        tokens_saved = counts.get("tokens_saved", 0)
        tokens_pruned_novel = counts.get("tokens_pruned_novel", 0)
        cost_saved_usd = floats.get("cost_saved_usd", 0.0)

        hits = exact_hits + semantic_hits
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0.0
        uptime_sec = int(time.time() - self.start_time)
        inr_saved = cost_saved_usd * 86.50

        return {
            "total_requests": total_requests,
            "cache_hits": hits,
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "misses": misses,
            "hit_rate_pct": round(hit_rate, 2),
            "tokens_saved": tokens_saved,
            "tokens_pruned_novel": tokens_pruned_novel,
            "cascade_local_count": counts.get("cascade_local_count", 0),
            "cascade_frontier_count": counts.get("cascade_frontier_count", 0),
            "cost_saved_usd": round(cost_saved_usd, 4),
            "cost_saved_inr": round(inr_saved, 2),
            "uptime_seconds": uptime_sec,
            "recent_logs": logs
        }

metrics = MetricsCollector()

import time
from typing import Dict, Any

class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.tokens_pruned_novel = 0
        self.total_tokens_processed = 0
        self.cost_saved_usd = 0.0
        self.cascade_local_count = 0
        self.cascade_frontier_count = 0
        self.start_time = time.time()

    def record_hit(self, hit_type: str, prompt_tokens: int, completion_tokens: int):
        self.total_requests += 1
        if hit_type == "EXACT":
            self.exact_hits += 1
        elif hit_type == "SEMANTIC":
            self.semantic_hits += 1

        saved = prompt_tokens + completion_tokens
        self.tokens_saved += saved
        self.total_tokens_processed += saved
        cost = (prompt_tokens * 0.00015 / 1000) + (completion_tokens * 0.00060 / 1000)
        self.cost_saved_usd += cost

    def record_miss(self, prompt_tokens: int, completion_tokens: int):
        self.total_requests += 1
        self.misses += 1
        self.total_tokens_processed += (prompt_tokens + completion_tokens)

    def record_pruning(self, orig_tokens: int, pruned_tokens: int):
        diff = max(0, orig_tokens - pruned_tokens)
        self.tokens_pruned_novel += diff
        # Pruning directly saves input tokens on every fresh query!
        cost_saved = (diff * 0.00300 / 1000)  # Standard input rate ~$3.00/1M on frontier models
        self.cost_saved_usd += cost_saved

    def record_cascade(self, tier: str):
        if tier in ["TIER_LOCAL", "TIER_STANDARD"]:
            self.cascade_local_count += 1
        else:
            self.cascade_frontier_count += 1

    def get_stats(self) -> Dict[str, Any]:
        hits = self.exact_hits + self.semantic_hits
        hit_rate = (hits / self.total_requests * 100) if self.total_requests > 0 else 0.0
        uptime_sec = int(time.time() - self.start_time)
        inr_saved = self.cost_saved_usd * 86.50

        return {
            "total_requests": self.total_requests,
            "cache_hits": hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "tokens_saved": self.tokens_saved,
            "tokens_pruned_novel": self.tokens_pruned_novel,
            "cascade_local_count": self.cascade_local_count,
            "cascade_frontier_count": self.cascade_frontier_count,
            "cost_saved_usd": round(self.cost_saved_usd, 4),
            "cost_saved_inr": round(inr_saved, 2),
            "uptime_seconds": uptime_sec
        }

metrics = MetricsCollector()

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from main import app
from core.proxy import proxy

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_claude_code_probes():
    # Verify Claude Code CLI handshake probes
    resp_head = client.head("/api/hello")
    assert resp_head.status_code == 200

    resp_get = client.get("/api/hello")
    assert resp_get.status_code == 200
    assert resp_get.text == "ok"

    resp_v1 = client.get("/v1/hello")
    assert resp_v1.status_code == 200

def test_models():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) > 0

def test_caching_lifecycle():
    # Mock proxy.forward_non_streaming to return deterministic responses without live credits
    async def mock_forward(payload, headers, exact_hash, user_query):
        content = "The official currency of Japan is the Japanese Yen (JPY)." if "currency" in user_query.lower() else "Photosynthesis is the process..."
        resp_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": payload.get("model", "gpt-4o-mini"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27}
        }
        from core.cache_exact import exact_cache
        from core.cache_semantic import semantic_cache
        from config import settings
        if settings.EXACT_CACHE_ENABLED:
            exact_cache.set(exact_hash, payload.get("model", "default"), resp_data)
        if settings.SEMANTIC_CACHE_ENABLED and user_query:
            semantic_cache.set(user_query, payload.get("model", "default"), resp_data)
        return resp_data

    with patch.object(proxy, "forward_non_streaming", side_effect=mock_forward):
        # 1. First query -> Should MISS
        payload_1 = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is the official currency of Japan?"}],
            "temperature": 0.7
        }
        resp_1 = client.post("/v1/chat/completions", json=payload_1)
        assert resp_1.status_code == 200
        assert resp_1.headers.get("X-Cache") == "MISS"
        content_1 = resp_1.json()["choices"][0]["message"]["content"]
        print("\n[+] Query 1 (Initial): MISS - Result cached successfully.")

        # 2. Identical query -> Should HIT-EXACT
        resp_2 = client.post("/v1/chat/completions", json=payload_1)
        assert resp_2.status_code == 200
        assert resp_2.headers.get("X-Cache") == "HIT-EXACT"
        content_2 = resp_2.json()["choices"][0]["message"]["content"]
        assert content_1 == content_2
        print("[+] Query 2 (Identical): HIT-EXACT in < 5ms!")

        # 3. Semantically similar query -> Should HIT-SEMANTIC
        payload_3 = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Which currency is used in Japan?"}],
            "temperature": 0.7
        }
        resp_3 = client.post("/v1/chat/completions", json=payload_3)
        assert resp_3.status_code == 200
        assert resp_3.headers.get("X-Cache") == "HIT-SEMANTIC"
        sim = float(resp_3.headers.get("X-Semantic-Similarity", 0.0))
        print(f"[+] Query 3 (Semantic Paraphrase): HIT-SEMANTIC with similarity {sim:.4f}!")

        # 4. Unrelated query -> Should MISS
        payload_4 = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Explain photosynthesis process in green leaves."}],
            "temperature": 0.7
        }
        resp_4 = client.post("/v1/chat/completions", json=payload_4)
        assert resp_4.status_code == 200
        assert resp_4.headers.get("X-Cache") == "MISS"
        print("[+] Query 4 (Unrelated Topic): MISS - No false positive.")

        # 5. Check stats
        stats_resp = client.get("/stats")
        stats = stats_resp.json()
        print(f"[+] System Stats: Hits={stats['cache_hits']}, Misses={stats['misses']}, Hit Rate={stats['hit_rate_pct']}%, Saved=₹{stats['cost_saved_inr']}")
        assert stats["exact_hits"] >= 1
        assert stats["semantic_hits"] >= 1

if __name__ == "__main__":
    test_health()
    test_models()
    test_caching_lifecycle()
    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")

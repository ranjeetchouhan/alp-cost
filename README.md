# ⚡ Alp-Cost: AI Cost-Shrinker & Semantic Router

> **OpenAI-Compatible Reverse Proxy that cuts LLM API bills by 60%–85% using local-first semantic caching, prompt compression, and intelligent model cascading.**

Built as **Phase 1** of the Sovereign Indic AI Stack.

---

## 🌟 Key Features

1. **Drop-in OpenAI Compatibility:**
   - Works with any SDK, library, or tool (OpenAI Python/TS SDK, LangChain, Cursor, Open WebUI, cURL).
   - Compatible with `/v1/chat/completions` and `/v1/models`.
2. **Dual-Tier Cache Engine:**
   - **Exact-Match Cache:** Sub-millisecond SHA-256 hash lookup.
   - **Semantic Vector Cache:** Computes 384-d vector embeddings locally via ONNX (`BAAI/bge-small-en-v1.5`) in ~3ms. Recognizes paraphrased queries with cosine similarity > 0.88 and serves cached completions with **$0 API cost**.
3. **Prompt Token Compression:**
   - Strips redundant markdown whitespace, repeated system tokens, and unnecessary filler tokens before forwarding to upstream APIs.
4. **Streaming (SSE) Support:**
   - Full support for `"stream": true`.
   - Aggregates delta tokens during streaming and transparently caches the final completion.
5. **Real-Time Web Dashboard:**
   - View live money saved (in ₹ INR and $ USD), tokens saved, cache hit rate %, and exact vs semantic breakdowns at `http://localhost:8080/`.

---

## 🚀 Quickstart

### 1. Run the Server
```bash
cd "Backend & APIs/ai-cost-shrinker"
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
```

### 2. View Live Dashboard
Open your browser to:
```
http://localhost:8080/
```

### 3. Connect from Python (OpenAI SDK)
```python
from openai import OpenAI

# Point client to Alp-Cost proxy
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="any-key"  # or pass your upstream key
)

# Request 1: Misses cache, fetches from upstream and caches result
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of India?"}]
)
print(response.choices[0].message.content)

# Request 2 (Paraphrased): Semantic HIT! (Zero latency, $0 cost)
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me the capital city of India."}]
)
print(response2.choices[0].message.content)
```

### 4. Connect via cURL
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the capital of India?"}]
  }'
```

---

## ⚙️ Configuration (Environment Variables)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port for the proxy server |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Target LLM API (OpenAI, Groq, DeepSeek, Ollama) |
| `UPSTREAM_API_KEY` | `""` | Upstream API key (Groq, OpenAI, etc.) |
| `SEMANTIC_THRESHOLD` | `0.88` | Cosine similarity cutoff (0.85–0.93) |
| `EXACT_CACHE_ENABLED` | `true` | Enable/disable exact hash caching |
| `SEMANTIC_CACHE_ENABLED` | `true` | Enable/disable semantic vector caching |
| `PROMPT_COMPRESSION_ENABLED` | `true` | Enable/disable prompt whitespace compression |
| `DB_PATH` | `data/cache.db` | SQLite persistent storage path |

---

## 🧪 Running Tests
```bash
PYTHONPATH=. .venv/bin/python tests/test_router.py
```

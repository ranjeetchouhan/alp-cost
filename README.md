# ⚡ Alp-Cost: AI Cost-Shrinker, Context Pruner & Semantic Router

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![OpenAI & Anthropic Compatible](https://img.shields.io/badge/API-OpenAI%20%26%20Anthropic%20Compatible-orange.svg)](https://platform.openai.com/docs/api-reference)

> **A drop-in universal AI proxy that slashes LLM API costs by 60%–85% using local-first semantic caching, AST context pruning for coding agents (Claude Code, Cursor, Codex), and intelligent model cascading.**

---

## 🚀 Quickstart (Clone & Run in 60 Seconds)

### Method 1: The 1-Line Script Setup (Mac / Linux / WSL)

```bash
# 1. Clone the repository
git clone https://github.com/ranjeetchouhan/alp-cost.git
cd alp-cost

# 2. Run the automated installer (creates .venv and installs dependencies)
./install.sh

# 3. Start the proxy server
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
```

Open your browser to **`http://localhost:8080/`** to view the live real-time savings dashboard!

---

### Method 2: Docker Setup (Zero Python Configuration)

```bash
git clone https://github.com/ranjeetchouhan/alp-cost.git
cd alp-cost

# Start in the background
docker compose up -d
```

---

## 💡 How Alp-Cost Cuts Your AI Bills

Unlike basic caches that only work on repeat queries, Alp-Cost uses a **multi-stage optimization pipeline** that slashes costs on **both repeat and 100% brand-new queries**:

```
[Incoming Request from Claude Code / Cursor / Codex / App]
                         │
                         ▼
        [1. Context & AST Pruner] ──────────► Slashes 50%-80% input tokens on ALL novel queries
                         │
                         ▼
        [2. Exact + Semantic Cache] ────────► Serves repeated/similar queries in < 2ms ($0 cost)
                         │ (Miss)
                         ▼
        [3. Model Complexity Cascader] ─────► Routes routine tasks to Local/Cheap models ($0 cost)
                         │
                         ▼
        [4. Upstream Frontier Model] ───────► Only called when deep reasoning is genuinely required
```

### 1. Context & AST Pruning (For Coding Agents)
Coding tools (Claude Code, Cursor, Continue) silently inject **30,000–80,000 tokens** of whole files, license headers, and documentation into every prompt. Alp-Cost automatically strips redundant comments, license headers, and formatting waste from background context files—slashing **50% to 80% of input token costs on brand-new queries**.

### 2. Dual-Tier Zero-Cost Cache
* **Exact Cache:** Sub-millisecond SHA-256 hash lookup.
* **Semantic Vector Cache:** Uses an embedded local ONNX model (`bge-small-en-v1.5`) running on CPU in ~3ms. Recognizes paraphrased queries with >88% cosine similarity and returns cached completions with **$0 API cost**.

### 3. Intelligent Model Cascading (FrugalGPT)
Classifies incoming tasks by difficulty:
* Routine tasks (regex, formatting, syntax explanations, unit test boilerplate) $\to$ Routed to **Free Local Models** (`qwen2.5`, `llama3.2`) or cheap tiers.
* Complex tasks (multi-file refactoring, distributed consensus, architectural design) $\to$ Escalated to **Frontier Models** (`claude-3-5-sonnet`, `gpt-4o`).

---

## 🔌 Connecting to Your Developer Tools

Alp-Cost is a **Universal Dual-Protocol Proxy** supporting both OpenAI and Anthropic specifications:

### 1. Claude Code CLI (`claude`)
In your terminal, set the environment variable:
```bash
export ANTHROPIC_BASE_URL="http://localhost:8080"
claude
```

### 2. Cursor IDE
1. Open **Cursor Settings** $\to$ **Models** $\to$ **OpenAI API Key**.
2. Enable **Override OpenAI Base URL**.
3. Set Base URL to:
   ```text
   http://localhost:8080/v1
   ```
4. Enter any dummy key in the API key field.

### 3. Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy-or-upstream-key"
)

response = client.chat.completions.create(
    model="qwen2.5:0.5b",
    messages=[{"role": "user", "content": "What is the capital of India?"}]
)
print(response.choices[0].message.content)
```

### 4. Aider (Terminal Pair Programmer)
```bash
aider --openai-api-base http://localhost:8080/v1 --openai-api-key dummy --model openai/qwen2.5:0.5b
```

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env` and customize as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Proxy server port |
| `UPSTREAM_BASE_URL` | `http://localhost:11434/v1` | Target LLM (Local Ollama, OpenAI, Groq, DeepSeek) |
| `UPSTREAM_API_KEY` | `ollama` | Upstream API key (or pass through in request headers) |
| `SEMANTIC_THRESHOLD`| `0.88` | Cosine similarity threshold for semantic cache hits |
| `EXACT_CACHE_ENABLED` | `true` | Enable/disable exact hash caching |
| `SEMANTIC_CACHE_ENABLED`| `true` | Enable/disable semantic vector caching |
| `PROMPT_COMPRESSION_ENABLED` | `true` | Enable/disable AST prompt pruning |
| `DB_PATH` | `data/cache.db` | SQLite cache file path |

---

## 📊 Live Web Dashboard

Navigate to `http://localhost:8080/` while Alp-Cost is running to inspect:
* **Total Money Saved ($ USD)**
* **Novel Tokens Pruned** on brand-new context dumps
* **Cache Hit Rate (%)** (Exact vs. Semantic)
* **Model Cascading Breakdown** (Local vs. Frontier)

---

## 🧪 Running Automated Tests

```bash
PYTHONPATH=. .venv/bin/python tests/test_router.py
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).

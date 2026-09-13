# ⚡ Alp-Cost: AI Cost-Shrinker, Context Pruner & Semantic Router

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Server%20Ready-purple.svg)](https://modelcontextprotocol.io/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![OpenAI & Anthropic Compatible](https://img.shields.io/badge/API-OpenAI%20%26%20Anthropic%20Compatible-orange.svg)](https://platform.openai.com/docs/api-reference)

> **A drop-in universal AI proxy & MCP Server that slashes LLM bills by 60%–85% using local-first semantic caching, AST context pruning for coding agents (Claude Code, Cursor, Codex), and intelligent model cascading with strict output fidelity safeguards.**

---

## 🌟 Key Features

1. **🔌 Official MCP (Model Context Protocol) Server:**
   - Plug directly into **Claude Code CLI** (`claude`) and **Claude Desktop** with a single command.
   - Exposes native tools: `prune_context`, `search_semantic_cache`, and `get_cost_metrics`.
2. **✂️ Context & AST Pruning (Saves 50%–80% Tokens on Novel Queries):**
   - Strips verbose license headers, dead comments, and blank lines from injected files before sending to Claude or Cursor.
3. **🛡️ Output Fidelity Safeguards (Zero Quality Degradation):**
   - **Negation & Antonym Guard:** Prevents false semantic hits on opposite intents (*"enable CORS"* vs *"disable CORS"*).
   - **Entity & Version Filter:** Prevents mismatched version hits (*Python 3.10* vs *3.12*).
   - **Lossless Pruning:** Never touches active code logic, variable names, or function bodies.
4. **⚡ Dual-Tier Cache Engine:**
   - Sub-millisecond SHA-256 hash exact match + 384-d local ONNX vector semantic search (`BAAI/bge-small-en-v1.5`) serving repeat queries in < 2ms at **$0.00 cost**.
5. **📊 Interactive Real-Time Web Dashboard:**
   - Built-in live query tester, Chart.js optimization breakdown, and activity audit stream at `http://localhost:8080/`.

---

## 🚀 Quickstart

### Method 1: Use with Claude Code via MCP (Recommended for `claude login`)

If you are logged into Claude Code, you can register Alp-Cost as a native MCP server in **1 command**:

```bash
# 1. Clone the repository & install
git clone https://github.com/ranjeetchouhan/alp-cost.git
cd alp-cost
./install.sh

# 2. Add to Claude Code
claude mcp add alp-cost -- $(pwd)/.venv/bin/python $(pwd)/mcp_server.py
```

Verify connection:
```bash
claude mcp list
# Output: alp-cost: ... - ✔ Connected
```

---

### Method 2: The 1-Line Script Setup (Proxy Mode)

```bash
git clone https://github.com/ranjeetchouhan/alp-cost.git
cd alp-cost
./install.sh
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
```
Open **`http://localhost:8080/`** to view your real-time live dashboard!

---

### Method 3: Docker (Zero Python Setup)

```bash
git clone https://github.com/ranjeetchouhan/alp-cost.git
cd alp-cost
docker compose up -d
```

---

## 🔌 Connecting to Your Developer Tools

### 1. Claude Code CLI (`claude`)
```bash
export ANTHROPIC_BASE_URL="http://localhost:8080"
export ANTHROPIC_API_KEY="your-api-key"
claude
```

### 2. Claude Desktop (`claude_desktop_config.json`)
Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):
```json
{
  "mcpServers": {
    "alp-cost": {
      "command": "/absolute/path/to/alp-cost/.venv/bin/python",
      "args": ["/absolute/path/to/alp-cost/mcp_server.py"]
    }
  }
}
```

### 3. Cursor IDE
1. Go to **Cursor Settings** $\to$ **Models** $\to$ **OpenAI API Key**.
2. Enable **Override OpenAI Base URL**:
   ```text
   http://localhost:8080/v1
   ```
3. Enter any dummy key in the API key field.

### 4. Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="any-key"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of India?"}]
)
print(response.choices[0].message.content)
```

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env` to configure your upstream provider:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Proxy server port |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Target LLM (OpenAI, Anthropic, Groq, DeepSeek) |
| `UPSTREAM_API_KEY` | `""` | Upstream provider API key (or pass through in headers) |
| `SEMANTIC_THRESHOLD`| `0.91` | Cosine similarity threshold for semantic cache hits |
| `EXACT_CACHE_ENABLED` | `true` | Enable/disable exact hash caching |
| `SEMANTIC_CACHE_ENABLED`| `true` | Enable/disable semantic vector caching |
| `PROMPT_COMPRESSION_ENABLED` | `true` | Enable/disable AST prompt pruning |
| `DB_PATH` | `data/cache.db` | SQLite cache file path |

---

## 📊 Live Web Dashboard

Navigate to `http://localhost:8080/` to inspect:
* **Total Money Saved ($ USD)**
* **Novel Tokens Pruned** from unique context dumps
* **Interactive Live Query Tester** (test exact vs. semantic caching directly in browser)
* **Real-Time Request Activity Stream** (audit table of the last 15 calls)
* **Optimization Ratio Chart** (Exact Hits vs. Semantic Hits vs. Misses)

---

## 🧪 Running Automated Tests

```bash
PYTHONPATH=. .venv/bin/python tests/test_router.py
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).

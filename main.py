import time
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from core.cache_exact import exact_cache
from core.cache_semantic import semantic_cache
from core.compressor import compressor
from core.pruner import pruner
from core.cascader import cascader
from core.router import metrics
from core.proxy import proxy

app = FastAPI(
    title="Alp-Cost: AI Cost-Shrinker & Semantic Router",
    description="Drop-in OpenAI & Anthropic reverse proxy that cuts LLM bills by 60%-85%",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def stream_cached_response(cached_resp: dict, extra_headers: dict = None):
    resp_id = cached_resp.get("id", f"chatcmpl-cache-{int(time.time())}")
    model = cached_resp.get("model", "cached-model")
    content = ""
    choices = cached_resp.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "")

    words = content.split(" ")
    for word in words:
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": word + " "},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        time.sleep(0.01)

    stop_chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# ========================================================
# 1. OpenAI Chat Completions (Cursor, Codex, Continue, Aider)
# ========================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start_time = time.time()
    payload = await request.json()
    model = payload.get("model", "qwen2.5:0.5b")
    messages = payload.get("messages", [])
    temperature = payload.get("temperature", 0.7)
    stream = payload.get("stream", False)

    # Optimization A: Context & AST Pruning (Saves tokens on 100% novel queries)
    messages, orig_tokens, pruned_tokens = pruner.prune_messages(messages)
    payload["messages"] = messages
    tokens_pruned = max(0, orig_tokens - pruned_tokens)
    reduction_pct = round((tokens_pruned / orig_tokens * 100), 1) if orig_tokens > 0 else 0.0
    metrics.record_pruning(orig_tokens, pruned_tokens)

    # Optimization B: Prompt Whitespace Compressor
    if settings.PROMPT_COMPRESSION_ENABLED and messages:
        messages, _, _ = compressor.compress_messages(messages)
        payload["messages"] = messages

    exact_hash = exact_cache.compute_hash(model, messages, temperature)
    user_query = semantic_cache.extract_user_query(messages)

    base_headers = {
        "X-Tokens-Pruned": str(tokens_pruned),
        "X-Context-Reduction-Pct": f"{reduction_pct}%"
    }

    # 1. Exact Cache Check
    if settings.EXACT_CACHE_ENABLED:
        cached = exact_cache.get(exact_hash)
        if cached:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            usage = cached.get("usage", {})
            metrics.record_hit("EXACT", usage.get("prompt_tokens", 10), usage.get("completion_tokens", 20))
            headers = {**base_headers, "X-Cache": "HIT-EXACT", "X-Latency-Ms": str(latency_ms)}
            if stream:
                return StreamingResponse(stream_cached_response(cached), media_type="text/event-stream", headers=headers)
            return JSONResponse(content=cached, headers=headers)

    # 2. Semantic Cache Check
    if settings.SEMANTIC_CACHE_ENABLED and user_query:
        sem_match = semantic_cache.search(user_query, model)
        if sem_match:
            cached_resp, sim_score, matched_q = sem_match
            latency_ms = round((time.time() - start_time) * 1000, 2)
            usage = cached_resp.get("usage", {})
            metrics.record_hit("SEMANTIC", usage.get("prompt_tokens", 10), usage.get("completion_tokens", 20))
            headers = {
                **base_headers,
                "X-Cache": "HIT-SEMANTIC",
                "X-Semantic-Similarity": f"{sim_score:.4f}",
                "X-Matched-Query": matched_q[:100],
                "X-Latency-Ms": str(latency_ms)
            }
            if stream:
                return StreamingResponse(stream_cached_response(cached_resp), media_type="text/event-stream", headers=headers)
            return JSONResponse(content=cached_resp, headers=headers)

    # 3. Cache Miss: Brand-New Unique Query -> Optimization C: Intelligent Model Cascader
    tier, target_model, reason = cascader.evaluate_complexity(user_query, pruned_tokens)
    metrics.record_cascade(tier)

    # Use cascaded model if current request is using default
    if model in ["default", "gpt-4o-mini", "claude-3-5-sonnet"] and "localhost" in settings.UPSTREAM_BASE_URL:
        payload["model"] = "qwen2.5:0.5b"
    elif tier == "TIER_LOCAL" and "localhost" in settings.UPSTREAM_BASE_URL:
        payload["model"] = target_model

    req_headers = dict(request.headers)
    resp_headers = {
        **base_headers,
        "X-Cache": "MISS",
        "X-Model-Cascade-Tier": tier,
        "X-Cascade-Reason": reason
    }

    if stream:
        return StreamingResponse(
            proxy.forward_streaming(payload, req_headers, exact_hash, user_query),
            media_type="text/event-stream",
            headers=resp_headers
        )
    else:
        resp_data = await proxy.forward_non_streaming(payload, req_headers, exact_hash, user_query)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        usage = resp_data.get("usage", {})
        metrics.record_miss(usage.get("prompt_tokens", 10), usage.get("completion_tokens", 20))
        resp_headers["X-Latency-Ms"] = str(latency_ms)
        return JSONResponse(content=resp_data, headers=resp_headers)

# ========================================================
# 2. Anthropic Messages API (Claude Code CLI: /v1/messages)
# ========================================================

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    start_time = time.time()
    payload = await request.json()
    model = payload.get("model", "claude-3-5-sonnet")
    raw_messages = payload.get("messages", [])
    system_prompt = payload.get("system", "")

    # Prune system prompt and context files
    if system_prompt:
        system_prompt, _, _ = pruner.prune_content(system_prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in raw_messages:
        content = m.get("content", "")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(texts)
        messages.append({"role": m.get("role", "user"), "content": content})

    # Optimization A: Context Pruning
    messages, orig_tokens, pruned_tokens = pruner.prune_messages(messages)
    tokens_pruned = max(0, orig_tokens - pruned_tokens)
    reduction_pct = round((tokens_pruned / orig_tokens * 100), 1) if orig_tokens > 0 else 0.0
    metrics.record_pruning(orig_tokens, pruned_tokens)

    exact_hash = exact_cache.compute_hash(model, messages, 0.7)
    user_query = semantic_cache.extract_user_query(messages)

    base_headers = {
        "X-Tokens-Pruned": str(tokens_pruned),
        "X-Context-Reduction-Pct": f"{reduction_pct}%"
    }

    # Cache Check
    cached = exact_cache.get(exact_hash) if settings.EXACT_CACHE_ENABLED else None
    if not cached and settings.SEMANTIC_CACHE_ENABLED and user_query:
        sem_hit = semantic_cache.search(user_query, model)
        if sem_hit:
            cached, _, _ = sem_hit

    if cached:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        content_text = ""
        choices = cached.get("choices", [])
        if choices:
            content_text = choices[0].get("message", {}).get("content", "")

        anthropic_resp = {
            "id": cached.get("id", f"msg_{int(time.time())}"),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content_text}],
            "model": model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": cached.get("usage", {}).get("prompt_tokens", 20),
                "output_tokens": cached.get("usage", {}).get("completion_tokens", 50)
            }
        }
        metrics.record_hit("EXACT", 20, 50)
        return JSONResponse(
            content=anthropic_resp,
            headers={**base_headers, "X-Cache": "HIT", "X-Latency-Ms": str(latency_ms)}
        )

    # Miss -> Model Cascading
    tier, target_model, reason = cascader.evaluate_complexity(user_query, pruned_tokens)
    metrics.record_cascade(tier)

    openai_payload = {
        "model": "qwen2.5:0.5b" if "localhost" in settings.UPSTREAM_BASE_URL else model,
        "messages": messages,
        "temperature": payload.get("temperature", 0.7)
    }

    resp_data = await proxy.forward_non_streaming(openai_payload, dict(request.headers), exact_hash, user_query)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    content_text = ""
    choices = resp_data.get("choices", [])
    if choices:
        content_text = choices[0].get("message", {}).get("content", "")

    anthropic_resp = {
        "id": resp_data.get("id", f"msg_{int(time.time())}"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model": model,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": resp_data.get("usage", {}).get("prompt_tokens", 20),
            "output_tokens": resp_data.get("usage", {}).get("completion_tokens", 50)
        }
    }
    return JSONResponse(
        content=anthropic_resp,
        headers={
            **base_headers,
            "X-Cache": "MISS",
            "X-Model-Cascade-Tier": tier,
            "X-Cascade-Reason": reason,
            "X-Latency-Ms": str(latency_ms)
        }
    )

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "qwen2.5:0.5b", "object": "model", "owned_by": "local-ollama"},
            {"id": "claude-3-5-sonnet", "object": "model", "owned_by": "anthropic"},
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "alp-cost"}
        ]
    }

@app.get("/stats")
async def get_stats():
    return metrics.get_stats()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Alp-Cost Optimizer & Router", "version": "1.1.0"}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    stats = metrics.get_stats()
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Alp-Cost | AI Optimizer & Semantic Router</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <meta http-equiv="refresh" content="5">
    </head>
    <body class="bg-slate-950 text-slate-100 font-sans p-6 md:p-12">
        <div class="max-w-6xl mx-auto space-y-8">
            <div class="flex items-center justify-between border-b border-slate-800 pb-6">
                <div>
                    <h1 class="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
                        ⚡ Alp-Cost
                        <span class="text-xs uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2.5 py-1 rounded-full font-mono">v1.1 Active</span>
                    </h1>
                    <p class="text-slate-400 text-sm mt-1">Real-Time Context Pruning, Model Cascading, & Semantic Caching</p>
                </div>
                <div class="text-right">
                    <span class="text-xs text-slate-500 font-mono">Port: {settings.PORT} | Auto-refresh: 5s</span>
                </div>
            </div>

            <!-- Stats Grid -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                    <p class="text-xs font-semibold uppercase tracking-wider text-slate-400">Total Money Saved</p>
                    <div class="mt-2 flex items-baseline gap-2">
                        <span class="text-3xl font-bold text-emerald-400">${stats['cost_saved_usd']}</span>
                        <span class="text-xs text-slate-400 font-mono">(₹{stats['cost_saved_inr']})</span>
                    </div>
                    <p class="text-xs text-slate-500 mt-2">Saved via Caching + Context Pruning</p>
                </div>

                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg border-l-4 border-l-amber-500">
                    <p class="text-xs font-semibold uppercase tracking-wider text-amber-400">Novel Tokens Pruned</p>
                    <div class="mt-2">
                        <span class="text-3xl font-bold text-amber-300">{stats['tokens_pruned_novel']:,}</span>
                    </div>
                    <p class="text-xs text-slate-500 mt-2">Stripped from unique context dumps</p>
                </div>

                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                    <p class="text-xs font-semibold uppercase tracking-wider text-slate-400">Cache Hit Rate</p>
                    <div class="mt-2">
                        <span class="text-3xl font-bold text-sky-400">{stats['hit_rate_pct']}%</span>
                    </div>
                    <p class="text-xs text-slate-500 mt-2">{stats['cache_hits']} hits / {stats['total_requests']} requests</p>
                </div>

                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg border-l-4 border-l-indigo-500">
                    <p class="text-xs font-semibold uppercase tracking-wider text-indigo-400">Model Cascade Routing</p>
                    <div class="mt-2 flex gap-4 text-sm font-mono">
                        <div><span class="text-emerald-400 font-bold">{stats['cascade_local_count']}</span> <span class="text-slate-500 text-xs">Local ($0)</span></div>
                        <div><span class="text-purple-400 font-bold">{stats['cascade_frontier_count']}</span> <span class="text-slate-500 text-xs">Frontier</span></div>
                    </div>
                    <p class="text-xs text-slate-500 mt-2">Auto-routed by complexity classifier</p>
                </div>
            </div>

            <!-- Two Column Section -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Optimization Features -->
                <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
                    <h3 class="text-base font-semibold text-slate-200">🚀 Active Novel-Query Optimizations</h3>
                    <ul class="space-y-3 text-sm text-slate-300">
                        <li class="flex items-start gap-2">
                            <span class="text-emerald-400 font-bold">✓</span>
                            <span><strong>Context & AST Pruner:</strong> Strips redundant license headers, boilerplate comments, and blank lines from injected files before upstream calls.</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-emerald-400 font-bold">✓</span>
                            <span><strong>Complexity Cascader:</strong> Detects routine code tasks (regex, formatting, syntax, docstrings) and routes to local/cheap tiers.</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-emerald-400 font-bold">✓</span>
                            <span><strong>Dual Protocol:</strong> Accepts both OpenAI (<code>/v1/chat/completions</code>) and Claude Code (<code>/v1/messages</code>).</span>
                        </li>
                    </ul>
                </div>

                <!-- Live Endpoints -->
                <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
                    <h3 class="text-base font-semibold text-slate-200">🔌 Connected Clients</h3>
                    <div class="space-y-2 text-xs font-mono text-slate-300">
                        <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                            <span class="text-indigo-400">Claude Code CLI:</span> export ANTHROPIC_BASE_URL="http://localhost:{settings.PORT}"
                        </div>
                        <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                            <span class="text-sky-400">Cursor / Codex:</span> Override Base URL = http://localhost:{settings.PORT}/v1
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

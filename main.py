import time
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
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
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def stream_cached_response(cached_resp: dict):
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

    # Optimization A: Context & AST Pruning
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
            metrics.log_request(user_query, model, "HIT-EXACT", latency_ms, tokens_pruned, 0.0003)
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
            metrics.log_request(user_query, model, "HIT-SEMANTIC", latency_ms, tokens_pruned, 0.0003)
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

    # 3. Cache Miss: Brand-New Query -> Model Cascader
    tier, target_model, reason = cascader.evaluate_complexity(user_query, pruned_tokens)
    metrics.record_cascade(tier)

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
        metrics.log_request(user_query, payload.get("model", model), "MISS", latency_ms, tokens_pruned, 0.0)
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
        metrics.log_request(user_query, model, "HIT-EXACT", latency_ms, tokens_pruned, 0.0003)
        return JSONResponse(
            content=anthropic_resp,
            headers={**base_headers, "X-Cache": "HIT", "X-Latency-Ms": str(latency_ms)}
        )

    tier, target_model, reason = cascader.evaluate_complexity(user_query, pruned_tokens)
    metrics.record_cascade(tier)

    # Native forward to Anthropic API
    anthropic_payload = {
        "model": model,
        "messages": raw_messages,
        "max_tokens": payload.get("max_tokens", 4096),
        "temperature": payload.get("temperature", 0.7)
    }
    if system_prompt:
        anthropic_payload["system"] = system_prompt

    resp_data = await proxy.forward_anthropic(anthropic_payload, dict(request.headers), exact_hash, user_query)
    latency_ms = round((time.time() - start_time) * 1000, 2)
    metrics.log_request(user_query, model, "MISS", latency_ms, tokens_pruned, 0.0)

    return JSONResponse(
        content=resp_data,
        headers={
            **base_headers,
            "X-Cache": "MISS",
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
    return {"status": "healthy", "service": "Alp-Cost Optimizer & Router", "version": "1.2.0"}

@app.get("/", response_class=FileResponse)
async def dashboard():
    index_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(index_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

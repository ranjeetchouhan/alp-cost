import json
import time
import httpx
from fastapi import HTTPException
from typing import AsyncGenerator, Dict, Any, Optional
from config import settings
from core.cache_exact import exact_cache
from core.cache_semantic import semantic_cache

class UpstreamProxy:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)

    # ----------------------------------------------------
    # 1. Native Anthropic Forwarding (Team & Enterprise)
    # ----------------------------------------------------
    async def forward_anthropic(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        exact_hash: str,
        user_query: str
    ) -> Dict[str, Any]:
        """Directly forwards requests to Anthropic's native Messages API."""
        api_key = (
            headers.get("x-api-key")
            or settings.UPSTREAM_API_KEY
            or headers.get("authorization", "").replace("Bearer ", "")
        )
        version = headers.get("anthropic-version", "2023-06-01")

        target_url = "https://api.anthropic.com/v1/messages"
        req_headers = {
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": version
        }

        resp = await self.client.post(target_url, json=payload, headers=req_headers)
        if resp.status_code != 200:
            error_detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text}
            raise HTTPException(status_code=resp.status_code, detail=error_detail)

        data = resp.json()

        # Save to Exact & Semantic Caches
        if settings.EXACT_CACHE_ENABLED:
            exact_cache.set(exact_hash, payload.get("model", "claude"), data)
        if settings.SEMANTIC_CACHE_ENABLED and user_query:
            semantic_cache.set(user_query, payload.get("model", "claude"), data)

        return data

    # ----------------------------------------------------
    # 2. OpenAI-Compatible Forwarding (Cursor, Codex, etc.)
    # ----------------------------------------------------
    async def forward_non_streaming(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        exact_hash: str,
        user_query: str
    ) -> Dict[str, Any]:
        upstream_key = settings.UPSTREAM_API_KEY or headers.get("authorization", "").replace("Bearer ", "")
        target_url = f"{settings.UPSTREAM_BASE_URL.rstrip('/')}/chat/completions"
        req_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {upstream_key}"
        }

        resp = await self.client.post(target_url, json=payload, headers=req_headers)
        if resp.status_code != 200:
            error_detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text}
            raise HTTPException(status_code=resp.status_code, detail=error_detail)

        data = resp.json()

        if settings.EXACT_CACHE_ENABLED:
            exact_cache.set(exact_hash, payload.get("model", "default"), data)
        if settings.SEMANTIC_CACHE_ENABLED and user_query:
            semantic_cache.set(user_query, payload.get("model", "default"), data)

        return data

    async def forward_streaming(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        exact_hash: str,
        user_query: str
    ) -> AsyncGenerator[str, None]:
        upstream_key = settings.UPSTREAM_API_KEY or headers.get("authorization", "").replace("Bearer ", "")
        target_url = f"{settings.UPSTREAM_BASE_URL.rstrip('/')}/chat/completions"
        req_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {upstream_key}"
        }

        collected_content = []
        model = payload.get("model", "default")
        resp_id = ""

        async with self.client.stream("POST", target_url, json=payload, headers=req_headers) as response:
            if response.status_code != 200:
                err_text = await response.aread()
                yield f"data: {json.dumps({'error': err_text.decode('utf-8')})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                if not line:
                    continue
                yield f"{line}\n\n"
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    try:
                        chunk_json = json.loads(line[6:])
                        resp_id = chunk_json.get("id", resp_id)
                        choices = chunk_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                collected_content.append(content)
                    except Exception:
                        pass

        full_text = "".join(collected_content)
        if full_text:
            full_response = self._create_response_obj(resp_id, model, full_text)
            if settings.EXACT_CACHE_ENABLED:
                exact_cache.set(exact_hash, model, full_response)
            if settings.SEMANTIC_CACHE_ENABLED and user_query:
                semantic_cache.set(user_query, model, full_response)

    def _create_response_obj(self, resp_id: str, model: str, content: str) -> Dict[str, Any]:
        return {
            "id": resp_id or f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": len(content.split()), "total_tokens": 20 + len(content.split())}
        }

proxy = UpstreamProxy()

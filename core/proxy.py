import json
import time
import httpx
from fastapi.responses import Response, JSONResponse
from typing import AsyncGenerator, Dict, Any, Tuple, Optional
from config import settings
from core.cache_exact import exact_cache
from core.cache_semantic import semantic_cache

class UpstreamProxy:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)

    # ----------------------------------------------------
    # 1. Native Anthropic Forwarding (Team & Enterprise)
    # ----------------------------------------------------
    async def forward_anthropic_non_streaming(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        exact_hash: str,
        user_query: str
    ) -> Tuple[int, Dict[str, Any]]:
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
        if headers.get("anthropic-beta"):
            req_headers["anthropic-beta"] = headers.get("anthropic-beta")

        resp = await self.client.post(target_url, json=payload, headers=req_headers)
        
        # Parse JSON
        try:
            data = resp.json()
        except Exception:
            data = {"error": resp.text}

        if resp.status_code == 200:
            if settings.EXACT_CACHE_ENABLED:
                exact_cache.set(exact_hash, payload.get("model", "claude"), data)
            if settings.SEMANTIC_CACHE_ENABLED and user_query:
                semantic_cache.set(user_query, payload.get("model", "claude"), data)

        return resp.status_code, data

    async def forward_anthropic_streaming(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        exact_hash: str,
        user_query: str
    ):
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
        if headers.get("anthropic-beta"):
            req_headers["anthropic-beta"] = headers.get("anthropic-beta")

        collected_text = []
        msg_id = ""
        model = payload.get("model", "claude")

        async with self.client.stream("POST", target_url, json=payload, headers=req_headers) as response:
            if response.status_code != 200:
                err_text = await response.aread()
                # Yield error event
                yield f"event: error\ndata: {err_text.decode('utf-8', errors='replace')}\n\n"
                return

            async for line in response.aiter_lines():
                if not line:
                    yield "\n"
                    continue
                yield f"{line}\n"

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type", "")
                        if event_type == "message_start":
                            msg_id = data.get("message", {}).get("id", "")
                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                collected_text.append(delta.get("text", ""))
                    except Exception:
                        pass

        full_content = "".join(collected_text)
        if full_content:
            cached_obj = {
                "id": msg_id or f"msg_{int(time.time())}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_content}],
                "model": model,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": len(full_content.split())}
            }
            if settings.EXACT_CACHE_ENABLED:
                exact_cache.set(exact_hash, model, cached_obj)
            if settings.SEMANTIC_CACHE_ENABLED and user_query:
                semantic_cache.set(user_query, model, cached_obj)

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
        data = resp.json()

        if resp.status_code == 200:
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

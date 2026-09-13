import sys
from mcp.server.fastmcp import FastMCP
from core.pruner import pruner
from core.cache_semantic import semantic_cache
from core.router import metrics

# Initialize FastMCP Server
mcp = FastMCP("Alp-Cost Optimizer")

@mcp.tool()
def prune_context(code_or_text: str) -> dict:
    """
    Prunes redundant legal notices, divider comments, and blank lines from code or context files
    before sending across tools or prompts, cutting token usage by 50% - 80%.
    """
    cleaned, orig_tokens, pruned_tokens = pruner.prune_content(code_or_text)
    tokens_saved = max(0, orig_tokens - pruned_tokens)
    reduction_pct = round((tokens_saved / orig_tokens * 100), 1) if orig_tokens > 0 else 0.0
    return {
        "pruned_content": cleaned,
        "original_tokens": orig_tokens,
        "pruned_tokens": pruned_tokens,
        "tokens_saved": tokens_saved,
        "reduction_pct": f"{reduction_pct}%"
    }

@mcp.tool()
def search_semantic_cache(query: str, model: str = "claude-3-5-sonnet") -> dict:
    """
    Searches the Alp-Cost semantic cache for a previously solved query.
    Returns the cached answer if a high-confidence semantic match is found.
    """
    match = semantic_cache.search(query, model)
    if match:
        resp, score, matched_q = match
        content = ""
        choices = resp.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        return {
            "found": True,
            "similarity_score": round(score, 4),
            "original_query": matched_q,
            "solution": content or resp
        }
    return {"found": False, "message": "No cache hit found for this query."}

@mcp.tool()
def get_cost_metrics() -> dict:
    """
    Returns real-time money saved (in USD and INR), hit rate, and tokens pruned across all sessions.
    """
    return metrics.get_stats()

if __name__ == "__main__":
    mcp.run()

import re
from typing import Dict, Any, Tuple

class ModelCascader:
    """
    Intelligently routes novel queries based on technical complexity:
    - Simple tasks (regex, formatting, syntax, docstrings) -> Local/Cheap Tier ($0.00)
    - Complex tasks (refactoring, concurrency, architecture) -> Frontier Tier
    """

    SIMPLE_PATTERNS = [
        r'(?i)\b(regex|regular expression|format|prettify|lint)\b',
        r'(?i)\b(write a docstring|add comments|explain this error|syntax error)\b',
        r'(?i)\b(convert (json|yaml|csv|xml|dict|class)|to json|to yaml)\b',
        r'(?i)\b(translate|rename variable|generate getter|boilerplate)\b',
        r'(?i)\b(what does this (function|line|regex|code) do)\b',
        r'(?i)\b(simple unit test|mock data|dummy data)\b'
    ]

    COMPLEX_PATTERNS = [
        r'(?i)\b(refactor across|architecture|distributed|consensus|microservice)\b',
        r'(?i)\b(race condition|deadlock|concurrency|thread safe|memory leak)\b',
        r'(?i)\b(security audit|vulnerability|exploit|cryptograph|cve)\b',
        r'(?i)\b(mathematical proof|algorithm optimization|dynamic programming)\b',
        r'(?i)\b(database schema migration|multi-tenant|kubernetes cluster)\b'
    ]

    def evaluate_complexity(self, user_query: str, total_tokens: int) -> Tuple[str, str, str]:
        """
        Returns: (tier, recommended_model, explanation)
        """
        # 1. Check for complex signals
        for pat in self.COMPLEX_PATTERNS:
            if re.search(pat, user_query):
                return "TIER_FRONTIER", "claude-3-5-sonnet", "Detected high-complexity architectural or concurrency task."

        # 2. Check for simple signals
        for pat in self.SIMPLE_PATTERNS:
            if re.search(pat, user_query):
                return "TIER_LOCAL", "qwen2.5:0.5b", "Detected routine boilerplate, formatting, or explanation task."

        # 3. Token-based heuristic: Very short or focused query with moderate context
        if len(user_query.split()) < 20 and total_tokens < 1500:
            return "TIER_LOCAL", "qwen2.5:0.5b", "Short, self-contained coding query suitable for local SLM."

        # Default fallback to requested model or standard tier
        return "TIER_STANDARD", "qwen2.5:0.5b", "Standard complexity task."

cascader = ModelCascader()

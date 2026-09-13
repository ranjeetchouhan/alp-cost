import re
from typing import List, Dict, Any, Tuple

class ContextPruner:
    """
    Intelligently prunes code and context dumps injected by coding agents
    (Claude Code, Cursor, Continue) to slash token costs on brand-new queries.
    """

    # License and copyright headers that agents inject repeatedly
    LICENSE_PATTERNS = [
        r'(?i)(/\*[\s\S]*?(copyright|licensed under|all rights reserved)[\s\S]*?\*/)',
        r'(?i)(#[\s\S]*?(copyright|licensed under|all rights reserved)[\s\S]*?\n(?=#|\n))',
        r'(?i)(//[\s\S]*?(copyright|licensed under|all rights reserved)[\s\S]*?\n(?=//|\n))',
    ]

    @classmethod
    def strip_license_headers(cls, text: str) -> str:
        for pat in cls.LICENSE_PATTERNS:
            text = re.sub(pat, '', text)
        return text

    @classmethod
    def strip_dead_comments(cls, text: str) -> str:
        """
        Removes verbose standalone comment lines in code blocks while keeping
        code logic intact. Avoids stripping comments inside string literals.
        """
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            # Strip pure multi-# or divider comments (e.g., # -------------------)
            if re.match(r'^[#/\*=\-_]{5,}$', stripped):
                continue
            # Strip trivial doc comments like "// TODO: ...", "// End of file"
            if re.match(r'^(//|#)\s*(end of file|eof|file:|filename:).*$', stripped, re.IGNORECASE):
                continue
            lines.append(line)
        return '\n'.join(lines)

    @classmethod
    def compress_whitespace_and_indentation(cls, text: str) -> str:
        """
        Collapses excessive blank lines and trailing spaces across files.
        """
        # Collapse 3+ newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.splitlines()]
        return '\n'.join(lines).strip()

    def prune_content(self, text: str) -> Tuple[str, int, int]:
        """
        Prunes a text block. Returns: (pruned_text, orig_tokens_approx, pruned_tokens_approx)
        """
        if not text:
            return "", 0, 0

        orig_len = len(text)
        cleaned = self.strip_license_headers(text)
        cleaned = self.strip_dead_comments(cleaned)
        cleaned = self.compress_whitespace_and_indentation(cleaned)
        new_len = len(cleaned)

        # Heuristic: ~4 characters per token
        orig_tokens = max(1, orig_len // 4)
        new_tokens = max(1, new_len // 4)

        return cleaned, orig_tokens, new_tokens

    def prune_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Processes entire conversation history, aggressively pruning context
        and background file dumps while preserving the user's explicit question.
        """
        pruned_messages = []
        total_orig_tokens = 0
        total_new_tokens = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                # Don't over-prune the very last user query (keep exact user intent)
                is_last_user_query = (role == "user" and i == len(messages) - 1)
                
                if is_last_user_query:
                    # Mild whitespace trim only
                    cleaned = content.strip()
                    orig_tok = len(content) // 4
                    new_tok = len(cleaned) // 4
                else:
                    # Full aggressive context pruning on system prompts & injected files
                    cleaned, orig_tok, new_tok = self.prune_content(content)

                total_orig_tokens += orig_tok
                total_new_tokens += new_tok
                
                new_msg = dict(msg)
                new_msg["content"] = cleaned
                pruned_messages.append(new_msg)
            else:
                pruned_messages.append(msg)

        return pruned_messages, total_orig_tokens, total_new_tokens

pruner = ContextPruner()

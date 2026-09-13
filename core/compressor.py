import re
from typing import List, Dict, Any, Tuple

class PromptCompressor:
    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        # 1. Normalize multiple newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 2. Strip trailing whitespaces on each line
        lines = [line.rstrip() for line in text.splitlines()]
        text = '\n'.join(lines)
        # 3. Collapse multiple horizontal spaces to single space, except in code blocks
        # (Simple heuristic outside of triple backticks)
        parts = text.split("```")
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r'[ \t]{2,}', ' ', parts[i])
        return "```".join(parts).strip()

    def compress_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Compresses a messages list by stripping token waste.
        Returns: (compressed_messages, original_char_count, compressed_char_count)
        """
        compressed = []
        orig_chars = 0
        comp_chars = 0

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                orig_chars += len(content)
                cleaned = self._clean_text(content)
                comp_chars += len(cleaned)
                new_msg = dict(msg)
                new_msg["content"] = cleaned
                compressed.append(new_msg)
            else:
                compressed.append(msg)

        return compressed, orig_chars, comp_chars

compressor = PromptCompressor()

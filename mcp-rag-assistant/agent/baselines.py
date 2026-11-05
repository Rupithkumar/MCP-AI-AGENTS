from __future__ import annotations

import re

from .mcp_agent import ToolCall, Turn

NOT_FOUND = "I could not find this in the documents."
_FIRST_HIT = re.compile(r"^\[1\] (\S+#\d+) \|[^\n]*\n([^\n]+)", re.MULTILINE)


class ExtractiveLLM:
    async def next_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        last = messages[-1]["content"]
        if isinstance(last, str):  
            name = next((t["name"] for t in tools if t["name"].endswith("search_docs")), None)
            if name is None:
                return Turn(text="No search tool is available.")
            return Turn(tool_calls=[ToolCall("baseline-1", name, {"query": last, "k": 3})])
        result = last[0]["content"]  
        match = _FIRST_HIT.search(result)
        if last[0].get("is_error") or not match:
            return Turn(text=NOT_FOUND)
        cite, text = match.groups()
        return Turn(text=f"{text} [{cite}]")

from __future__ import annotations

import asyncio

from .brains import make_brain

TOOLS = [
    {
        "name": "add_numbers",
        "description": "Add two numbers and return the sum.",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }
]
SYSTEM = "You are a helpful assistant. Use the provided tools when they help."
PROMPT = "Use the add_numbers tool to add 17 and 25, then tell me the result."

HINTS = [
    ("401", "The key was rejected. Check LLM_API_KEY (no quotes or spaces) and that it belongs to this provider."),
    ("403", "Access denied. The key may lack permission, or the provider blocks your region."),
    ("404", "Model or URL not found. Check LLM_MODEL (it must match the provider's list exactly) and LLM_BASE_URL."),
    ("429", "Rate limit reached. Wait a minute and run it again."),
    ("onnect", "Cannot reach the server. Check LLM_BASE_URL and your internet (or that Ollama is running)."),
    ("tool", "The provider or model may not support tool calling. Try a different model."),
]


def hint_for(error: str) -> str:
    for needle, hint in HINTS:
        if needle in error:
            return hint
    return "Unexpected error. Copy the message above when asking for help."


async def check() -> int:
    llm, label = make_brain("llm")  
    print(f"[brain] {label}")
    messages: list[dict] = [{"role": "user", "content": PROMPT}]
    try:
        first = await llm.next_turn(SYSTEM, messages, TOOLS)
        if not first.tool_calls:
            print("FAIL: the model answered without calling the tool.")
            print("      It may not support tool calling. Try a larger model from the provider's list.")
            return 1
        call = first.tool_calls[0]
        print(f"ok: the model called {call.name}({call.args})")
        try:
            total = float(call.args["a"]) + float(call.args["b"])
        except (KeyError, TypeError, ValueError):
            print("FAIL: the tool call had missing or invalid arguments. Try a larger model.")
            return 1
        answer_text = f"{total:g}"

        messages.append({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}],
        })
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call.id, "content": answer_text, "is_error": False}],
        })
        second = await llm.next_turn(SYSTEM, messages, TOOLS)
    except Exception as exc:  
        error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR {error[:300]}")
        print(f"      {hint_for(error)}")
        return 1

    tokens = first.input_tokens + second.input_tokens + first.output_tokens + second.output_tokens
    if "42" not in second.text:
        print(f"WARN: the final answer did not contain 42: {second.text[:120]!r}")
        return 1
    print(f"ok: the model used the tool result: {second.text.strip()[:120]!r}")
    print(f"PASS: this model works with the agent." + (f" ({tokens} tokens used)" if tokens else ""))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(check()))


if __name__ == "__main__":
    main()

from __future__ import annotations

SYSTEM = "You are a customer-support intent classifier."
STYLES = ("short", "zero_shot", "few_shot")


def label_block(labels: list[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def build_messages(text: str, labels: list[str], style: str, shots: list[dict] | None = None) -> list[dict]:
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if style == "short":
        user = f"Classify the customer message into its intent label.\nMessage: {text}\nReply with the label only."
    else:
        parts = [f"Labels:\n{label_block(labels)}\n"]
        if style == "few_shot":
            if not shots:
                raise ValueError("few_shot needs shots")
            examples = "\n".join(f"Message: {s['text']}\nLabel: {s['label']}\n" for s in shots)
            parts.append(f"Examples:\n{examples}")
        parts.append(f"Message: {text}\nReply with exactly one label from the list and nothing else.")
        user = "\n".join(parts)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

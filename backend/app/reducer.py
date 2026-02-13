from __future__ import annotations


def shrink_prompt_static(prompt: str, max_steps: int = 10) -> str:
    """
    Lightweight placeholder reducer for MVP.
    Keeps first and last paragraphs for a compact repro candidate.
    """
    chunks = [c for c in prompt.split("\n\n") if c.strip()]
    if len(chunks) <= 2:
        return prompt
    _ = max_steps
    return "\n\n".join([chunks[0], chunks[-1]])

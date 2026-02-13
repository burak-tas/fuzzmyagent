from __future__ import annotations

import time
from typing import Any

import httpx


class A2ARequestError(Exception):
    pass


async def call_a2a(
    endpoint: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float
) -> dict[str, Any]:
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(endpoint, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        body = ""
        if e.response is not None:
            try:
                body = (e.response.text or "")[:600]
            except Exception:
                body = ""
        raise A2ARequestError(
            f"Client/server error '{status}' for url '{endpoint}'. body='{body}'"
        )
    except httpx.TimeoutException:
        raise A2ARequestError(f"Request timed out for url '{endpoint}'")
    except Exception as e:
        raise A2ARequestError(str(e))

    return {"latency_ms": int((time.time() - t0) * 1000), "raw": data}


async def call_a2a_adaptive(
    endpoint: str,
    prompt: str,
    meta: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    # Try common agent endpoint payload shapes to handle provider differences.
    variants = [
        ("input_meta", {"input": prompt, "meta": meta}),
        ("input_only", {"input": prompt}),
        ("messages_string", {"messages": [{"role": "user", "content": prompt}], "meta": meta}),
        (
            "messages_content_blocks",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                "meta": meta,
            },
        ),
        ("text_meta", {"text": prompt, "meta": meta}),
        (
            "jsonrpc_message_send",
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {"message": prompt, "meta": meta},
            },
        ),
    ]

    errors: list[str] = []
    for variant_name, payload in variants:
        try:
            out = await call_a2a(endpoint, payload, headers, timeout_s)
            out["payload_variant"] = variant_name
            out["payload_used"] = payload
            return out
        except A2ARequestError as e:
            errors.append(f"{variant_name}: {e}")
            continue

    raise A2ARequestError(
        "All payload variants failed. " + " | ".join(errors[:6])
    )

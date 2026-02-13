from __future__ import annotations

import json
import time
from typing import Any

import httpx


class A2ARequestError(Exception):
    pass


def _infer_output_modes(agent_card: Any) -> list[str]:
    if not isinstance(agent_card, dict):
        return ["text"]
    for key in ("acceptedOutputModes", "defaultOutputModes", "defaultOutputMode"):
        v = agent_card.get(key)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
        if isinstance(v, str) and v:
            return [v]
    for parent in ("capabilities", "configuration"):
        pv = agent_card.get(parent)
        if isinstance(pv, dict):
            v = pv.get("acceptedOutputModes") or pv.get("defaultOutputModes")
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return v
    return ["text"]


def _make_message_id(seed: str) -> str:
    # Many A2A gateways validate messageId with a restricted pattern.
    # Create a stable-ish id like: msg-<time_ms>-<suffix> (lowercase, only [a-z0-9-]).
    base = (seed or "").lower().replace("_", "-")
    base = "".join(ch for ch in base if (ch.isalnum() or ch == "-"))
    suffix = (base[-12:] or "x")
    ms = int(time.time() * 1000)
    return f"msg-{ms}-{suffix}"


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
    agent_card: Any = None,
) -> dict[str, Any]:
    # Try common agent endpoint payload shapes to handle provider differences.
    prompt_text = (prompt or "").strip()
    parsed_prompt: Any = None
    try:
        parsed_prompt = json.loads(prompt_text) if prompt_text else None
    except Exception:
        parsed_prompt = None

    protocol_version = ""
    if isinstance(agent_card, dict):
        protocol_version = str(agent_card.get("protocolVersion") or "")

    jsonrpc_id = str(meta.get("testcase_id") or meta.get("run_id") or "1")
    effective_timeout_s = timeout_s
    if protocol_version:
        # JSON-RPC brokers frequently use blocking calls that can exceed 30s.
        effective_timeout_s = max(timeout_s, 90.0)

    accepted_output_modes = _infer_output_modes(agent_card)

    variants: list[tuple[str, dict[str, Any]]] = []

    # If prompt is already a JSON object (e.g. user pasted a JSON-RPC payload), send it as-is first.
    if isinstance(parsed_prompt, dict):
        variants.append(("raw_json_prompt", dict(parsed_prompt)))

    # Strong JSON-RPC envelope matching A2A 0.3 brokers (schema-validated).
    # Example target shape (from user):
    # {"jsonrpc":"2.0","id":"2","method":"message/send","params":{"message":{"role":"user","parts":[{"text":"...","kind":"text"}],"messageId":"...","kind":"message"},"configuration":{"acceptedOutputModes":["text"],"blocking":true}}}
    a2a_v03_message = {
        "kind": "message",
        "role": "user",
        "messageId": _make_message_id(jsonrpc_id),
        # Match common schema: { "text": "...", "kind": "text" }
        "parts": [{"text": prompt, "kind": "text"}],
    }
    a2a_v03_params = {
        "message": a2a_v03_message,
        "configuration": {"acceptedOutputModes": accepted_output_modes, "blocking": True},
    }

    # Most A2A 0.3 deployments use "message/send" (slash), not "message.send" (dot).
    # We keep the dot-method out by default because many gateways reject it with -32601.
    method = "message/send"
    variants.append(
        (
            "a2a_jsonrpc_v03_message/send",
            {"jsonrpc": "2.0", "id": jsonrpc_id, "method": method, "params": a2a_v03_params},
        )
    )
    # Some servers hang when blocking=true; try a non-blocking variant too.
    variants.append(
        (
            "a2a_jsonrpc_v03_message/send_nonblocking",
            {
                "jsonrpc": "2.0",
                "id": jsonrpc_id,
                "method": method,
                "params": {
                    "message": a2a_v03_message,
                    "configuration": {
                        "acceptedOutputModes": accepted_output_modes,
                        "blocking": False,
                    },
                },
            },
        )
    )
    # Some implementations don't accept "configuration" at all.
    variants.append(
        (
            "a2a_jsonrpc_v03_message/send_no_configuration",
            {
                "jsonrpc": "2.0",
                "id": jsonrpc_id,
                "method": method,
                "params": {"message": a2a_v03_message},
            },
        )
    )

    # Non-JSON-RPC payloads (only useful when the endpoint is not a JSON-RPC gateway).
    # JSON-RPC gateways will fail these consistently with "id field is required but missing".
    if not protocol_version:
        variants.extend(
            [
                ("input_meta", {"input": prompt, "meta": meta}),
                ("input_only", {"input": prompt}),
                (
                    "messages_string",
                    {"messages": [{"role": "user", "content": prompt}], "meta": meta},
                ),
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
            ]
        )

    errors: list[str] = []
    for variant_name, payload in variants:
        try:
            out = await call_a2a(endpoint, payload, headers, effective_timeout_s)
            out["payload_variant"] = variant_name
            out["payload_used"] = payload
            return out
        except A2ARequestError as e:
            errors.append(f"{variant_name}: {e}")
            continue

    raise A2ARequestError(
        "All payload variants failed. " + " | ".join(errors[:10])
    )

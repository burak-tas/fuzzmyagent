from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable

import yaml

from .a2a_client import call_a2a
from .fuzzing import generate_cases
from .models import Case, Run
from .reducer import shrink_prompt_static
from .rules import evaluate_rules, overall_pass, compile_rules

# In-memory run cancellation flags (MVP).
CANCEL: dict[str, bool] = {}

# WebSocket broadcast callback per run_id.
BROADCAST: dict[str, Callable[[dict[str, Any]], Any]] = {}


async def set_broadcast(run_id: str, fn: Callable[[dict[str, Any]], Any]) -> None:
    BROADCAST[run_id] = fn


async def _emit(run_id: str, event: dict[str, Any]) -> None:
    fn = BROADCAST.get(run_id)
    if fn:
        await fn(event)


async def emit_run_event(run_id: str, event: dict[str, Any]) -> None:
    await _emit(run_id, event)


async def stop_run(run_id: str) -> None:
    CANCEL[run_id] = True


async def run_job(run_id: str, session_factory) -> None:
    CANCEL[run_id] = False

    async with session_factory() as db:
        run = await db.get(Run, run_id)
        if not run:
            return
        run.status = "running"
        await db.commit()

    async with session_factory() as db:
        run = await db.get(Run, run_id)
        if not run:
            return
        rules_obj = yaml.safe_load(run.rules_yaml) or {}
        cfg_obj = yaml.safe_load(run.config_yaml) or {}

    compiled = compile_rules(rules_obj)
    target = cfg_obj.get("target", {})
    endpoint = target.get("endpoint", "")
    headers = target.get("headers", {}) or {}
    timeout_s = float(target.get("timeout_s", 30))

    runner_cfg = cfg_obj.get("runner", {})
    concurrency = int(runner_cfg.get("concurrency", 6))
    reduce_failures = bool(runner_cfg.get("reduce_failures", False))

    fuzz_cfg = cfg_obj.get("fuzzing", {})
    cases = generate_cases(fuzz_cfg)

    async with session_factory() as db:
        run = await db.get(Run, run_id)
        if not run:
            return
        run.progress_total = len(cases)
        await db.commit()

    sem = asyncio.Semaphore(concurrency)

    async def handle_one(i: int, prompt: str, mutation: str) -> None:
        async with sem:
            if CANCEL.get(run_id):
                return

            case_id = uuid.uuid4().hex
            payload = {"input": prompt, "meta": {"run_id": run_id, "case_index": i}}

            trace: dict[str, Any] = {"payload": payload, "tool_calls": []}
            response_text = ""
            ok = False

            try:
                a2a = await call_a2a(endpoint, payload, headers, timeout_s)
                trace["latency_ms"] = a2a["latency_ms"]
                trace["raw"] = a2a["raw"]

                raw = a2a["raw"]
                if isinstance(raw, dict):
                    response_text = raw.get("output") or raw.get("message") or json.dumps(raw)
                    if isinstance(raw.get("tool_calls"), list):
                        trace["tool_calls"] = raw["tool_calls"]
                else:
                    response_text = str(raw)

                rule_results = evaluate_rules(compiled, response_text, trace)
                ok = overall_pass(rule_results)

            except Exception as e:
                trace["error"] = str(e)
                rule_results = [
                    {
                        "rule_id": "endpoint_error",
                        "ok": False,
                        "severity": "error",
                        "message": "A2A call failed",
                        "evidence": str(e),
                    }
                ]
                ok = False

            prompt_min = ""
            if reduce_failures and not ok and prompt:
                prompt_min = shrink_prompt_static(prompt)

            async with session_factory() as db:
                c = Case(
                    id=case_id,
                    run_id=run_id,
                    index=i,
                    prompt=prompt,
                    prompt_min=prompt_min,
                    mutation=mutation,
                    response_text=response_text,
                    trace_json=json.dumps(trace),
                    result_json=json.dumps(
                        {
                            "rules": [
                                r.__dict__ if hasattr(r, "__dict__") else r
                                for r in rule_results
                            ]
                        }
                    ),
                    passed=1 if ok else 0,
                )
                db.add(c)

                run = await db.get(Run, run_id)
                if not run:
                    return
                run.progress_done += 1
                await db.commit()

                await _emit(
                    run_id,
                    {
                        "type": "case_done",
                        "run_id": run_id,
                        "done": run.progress_done,
                        "total": run.progress_total,
                        "case": {
                            "id": case_id,
                            "index": i,
                            "passed": ok,
                            "mutation": mutation,
                        },
                    },
                )

    tasks = [
        asyncio.create_task(handle_one(i, fc.prompt, fc.mutation))
        for i, fc in enumerate(cases)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    async with session_factory() as db:
        run = await db.get(Run, run_id)
        if run:
            run.status = "stopped" if CANCEL.get(run_id) else "finished"
            await db.commit()

    await _emit(run_id, {"type": "run_done", "run_id": run_id})

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RuleResult:
    rule_id: str
    ok: bool
    severity: str
    message: str
    evidence: Optional[str] = None


@dataclass
class CompiledRule:
    rule_id: str
    type: str
    severity: str
    params: dict[str, Any]


def compile_rules(rules_obj: dict[str, Any]) -> list[CompiledRule]:
    rules = rules_obj.get("rules", [])
    compiled: list[CompiledRule] = []
    for r in rules:
        compiled.append(
            CompiledRule(
                rule_id=str(r.get("id", "")) or f"rule_{len(compiled)+1}",
                type=r["type"],
                severity=r.get("severity", "error"),
                params={k: v for k, v in r.items() if k not in ("id", "type", "severity")},
            )
        )
    return compiled


def evaluate_rules(
    compiled: list[CompiledRule], response_text: str, trace: dict[str, Any]
) -> list[RuleResult]:
    out: list[RuleResult] = []
    for rule in compiled:
        t = rule.type
        p = rule.params

        if t == "max_length":
            max_chars = int(p.get("chars", 4000))
            ok = len(response_text) <= max_chars
            out.append(
                RuleResult(
                    rule.rule_id,
                    ok,
                    rule.severity,
                    f"Response length <= {max_chars} chars",
                    evidence=f"len={len(response_text)}",
                )
            )

        elif t == "forbidden_substrings":
            forbidden = p.get("values", [])
            hit = next((s for s in forbidden if s and s in response_text), None)
            ok = hit is None
            out.append(
                RuleResult(
                    rule.rule_id,
                    ok,
                    rule.severity,
                    "No forbidden substrings present",
                    evidence=hit,
                )
            )

        elif t == "regex_must_match":
            pattern = p.get("pattern", "")
            ok = bool(re.search(pattern, response_text, flags=re.MULTILINE))
            out.append(
                RuleResult(
                    rule.rule_id,
                    ok,
                    rule.severity,
                    f"Must match regex: {pattern}",
                    evidence=None if ok else response_text[:500],
                )
            )

        elif t == "json_parseable":
            try:
                json.loads(response_text)
                ok = True
                ev = None
            except Exception as e:
                ok = False
                ev = str(e)
            out.append(
                RuleResult(
                    rule.rule_id,
                    ok,
                    rule.severity,
                    "Response must be valid JSON",
                    evidence=ev,
                )
            )

        elif t == "tool_calls_allowlist":
            allow = set(p.get("allow", []))
            tool_calls = trace.get("tool_calls", [])
            bad = [c for c in tool_calls if c.get("name") not in allow]
            ok = len(bad) == 0
            out.append(
                RuleResult(
                    rule.rule_id,
                    ok,
                    rule.severity,
                    f"Tool calls must be within allowlist: {sorted(allow)}",
                    evidence=json.dumps(bad)[:800] if bad else None,
                )
            )

        else:
            out.append(
                RuleResult(
                    rule.rule_id,
                    True,
                    "warn",
                    f"Unknown rule type '{t}' (skipped)",
                )
            )

    return out


def overall_pass(rule_results: list[RuleResult]) -> bool:
    for r in rule_results:
        if r.severity == "error" and not r.ok:
            return False
    return True

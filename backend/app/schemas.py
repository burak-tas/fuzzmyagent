from typing import Any

from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    rules_yaml: str
    config_yaml: str


class RunInfo(BaseModel):
    id: str
    status: str
    created_at: int
    progress_done: int
    progress_total: int
    summary: dict[str, Any] = Field(default_factory=dict)


class CaseSummary(BaseModel):
    id: str
    index: int
    passed: bool
    mutation: str
    prompt: str


class CaseDetail(BaseModel):
    id: str
    index: int
    passed: bool
    mutation: str
    prompt: str
    prompt_min: str
    response_text: str
    trace: dict[str, Any]
    result: dict[str, Any]

const API_BASE = "http://localhost:8000";

export type DiscoveryResponse = {
  run_id: string;
  normalized_endpoint: string;
  agent_card: Record<string, unknown> | null;
  discovery: { attempts: Array<Record<string, unknown>>; warnings: string[] };
};

export async function discoverEndpoint(endpoint: string, runId?: string): Promise<DiscoveryResponse> {
  const r = await fetch(`${API_BASE}/api/discovery`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ endpoint, run_id: runId ?? null }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function testOpenAI(apiKey: string, model: string): Promise<{ ok: boolean }> {
  const r = await fetch(`${API_BASE}/api/openai/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, model }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function generateTestcases(payload: {
  run_id: string;
  api_key?: string;
  openai: Record<string, unknown>;
  agent: Record<string, unknown>;
  fuzzer: Record<string, unknown>;
  rules?: Record<string, unknown>;
  skip_llm?: boolean;
}) {
  const r = await fetch(`${API_BASE}/api/testcases/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function listTestcases(runId: string) {
  const r = await fetch(`${API_BASE}/api/runs/${runId}/testcases`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function createTestcase(
  runId: string,
  payload: { prompt: string; metadata?: Record<string, unknown> },
) {
  const r = await fetch(`${API_BASE}/api/runs/${runId}/testcases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function bulkCreateTestcases(
  runId: string,
  payload: { testcases: Array<{ prompt: string; metadata?: Record<string, unknown> }> },
) {
  const r = await fetch(`${API_BASE}/api/runs/${runId}/testcases/bulk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function patchTestcase(testcaseId: string, prompt: string) {
  const r = await fetch(`${API_BASE}/api/testcases/${testcaseId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function deleteTestcase(testcaseId: string) {
  const r = await fetch(`${API_BASE}/api/testcases/${testcaseId}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function startWizardRun(runId: string, payload: { endpoint?: string; execution: Record<string, unknown>; rules: Record<string, unknown> }) {
  const r = await fetch(`${API_BASE}/api/runs/${runId}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function stopWizardRun(runId: string) {
  const r = await fetch(`${API_BASE}/api/wizard/runs/${runId}/stop`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getWizardRun(runId: string) {
  const r = await fetch(`${API_BASE}/api/wizard/runs/${runId}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getReport(runId: string) {
  const r = await fetch(`${API_BASE}/api/runs/${runId}/report`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function runWebSocket(runId: string, onEvent: (e: any) => void) {
  const ws = new WebSocket(`ws://localhost:8000/ws/runs/${runId}`);
  ws.onmessage = (msg) => onEvent(JSON.parse(msg.data));
  ws.onopen = () => ws.send("hello");
  return ws;
}

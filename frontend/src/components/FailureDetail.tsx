import React from "react";

export function FailureDetail(props: { detail: any | null }) {
  if (!props.detail)
    return (
      <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12, opacity: 0.7 }}>
        Select a failure to inspect.
      </div>
    );

  const d = props.detail;
  const rules = d.result?.rules ?? [];

  return (
    <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
      <div style={{ fontWeight: 800 }}>Case #{d.index}</div>
      <div style={{ opacity: 0.8, marginBottom: 10 }}>Mutation: {d.mutation}</div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Prompt</div>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              background: "#fafafa",
              padding: 10,
              borderRadius: 10,
              border: "1px solid #eee",
            }}
          >
            {d.prompt}
          </pre>
        </div>
        <div>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Response</div>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              background: "#fafafa",
              padding: 10,
              borderRadius: 10,
              border: "1px solid #eee",
            }}
          >
            {d.response_text}
          </pre>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>Rule results</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {rules.map((r: any, idx: number) => (
            <div key={idx} style={{ border: "1px solid #eee", borderRadius: 10, padding: 10 }}>
              <div style={{ fontWeight: 700 }}>
                {r.rule_id} - {r.ok ? "OK" : "FAIL"} ({r.severity})
              </div>
              <div style={{ opacity: 0.85 }}>{r.message}</div>
              {r.evidence ? (
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    marginTop: 6,
                    background: "#fff",
                    padding: 8,
                    borderRadius: 8,
                    border: "1px solid #f0f0f0",
                  }}
                >
                  {r.evidence}
                </pre>
              ) : null}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

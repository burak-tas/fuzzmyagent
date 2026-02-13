import React from "react";

export function FailureList(props: {
  cases: any[];
  onSelect: (id: string) => void;
}) {
  const failed = props.cases.filter((c) => !c.passed);
  return (
    <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
      <div style={{ fontWeight: 700, marginBottom: 10 }}>Failures ({failed.length})</div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 8,
          maxHeight: 320,
          overflow: "auto",
        }}
      >
        {failed.map((c) => (
          <button
            key={c.id}
            onClick={() => props.onSelect(c.id)}
            style={{
              textAlign: "left",
              padding: 10,
              borderRadius: 10,
              border: "1px solid #ddd",
              background: "white",
              cursor: "pointer",
            }}
          >
            <div style={{ fontWeight: 650 }}>Case #{c.index} - {c.mutation}</div>
            <div
              style={{
                opacity: 0.8,
                fontSize: 12,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {c.prompt}
            </div>
          </button>
        ))}
        {failed.length === 0 && <div style={{ opacity: 0.7 }}>No failures yet.</div>}
      </div>
    </div>
  );
}

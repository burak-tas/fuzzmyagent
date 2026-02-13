import React from "react";

export function RunDashboard(props: { run: any | null }) {
  if (!props.run) return null;

  const { status, progress_done, progress_total, id } = props.run;
  const pct = progress_total
    ? Math.round((progress_done / progress_total) * 100)
    : 0;

  return (
    <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ fontWeight: 700 }}>Run: {id}</div>
          <div style={{ opacity: 0.8 }}>Status: {status}</div>
        </div>
        <div style={{ fontWeight: 700 }}>{pct}%</div>
      </div>
      <div
        style={{
          marginTop: 10,
          background: "#f5f5f5",
          borderRadius: 999,
          height: 10,
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: 10,
            borderRadius: 999,
            background: "#111",
          }}
        />
      </div>
      <div style={{ marginTop: 8, opacity: 0.8 }}>
        {progress_done} / {progress_total}
      </div>
    </div>
  );
}

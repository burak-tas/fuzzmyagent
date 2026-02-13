import React from "react";

export function Editor(props: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  height?: number;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontWeight: 700 }}>{props.label}</div>
      <textarea
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
        style={{
          width: "100%",
          height: props.height ?? 220,
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          fontSize: 13,
          padding: 12,
          borderRadius: 10,
          border: "1px solid #ddd",
        }}
      />
    </div>
  );
}

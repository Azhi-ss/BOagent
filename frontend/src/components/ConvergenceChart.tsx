import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartPoint } from "../types";

const TRAD = "#f2a516";
const LLM = "#16d69b";

interface ConvergenceChartProps {
  data: ChartPoint[];
  targetCol: string;
}

interface TooltipPayload {
  name: string;
  value: number;
  color: string;
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: number;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div
      style={{
        background: "var(--color-graphite-850)",
        border: "1px solid var(--color-graphite-600)",
        borderRadius: 8,
        padding: "10px 12px",
        fontFamily: "var(--font-mono)",
        fontSize: 12,
      }}
    >
      <div style={{ color: "var(--color-ink-300)", marginBottom: 6 }}>
        ITERATION {label}
      </div>
      {payload.map((p) => (
        <div
          key={p.name}
          style={{ color: p.color, display: "flex", justifyContent: "space-between", gap: 16 }}
        >
          <span>{p.name}</span>
          <span>{p.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

export function ConvergenceChart({ data, targetCol }: ConvergenceChartProps) {
  return (
    <div style={{ width: "100%", height: 420 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="rgb(54 66 90 / 0.4)" strokeDasharray="3 3" />
          <XAxis
            dataKey="iteration"
            stroke="var(--color-ink-500)"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            label={{
              value: "ITERATION",
              position: "insideBottom",
              offset: -4,
              fill: "var(--color-ink-500)",
              fontSize: 11,
            }}
          />
          <YAxis
            stroke="var(--color-ink-500)"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            label={{
              value: targetCol,
              angle: -90,
              position: "insideLeft",
              fill: "var(--color-ink-500)",
              fontSize: 11,
            }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Traditional BO (amber) */}
          <Line
            type="monotone"
            dataKey="trad_best"
            name="Trad · Best"
            stroke={TRAD}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="trad_gen"
            name="Trad · Gen"
            stroke={TRAD}
            strokeWidth={1.5}
            strokeDasharray="5 4"
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="trad_cand"
            name="Trad · Candidate"
            stroke={TRAD}
            strokeWidth={0}
            dot={{ r: 2.5, fill: TRAD, opacity: 0.5 }}
            isAnimationActive={false}
            connectNulls
          />

          {/* LLMBO (signal green) */}
          <Line
            type="monotone"
            dataKey="llm_best"
            name="LLMBO · Best"
            stroke={LLM}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="llm_gen"
            name="LLMBO · Gen"
            stroke={LLM}
            strokeWidth={1.5}
            strokeDasharray="5 4"
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="llm_cand"
            name="LLMBO · Candidate"
            stroke={LLM}
            strokeWidth={0}
            dot={{ r: 2.5, fill: LLM, opacity: 0.5 }}
            isAnimationActive={false}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ChartLegend() {
  const items = [
    { color: TRAD, label: "Traditional BO", style: "solid" },
    { color: LLM, label: "LLMBO", style: "solid" },
  ];
  return (
    <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "center" }}>
      {items.map((it) => (
        <div key={it.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{ width: 18, height: 3, background: it.color, borderRadius: 2 }}
          />
          <span style={{ fontSize: 12, color: "var(--color-ink-300)" }}>{it.label}</span>
        </div>
      ))}
      <span style={{ fontSize: 11, color: "var(--color-ink-500)", fontFamily: "var(--font-mono)" }}>
        — best · ╌ generalization · ● candidate
      </span>
    </div>
  );
}

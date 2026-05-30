import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
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
  value: number | [number, number];
  color: string;
  dataKey: string;
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
  // Only show the mean lines in the tooltip (skip the band areas).
  const means = payload.filter(
    (p) => p.dataKey === "trad_best_mean" || p.dataKey === "llm_best_mean",
  );
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
        第 {label} 轮迭代 (ITERATION)
      </div>
      {means.map((p) => (
        <div
          key={p.dataKey}
          style={{ color: p.color, display: "flex", justifyContent: "space-between", gap: 16 }}
        >
          <span>{p.name}</span>
          <span>{typeof p.value === "number" ? p.value.toFixed(4) : "—"}</span>
        </div>
      ))}
    </div>
  );
}

export function ConvergenceChart({ data, targetCol }: ConvergenceChartProps) {
  return (
    <div style={{ width: "100%", height: 420 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="rgb(54 66 90 / 0.4)" strokeDasharray="3 3" />
          <XAxis
            dataKey="iteration"
            stroke="var(--color-ink-500)"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            label={{
              value: "迭代轮次 (ITERATION)",
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

          {/* Variance bands (mean ∓ std), rendered first so means sit on top. */}
          <Area
            type="monotone"
            dataKey="trad_best_band"
            name="传统 · ±标准差"
            stroke="none"
            fill={TRAD}
            fillOpacity={0.13}
            isAnimationActive={false}
            connectNulls
            activeDot={false}
          />
          <Area
            type="monotone"
            dataKey="llm_best_band"
            name="LLMBO · ±标准差"
            stroke="none"
            fill={LLM}
            fillOpacity={0.13}
            isAnimationActive={false}
            connectNulls
            activeDot={false}
          />

          {/* Mean best-score lines (the headline comparison). */}
          <Line
            type="monotone"
            dataKey="trad_best_mean"
            name="Traditional BO · 均值"
            stroke={TRAD}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="llm_best_mean"
            name="LLMBO · 均值"
            stroke={LLM}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />

          {/* Generalization mean (dashed, thinner). */}
          <Line
            type="monotone"
            dataKey="trad_gen_mean"
            name="Traditional · 泛化"
            stroke={TRAD}
            strokeWidth={1.25}
            strokeDasharray="5 4"
            strokeOpacity={0.7}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="llm_gen_mean"
            name="LLMBO · 泛化"
            stroke={LLM}
            strokeWidth={1.25}
            strokeDasharray="5 4"
            strokeOpacity={0.7}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ChartLegend() {
  const items = [
    { color: TRAD, label: "Traditional BO" },
    { color: LLM, label: "LLMBO" },
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
        — 均值最优 (mean best) · ▒ ±1 标准差带 (std) · ╌ 泛化 (gen)
      </span>
    </div>
  );
}

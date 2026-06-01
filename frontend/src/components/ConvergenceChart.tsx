import { useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartPoint } from "../types";

const TRAD = "var(--color-amber-500)";
const LLM = "var(--color-signal-500)";

interface ConvergenceChartProps {
  data: ChartPoint[];
  targetCol: string;
}

interface TooltipPayload {
  name: string;
  value: number | [number, number];
  color: string;
  dataKey: string;
  payload: ChartPoint;
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

  const dataPoint = payload[0].payload;
  if (!dataPoint) return null;

  const tradBest = dataPoint.trad_best_mean;
  const tradBand = dataPoint.trad_best_band;
  const tradGen = dataPoint.trad_gen_mean;

  const llmBest = dataPoint.llm_best_mean;
  const llmBand = dataPoint.llm_best_band;
  const llmGen = dataPoint.llm_gen_mean;

  const tradStd = tradBand && tradBest !== undefined ? tradBand[1] - tradBest : 0;
  const llmStd = llmBand && llmBest !== undefined ? llmBand[1] - llmBest : 0;

  return (
    <div
      style={{
        background: "rgba(11, 15, 25, 0.95)",
        backdropFilter: "blur(16px)",
        border: "1px solid rgba(255, 255, 255, 0.12)",
        borderRadius: 12,
        padding: "16px",
        fontFamily: "var(--font-sans)",
        fontSize: 12,
        boxShadow: "0 15px 30px rgba(0, 0, 0, 0.6), inset 0 0 1px 1px rgba(255, 255, 255, 0.05)",
        width: 280,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div style={{ color: "var(--color-ink-500)", fontWeight: 700, fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase", borderBottom: "1px solid rgba(255, 255, 255, 0.08)", paddingBottom: 6 }}>
        第 {label} 轮迭代 (Iteration {label})
      </div>

      {/* Traditional BO */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 700, color: TRAD, fontSize: 11 }}>
          <span style={{ width: 8, height: 2, background: TRAD, borderRadius: 1 }} />
          Traditional BO
        </div>
        <div style={{ paddingLeft: 14, display: "flex", flexDirection: "column", gap: 2, fontSize: 10, color: "var(--color-ink-300)", fontFamily: "var(--font-mono)" }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>最优均值 (Best Mean):</span>
            <span style={{ color: "var(--color-ink-100)", fontWeight: 600 }}>{tradBest !== undefined && tradBest !== null ? tradBest.toFixed(4) : "—"}</span>
          </div>
          {tradStd > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>标准差 (Std Dev):</span>
              <span style={{ color: "var(--color-ink-500)" }}>±{tradStd.toFixed(4)}</span>
            </div>
          )}
          {tradGen !== undefined && tradGen !== null && (
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>测试泛化 (Gen Mean):</span>
              <span style={{ color: "var(--color-ink-300)" }}>{tradGen.toFixed(4)}</span>
            </div>
          )}
        </div>
      </div>

      {/* LLMBO */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 700, color: LLM, fontSize: 11 }}>
          <span style={{ width: 8, height: 2, background: LLM, borderRadius: 1 }} />
          LLMBO
        </div>
        <div style={{ paddingLeft: 14, display: "flex", flexDirection: "column", gap: 2, fontSize: 10, color: "var(--color-ink-300)", fontFamily: "var(--font-mono)" }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>最优均值 (Best Mean):</span>
            <span style={{ color: "var(--color-ink-100)", fontWeight: 600 }}>{llmBest !== undefined && llmBest !== null ? llmBest.toFixed(4) : "—"}</span>
          </div>
          {llmStd > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>标准差 (Std Dev):</span>
              <span style={{ color: "var(--color-ink-500)" }}>±{llmStd.toFixed(4)}</span>
            </div>
          )}
          {llmGen !== undefined && llmGen !== null && (
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>测试泛化 (Gen Mean):</span>
              <span style={{ color: "var(--color-ink-300)" }}>{llmGen.toFixed(4)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function ConvergenceChart({ data, targetCol }: ConvergenceChartProps) {
  const [showTrad, setShowTrad] = useState(true);
  const [showLlm, setShowLlm] = useState(true);

  // Compute dynamic domain for Y-Axis with padding
  const allValues = data
    .flatMap((d) => [
      d.trad_best_mean,
      d.trad_best_band ? d.trad_best_band[0] : null,
      d.trad_best_band ? d.trad_best_band[1] : null,
      d.trad_gen_mean,
      d.llm_best_mean,
      d.llm_best_band ? d.llm_best_band[0] : null,
      d.llm_best_band ? d.llm_best_band[1] : null,
      d.llm_gen_mean,
    ])
    .filter((v): v is number => typeof v === "number" && !isNaN(v));

  let yDomain: [number, number] | ["auto", "auto"] = ["auto", "auto"];
  if (allValues.length > 0) {
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const range = maxVal - minVal;
    const padding = range === 0 ? 0.1 : range * 0.08;
    yDomain = [minVal - padding, maxVal + padding];
  }

  // Calculate baseline from first iteration (usually the initial random points average)
  const baseline =
    data.length > 0
      ? ((data[0].trad_best_mean || 0) + (data[0].llm_best_mean || 0)) / 2
      : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, width: "100%", flex: 1, minHeight: 0 }}>
      {/* Interactive Legend */}
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          gap: 20,
          flexWrap: "wrap",
          alignItems: "center",
          userSelect: "none",
          flexShrink: 0,
        }}
      >
        <div
          onClick={() => setShowTrad(!showTrad)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            cursor: "pointer",
            opacity: showTrad ? 1 : 0.35,
            transition: "opacity 0.2s",
          }}
        >
          <span
            style={{
              width: 14,
              height: 3,
              background: TRAD,
              borderRadius: 9,
              boxShadow: showTrad ? `0 0 6px ${TRAD}80` : "none",
            }}
          />
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: showTrad ? "var(--color-ink-100)" : "var(--color-ink-500)",
              fontFamily: "var(--font-display)",
            }}
          >
            Traditional BO
          </span>
        </div>

        <div
          onClick={() => setShowLlm(!showLlm)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            cursor: "pointer",
            opacity: showLlm ? 1 : 0.35,
            transition: "opacity 0.2s",
          }}
        >
          <span
            style={{
              width: 14,
              height: 3,
              background: LLM,
              borderRadius: 9,
              boxShadow: showLlm ? `0 0 6px ${LLM}80` : "none",
            }}
          />
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: showLlm ? "var(--color-ink-100)" : "var(--color-ink-500)",
              fontFamily: "var(--font-display)",
            }}
          >
            LLMBO
          </span>
        </div>

        <span style={{ fontSize: 10, color: "var(--color-ink-500)", fontFamily: "var(--font-sans)" }}>
          — 均值最优 · ▒ ±1 标准差 · ╌ 泛化 (点击图例切换显示)
        </span>
      </div>

      {/* Chart container */}
      <div style={{ width: "100%", height: 400 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 16, right: 12, bottom: 8, left: -24 }}>
            <CartesianGrid stroke="rgba(255, 255, 255, 0.03)" strokeDasharray="4 4" />
            <XAxis
              dataKey="iteration"
              stroke="var(--color-ink-500)"
              tickLine={false}
              axisLine={{ stroke: "rgba(255, 255, 255, 0.08)" }}
              tick={{ fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-ink-500)" }}
              label={{
                value: "迭代轮次 (ITERATIONS)",
                position: "insideBottom",
                offset: -4,
                fill: "var(--color-ink-500)",
                fontSize: 10,
                fontFamily: "var(--font-display)",
                fontWeight: 600,
                letterSpacing: "0.05em",
              }}
            />
            <YAxis
              width={60}
              stroke="var(--color-ink-500)"
              tickLine={false}
              axisLine={{ stroke: "rgba(255, 255, 255, 0.08)" }}
              tick={{ fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-ink-500)" }}
              domain={yDomain}
              label={{
                value: targetCol,
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fill: "var(--color-ink-500)",
                fontSize: 10,
                fontFamily: "var(--font-display)",
                fontWeight: 600,
                letterSpacing: "0.05em",
              }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: "rgba(255, 255, 255, 0.1)", strokeDasharray: "3 3" }} />

            {/* Baseline Reference Line */}
            {baseline !== null && (
              <ReferenceLine
                y={baseline}
                stroke="rgba(255, 255, 255, 0.15)"
                strokeDasharray="4 4"
                label={{
                  value: "随机初始基准 (Baseline)",
                  position: "insideBottomLeft",
                  fill: "var(--color-ink-500)",
                  fontSize: 9,
                  fontFamily: "var(--font-mono)",
                  fontWeight: 500,
                  offset: 8,
                }}
              />
            )}

            {/* Variance bands (mean ∓ std) */}
            {showTrad && (
              <Area
                type="monotone"
                dataKey="trad_best_band"
                name="传统 · ±标准差"
                stroke="none"
                fill={TRAD}
                fillOpacity={0.07}
                isAnimationActive={false}
                connectNulls
                activeDot={false}
              />
            )}
            {showLlm && (
              <Area
                type="monotone"
                dataKey="llm_best_band"
                name="LLMBO · ±标准差"
                stroke="none"
                fill={LLM}
                fillOpacity={0.07}
                isAnimationActive={false}
                connectNulls
                activeDot={false}
              />
            )}

            {/* Mean best-score lines */}
            {showTrad && (
              <Line
                type="monotone"
                dataKey="trad_best_mean"
                name="Traditional BO · 均值"
                stroke={TRAD}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 0, fill: TRAD }}
                isAnimationActive={false}
                connectNulls
              />
            )}
            {showLlm && (
              <Line
                type="monotone"
                dataKey="llm_best_mean"
                name="LLMBO · 均值"
                stroke={LLM}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 0, fill: LLM }}
                isAnimationActive={false}
                connectNulls
              />
            )}

            {/* Generalization mean */}
            {showTrad && (
              <Line
                type="monotone"
                dataKey="trad_gen_mean"
                name="Traditional · 泛化"
                stroke={TRAD}
                strokeWidth={1.25}
                strokeDasharray="4 4"
                strokeOpacity={0.5}
                dot={false}
                activeDot={false}
                isAnimationActive={false}
                connectNulls
              />
            )}
            {showLlm && (
              <Line
                type="monotone"
                dataKey="llm_gen_mean"
                name="LLMBO · 泛化"
                stroke={LLM}
                strokeWidth={1.25}
                strokeDasharray="4 4"
                strokeOpacity={0.5}
                dot={false}
                activeDot={false}
                isAnimationActive={false}
                connectNulls
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}


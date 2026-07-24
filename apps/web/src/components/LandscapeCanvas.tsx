import { useMemo, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { OperationalVariable, OperationalObservation } from "../types";

const LLM = "var(--color-signal-500)"; // Emerald-500
const BEST = "var(--color-amber-500)"; // Amber-500
const OBSERVED = "var(--color-ink-500)"; // Slate-500

interface LandscapeCanvasProps {
  variables: OperationalVariable[];
  history: OperationalObservation[];
  suggestions: Record<string, number>[];
  selectedSuggestion?: Record<string, number> | null;
}

export function LandscapeCanvas({
  variables,
  history,
  suggestions,
  selectedSuggestion,
}: LandscapeCanvasProps) {
  const [xIdx, setXIdx] = useState(0);
  const [yIdx, setYIdx] = useState(variables.length > 1 ? 1 : 0);

  // Axis selection
  const xVar = variables[xIdx];
  const yVar = variables[yIdx];

  const chartData = useMemo(() => {
    if (!xVar || !yVar) return { observed: [], candidates: [], bestPoint: null };

    // Shadow Projection: We take all points and just grab the current X/Y dimensions
    const observed = history.map((obs, i) => ({
      x: obs.config[xVar.name],
      y: obs.config[yVar.name],
      score: obs.score,
      id: `Obs ${i + 1}`,
      type: "observed",
      fullConfig: obs.config
    }));

    const candidates = suggestions.map((s, i) => ({
      x: s[xVar.name],
      y: s[yVar.name],
      id: i === 0 ? "Top Suggestion" : `Option ${i + 1}`,
      type: "candidate",
      isSelected: JSON.stringify(s) === JSON.stringify(selectedSuggestion),
      fullConfig: s
    }));

    const best = history.length > 0 
      ? history.reduce((prev, curr) => (prev.score > curr.score ? prev : curr))
      : null;

    const bestPoint = best ? {
      x: best.config[xVar.name],
      y: best.config[yVar.name],
      score: best.score,
      id: "Current Best",
      type: "best",
      fullConfig: best.config
    } : null;

    return { observed, candidates, bestPoint };
  }, [xVar, yVar, history, suggestions, selectedSuggestion]);

  if (!xVar || !yVar) {
    return (
      <div className="panel" style={{ height: 300, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-ink-500)", fontSize: 12 }}>
        请定义变量以开启可视化空间
      </div>
    );
  }

  return (
    <section className="panel" style={{ padding: 24, position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: LLM, fontFamily: "var(--font-display)", letterSpacing: "0.02em" }}>
            优化地形投影 OPTIMIZATION LANDSCAPE
          </h3>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
             <select 
               className="field-input" 
               style={{ width: "auto", height: 24, padding: "0 24px 0 8px", fontSize: 10, borderRadius: 6, background: "rgba(255,255,255,0.05)" }}
               value={xIdx}
               onChange={e => setXIdx(Number(e.target.value))}
             >
               {variables.map((v, i) => <option key={i} value={i}>X: {v.name}</option>)}
             </select>
             <span style={{ color: "var(--color-ink-500)", fontSize: 10 }}>vs</span>
             <select 
               className="field-input" 
               style={{ width: "auto", height: 24, padding: "0 24px 0 8px", fontSize: 10, borderRadius: 6, background: "rgba(255,255,255,0.05)" }}
               value={yIdx}
               onChange={e => setYIdx(Number(e.target.value))}
             >
               {variables.map((v, i) => <option key={i} value={i}>Y: {v.name}</option>)}
             </select>
          </div>
        </div>
        
        <div style={{ display: "flex", gap: 12, fontSize: 10, color: "var(--color-ink-500)", fontWeight: 600, fontFamily: "var(--font-display)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: OBSERVED }} /> 已观测
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: LLM }} /> 建议中
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: BEST }} /> 当前最佳
          </div>
        </div>
      </div>

      <div style={{ width: "100%", height: 320, background: "rgba(0,0,0,0.25)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.03)", padding: "10px 0" }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis 
              type="number" 
              dataKey="x" 
              name={xVar.name} 
              unit={xVar.unit ? ` ${xVar.unit}` : ""} 
              domain={[xVar.min, xVar.max]}
              stroke="var(--color-ink-500)"
              fontSize={10}
              tick={{ fill: "var(--color-ink-500)" }}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name={yVar.name} 
              unit={yVar.unit ? ` ${yVar.unit}` : ""} 
              domain={[yVar.min, yVar.max]}
              stroke="var(--color-ink-500)"
              fontSize={10}
              tick={{ fill: "var(--color-ink-500)" }}
            />
            <ZAxis type="number" range={[50, 400]} />
            <Tooltip 
              cursor={{ strokeDasharray: "3 3", stroke: "rgba(255,255,255,0.2)" }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  const typeColorMap: Record<string, string> = {
                    candidate: LLM,
                    best: BEST
                  };
                  const labelColor = typeColorMap[data.type] || "var(--color-ink-300)";

                  return (
                    <div style={{ background: "rgba(11, 15, 25, 0.95)", border: "1px solid rgba(255,255,255,0.12)", padding: "12px 16px", borderRadius: 12, fontSize: 11, backdropFilter: "blur(16px)", boxShadow: "0 10px 20px rgba(0,0,0,0.5)" }}>
                      <div style={{ fontWeight: 700, color: labelColor, marginBottom: 8, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                        {data.id}
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        {Object.entries(data.fullConfig).map(([k, v]) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 20, fontFamily: "var(--font-mono)" }}>
                            <span style={{ color: "var(--color-ink-500)" }}>{k}:</span>
                            <span style={{ color: "var(--color-ink-100)" }}>
                              {typeof v === "number" ? v.toFixed(3) : String(v)}
                            </span>
                          </div>
                        ))}
                      </div>
                      {data.score !== undefined && (
                        <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.08)", color: LLM, fontWeight: 700, display: "flex", justifyContent: "space-between" }}>
                          <span>SCORE:</span>
                          <span>{data.score.toFixed(4)}</span>
                        </div>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
            
            {/* Observed Points (Shadow Projection) */}
            <Scatter name="Observed" data={chartData.observed} fill={OBSERVED} fillOpacity={0.6} isAnimationActive={false}>
              {chartData.observed.map((entry, index) => (
                <Cell key={`cell-obs-${index}`} stroke="rgba(255,255,255,0.2)" />
              ))}
            </Scatter>

            {/* Candidate Points */}
            <Scatter name="Candidates" data={chartData.candidates} isAnimationActive={false}>
              {chartData.candidates.map((entry, index) => (
                <Cell 
                  key={`cell-can-${index}`} 
                  fill={entry.isSelected ? "#fff" : LLM} 
                  stroke={LLM}
                  strokeWidth={entry.isSelected ? 2 : 1}
                  fillOpacity={entry.isSelected ? 1 : 0.4}
                  style={{ filter: entry.isSelected ? `drop-shadow(0 0 8px ${LLM})` : "none" }}
                />
              ))}
            </Scatter>

            {/* Best Point Highlight */}
            {chartData.bestPoint && (
              <Scatter name="Best" data={[chartData.bestPoint]} fill={BEST} isAnimationActive={false}>
                <Cell stroke="#fff" strokeWidth={2} style={{ filter: `drop-shadow(0 0 10px ${BEST})` }} />
              </Scatter>
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      
      <div style={{ marginTop: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 10, color: "var(--color-ink-500)", display: "flex", gap: 12 }}>
          <span>投影模式: 影子叠加 (Shadow Projection)</span>
          <span>维数: {variables.length}D</span>
        </div>
        <div style={{ fontSize: 10, color: "var(--color-ink-500)", fontStyle: "italic" }}>
          提示：点击轴标签或上方下拉框切换观察维度
        </div>
      </div>
    </section>
  );
}

import { useState, useCallback, useMemo } from "react";
import { getOperationalSuggestion } from "./lib/api";
import { Field, NumberField } from "./components/Field";
import type { 
  OperationalVariable, 
  OperationalObservation, 
  LLMBOConfig,
  OperationalSuggestResponse
} from "./types";

const LLM = "#16d69b";

export function OperationalMode() {
  const [target, setTarget] = useState("Power Conversion Efficiency");
  const [variables, setVariables] = useState<OperationalVariable[]>([
    { name: "Temperature", min: 100, max: 200, unit: "°C" },
    { name: "Concentration", min: 0, max: 1, unit: "M" }
  ]);
  const [history, setHistory] = useState<OperationalObservation[]>([]);
  const [llmConfig, setLlmConfig] = useState<LLMBOConfig>({
    acquisition: "ucb",
    xi: 0.01,
    kappa: 2.576,
    n_candidates: 3,
    n_templates: 2,
    top_k: 20,
    alpha: 0.1,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OperationalSuggestResponse | null>(null);
  const [error, setError] = useState("");
  
  // Form for new observation
  const [inputConfig, setInputConfig] = useState<Record<string, number>>({});
  const [inputScore, setInputScore] = useState<number | "">("");

  const addVariable = () => {
    setVariables([...variables, { name: `Var ${variables.length + 1}`, min: 0, max: 100, unit: "" }]);
  };

  const removeVariable = (index: number) => {
    setVariables(variables.filter((_, i) => i !== index));
  };

  const updateVariable = (index: number, field: keyof OperationalVariable, value: any) => {
    const next = [...variables];
    next[index] = { ...next[index], [field]: value };
    setVariables(next);
  };

  const handleSuggest = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await getOperationalSuggestion({
        target,
        variables,
        history,
        llm_config: llmConfig,
        seed: Math.floor(Math.random() * 1000)
      });
      setResult(res);
      if (res.suggestions.length > 0) {
        setInputConfig(res.suggestions[0]);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const addObservation = () => {
    if (inputScore === "") return;
    const obs: OperationalObservation = {
      config: { ...inputConfig },
      score: Number(inputScore)
    };
    setHistory([...history, obs]);
    setInputScore("");
    setResult(null); // Clear suggestion once acted upon
  };

  return (
    <div className="fade-rise" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
      {/* Left Column: Config & History */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Search Space Config */}
        <section className="panel" style={{ padding: 20 }}>
          <div style={{ marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: LLM }}>1 · 搜索空间 (SEARCH SPACE)</h3>
            <button 
              onClick={addVariable}
              style={{ fontSize: 11, background: "var(--color-graphite-700)", border: "none", borderRadius: 4, padding: "4px 8px" }}
            >
              + 添加变量 (ADD VARIABLE)
            </button>
          </div>
          
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Field label="优化目标 (Optimization Target)">
              <input 
                className="field-input" 
                value={target} 
                onChange={e => setTarget(e.target.value)}
                placeholder="e.g. Power Conversion Efficiency (光电转换效率)"
              />
            </Field>
            
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 40px", gap: 8, fontSize: 10, color: "var(--color-ink-500)", fontWeight: 700 }}>
                <span>名称 (NAME)</span>
                <span>最小值 (MIN)</span>
                <span>最大值 (MAX)</span>
                <span>单位 (UNIT)</span>
                <span></span>
              </div>
              {variables.map((v, i) => (
                <div key={i} style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 40px", gap: 8, alignItems: "center" }}>
                  <input 
                    className="field-input" 
                    value={v.name} 
                    onChange={e => updateVariable(i, "name", e.target.value)} 
                  />
                  <input 
                    className="field-input" 
                    type="number"
                    value={v.min} 
                    onChange={e => updateVariable(i, "min", Number(e.target.value))} 
                  />
                  <input 
                    className="field-input" 
                    type="number"
                    value={v.max} 
                    onChange={e => updateVariable(i, "max", Number(e.target.value))} 
                  />
                  <input 
                    className="field-input" 
                    value={v.unit} 
                    onChange={e => updateVariable(i, "unit", e.target.value)} 
                  />
                  <button 
                    onClick={() => removeVariable(i)}
                    style={{ background: "none", border: "none", color: "var(--color-fault-400)", cursor: "pointer" }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* History */}
        <section className="panel" style={{ padding: 20, flex: 1 }}>
          <h3 style={{ marginTop: 0, marginBottom: 16, fontSize: 14, fontWeight: 700, color: LLM }}>2 · 观测历史 (OBSERVATION HISTORY)</h3>
          {history.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px 0", color: "var(--color-ink-500)", fontSize: 12, fontStyle: "italic" }}>
              No observation data. Add your first data point or consult the Agent for suggestions.
              <br />(暂无观测数据。请添加实验点或咨询 Agent 获取建议。)
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--color-ink-500)", borderBottom: "1px solid var(--color-graphite-700)" }}>
                    <th style={{ padding: "8px 4px" }}>#</th>
                    {variables.map(v => <th key={v.name} style={{ padding: "8px 4px" }}>{v.name}</th>)}
                    <th style={{ padding: "8px 4px", textAlign: "right" }}>{target}</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((obs, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid var(--color-graphite-850)" }}>
                      <td style={{ padding: "8px 4px", color: "var(--color-ink-500)" }}>{i + 1}</td>
                      {variables.map(v => (
                        <td key={v.name} style={{ padding: "8px 4px", fontFamily: "var(--font-mono)" }}>
                          {obs.config[v.name]?.toFixed(3) ?? "—"}
                        </td>
                      ))}
                      <td style={{ padding: "8px 4px", textAlign: "right", fontFamily: "var(--font-mono)", fontWeight: 700, color: LLM }}>
                        {obs.score.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px dashed var(--color-graphite-700)" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "var(--color-ink-500)", marginBottom: 10 }}>手动录入数据 (MANUAL DATA ENTRY)</div>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${variables.length}, 1fr) 100px auto`, gap: 8, alignItems: "end" }}>
               {variables.map(v => (
                 <Field key={v.name} label={v.name}>
                   <input 
                    className="field-input" 
                    type="number"
                    value={inputConfig[v.name] ?? ""} 
                    onChange={e => setInputConfig({...inputConfig, [v.name]: Number(e.target.value)})}
                   />
                 </Field>
               ))}
               <Field label={target}>
                 <input 
                  className="field-input" 
                  type="number" 
                  value={inputScore}
                  onChange={e => setInputScore(e.target.value === "" ? "" : Number(e.target.value))}
                 />
               </Field>
               <button 
                className="run-btn" 
                onClick={addObservation}
                disabled={inputScore === ""}
                style={{ padding: "8px 12px", fontSize: 12, background: LLM, color: "#07090d" }}
               >
                添加 (ADD)
               </button>
            </div>
          </div>
        </section>
      </div>

      {/* Right Column: Suggestions & Reasoning */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div className="panel" style={{ padding: 24, textAlign: "center" }}>
          <button 
            className="run-btn"
            onClick={handleSuggest}
            disabled={loading}
            style={{ 
              width: "100%", 
              fontSize: 16, 
              background: `linear-gradient(90deg, #16d69b, #3ef0b0)`, 
              color: "#07090d",
              boxShadow: loading ? "none" : `0 0 20px rgba(22, 214, 155, 0.3)`
            }}
          >
            {loading ? "Consulting AGENT..." : "Consult AGENT for Next Step (咨询下一步方案)"}
          </button>
          {error && <div style={{ color: "var(--color-fault-400)", fontSize: 12, marginTop: 12 }}>⚠ {error}</div>}
        </div>

        {result && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
             {/* Reasoning Card */}
             <section className="panel" style={{ padding: 20 }}>
                <h3 style={{ marginTop: 0, marginBottom: 12, fontSize: 14, fontWeight: 700, color: LLM }}>AGENT 推理分析 (REASONING)</h3>
                <div 
                  style={{ 
                    fontSize: 13, 
                    lineHeight: 1.6, 
                    color: "var(--color-ink-100)", 
                    whiteSpace: "pre-wrap",
                    maxHeight: 400,
                    overflowY: "auto",
                    padding: 16,
                    background: "rgba(0,0,0,0.2)",
                    borderRadius: 8,
                    border: "1px solid var(--color-graphite-700)"
                  }}
                >
                  {result.analysis}
                </div>
             </section>

             {/* Suggestions List */}
             <section className="panel" style={{ padding: 20 }}>
                <h3 style={{ marginTop: 0, marginBottom: 16, fontSize: 14, fontWeight: 700, color: LLM }}>建议配方 (SUGGESTED FORMULATIONS)</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {result.suggestions.map((s, i) => (
                    <div 
                      key={i} 
                      onClick={() => setInputConfig(s)}
                      style={{ 
                        padding: 14, 
                        background: inputConfig === s ? "rgba(22, 214, 155, 0.1)" : "var(--color-graphite-880)",
                        border: `1px solid ${inputConfig === s ? LLM : "var(--color-graphite-700)"}`,
                        borderRadius: 10,
                        cursor: "pointer",
                        transition: "all 0.2s"
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: i === 0 ? LLM : "var(--color-ink-500)" }}>
                          {i === 0 ? "★ 首选推荐 (TOP RECOMMENDATION)" : `备选方案 (OPTION) ${i + 1}`}
                        </span>
                        {inputConfig === s && <span style={{ fontSize: 10, color: LLM }}>已选 (SELECTED)</span>}
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                        {Object.entries(s).map(([k, v]) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "var(--font-mono)" }}>
                            <span style={{ color: "var(--color-ink-500)" }}>{k}:</span>
                            <span>{v.toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
             </section>
          </div>
        )}
      </div>
    </div>
  );
}

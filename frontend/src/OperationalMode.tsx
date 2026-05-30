import { useState, useCallback } from "react";
import { getOperationalSuggestion } from "./lib/api";
import { Field, NumberField, AcqSelect } from "./components/Field";
import { AcquisitionConfig } from "./components/AcquisitionConfig";
import { PlusIcon, TrashIcon, SparklesIcon } from "./components/Icons";
import { LandscapeCanvas } from "./components/LandscapeCanvas";
import type { 
  OperationalVariable, 
  OperationalObservation, 
  LLMBOConfig,
  OperationalSuggestResponse
} from "./types";

const LLM = "#10b981"; // Emerald-500

function SuggestionCard({ 
  suggestion, 
  index, 
  isSelected, 
  onClick 
}: { 
  suggestion: Record<string, number>; 
  index: number; 
  isSelected: boolean; 
  onClick: () => void;
}) {
  return (
    <div 
      onClick={onClick}
      style={{ 
        padding: "16px 18px", 
        background: isSelected ? "rgba(16, 185, 129, 0.08)" : "rgba(15, 23, 42, 0.3)",
        border: `1px solid ${isSelected ? LLM : "rgba(255, 255, 255, 0.06)"}`,
        borderRadius: 12,
        cursor: "pointer",
        transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
        boxShadow: isSelected ? `0 0 15px -3px ${LLM}20` : "none"
      }}
      onMouseEnter={e => {
        if (!isSelected) e.currentTarget.style.borderColor = "rgba(255,255,255,0.15)";
        e.currentTarget.style.transform = "translateY(-1px)";
      }}
      onMouseLeave={e => {
        if (!isSelected) e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)";
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: index === 0 ? LLM : "var(--color-ink-500)", fontFamily: "var(--font-display)", letterSpacing: "0.05em" }}>
          {index === 0 ? "★ 首选建议配方 (TOP RECOMMENDATION)" : `备选参考配方 (OPTION) ${index + 1}`}
        </span>
        {isSelected && (
          <span style={{ fontSize: 10, color: LLM, fontWeight: 700, fontFamily: "var(--font-display)", letterSpacing: "0.05em" }}>
            已选用 (SELECTED)
          </span>
        )}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {Object.entries(suggestion).map(([k, v]) => (
          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, fontFamily: "var(--font-mono)" }}>
            <span style={{ color: "var(--color-ink-500)" }}>{k}:</span>
            <span style={{ color: "var(--color-ink-100)", fontWeight: 600 }}>{v.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

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
    chat_engine: "deepseek-v4-flash",
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

  const removeObservation = (index: number) => {
    setHistory(history.filter((_, i) => i !== index));
  };

  const clearHistory = () => {
    if (window.confirm("确认清空所有历史数据？ (Are you sure you want to clear all history?)")) {
      setHistory([]);
      setResult(null);
    }
  };

  return (
    <div className="fade-rise" style={{ display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 24 }}>
      {/* Left Column: Config & History */}
      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        
        {/* Search Space Config */}
        <section className="panel" style={{ padding: 24 }}>
          <div style={{ marginBottom: 20, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: LLM, fontFamily: "var(--font-display)", letterSpacing: "0.02em" }}>
              1 · 实验空间设计 SEARCH SPACE
            </h3>
            <button 
              onClick={addVariable}
              style={{ 
                fontSize: 11, 
                background: "rgba(255, 255, 255, 0.05)", 
                border: "1px solid rgba(255, 255, 255, 0.1)", 
                borderRadius: 6, 
                padding: "6px 12px",
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                fontWeight: 600,
                color: "var(--color-ink-300)"
              }}
            >
              <PlusIcon size={12} />
              <span>添加变量 ADD</span>
            </button>
          </div>
          
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <Field label="优化目标" subLabel="Optimization Target">
              <input 
                className="field-input" 
                value={target} 
                onChange={e => setTarget(e.target.value)}
                placeholder="e.g. Power Conversion Efficiency"
              />
            </Field>
            
            <div className="sub-panel">
              <span className="sub-panel-title">变量定义 VARIABLES DEFINITION</span>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1.2fr 40px", gap: 8, fontSize: 10, color: "var(--color-ink-500)", fontWeight: 700, fontFamily: "var(--font-display)", letterSpacing: "0.04em" }}>
                  <span>名称 NAME</span>
                  <span>最小 MIN</span>
                  <span>最大 MAX</span>
                  <span>单位 UNIT</span>
                  <span></span>
                </div>
                {variables.map((v, i) => (
                  <div key={i} style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1.2fr 40px", gap: 8, alignItems: "center" }}>
                    <input className="field-input" value={v.name} onChange={e => updateVariable(i, "name", e.target.value)} />
                    <input className="field-input" type="number" value={v.min} onChange={e => updateVariable(i, "min", Number(e.target.value))} />
                    <input className="field-input" type="number" value={v.max} onChange={e => updateVariable(i, "max", Number(e.target.value))} />
                    <input className="field-input" value={v.unit} placeholder="—" onChange={e => updateVariable(i, "unit", e.target.value)} />
                    <button 
                      onClick={() => removeVariable(i)}
                      style={{ background: "none", border: "none", color: "var(--color-fault-400)", cursor: "pointer", opacity: 0.7 }}
                    >
                      <TrashIcon size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* History */}
        <section className="panel" style={{ padding: 24, display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: LLM, fontFamily: "var(--font-display)", letterSpacing: "0.02em" }}>
              2 · 实验观测历史 OBSERVATIONS
            </h3>
            {history.length > 0 && (
              <button 
                onClick={clearHistory}
                style={{ 
                  fontSize: 11, 
                  background: "rgba(244, 63, 94, 0.1)", 
                  border: "1px solid rgba(244, 63, 94, 0.2)", 
                  borderRadius: 6, 
                  padding: "6px 12px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  fontWeight: 600,
                  color: "var(--color-fault-400)",
                  cursor: "pointer"
                }}
              >
                <TrashIcon size={12} />
                <span>清空 CLEAR</span>
              </button>
            )}
          </div>
          
          <div style={{ overflowX: "auto", border: "1px solid rgba(255, 255, 255, 0.05)", borderRadius: 12, background: "rgba(0,0,0,0.1)", marginBottom: 20 }}>
            {history.length === 0 ? (
              <div style={{ textAlign: "center", padding: "48px 0", color: "var(--color-ink-500)", fontSize: 12, fontStyle: "italic" }}>
                暂无数据。录入实测点或咨询 Agent。
              </div>
            ) : (
              <table className="sci-table" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={{ width: 50 }}>#</th>
                    {variables.map(v => <th key={v.name}>{v.name}</th>)}
                    <th style={{ textAlign: "right" }}>{target}</th>
                    <th style={{ width: 40 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((obs, i) => (
                    <tr key={i}>
                      <td style={{ color: "var(--color-ink-500)", fontWeight: 600 }}>{i + 1}</td>
                      {variables.map(v => (
                        <td key={v.name} style={{ fontFamily: "var(--font-mono)" }}>
                          {obs.config[v.name]?.toFixed(3) ?? "—"}
                        </td>
                      ))}
                      <td style={{ textAlign: "right", fontFamily: "var(--font-mono)", fontWeight: 700, color: LLM }}>
                        {obs.score.toFixed(4)}
                      </td>
                      <td style={{ textAlign: "center" }}>
                        <button 
                          onClick={() => removeObservation(i)}
                          style={{ background: "none", border: "none", color: "var(--color-ink-500)", cursor: "pointer", opacity: 0.7 }}
                          onMouseEnter={e => e.currentTarget.style.opacity = '1'}
                          onMouseLeave={e => e.currentTarget.style.opacity = '0.7'}
                        >
                          <TrashIcon size={14} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          
          <div className="sub-panel">
            <span className="sub-panel-title">录入新观测点 MANUAL ENTRY</span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "flex-end" }}>
               {variables.map(v => (
                 <Field key={v.name} label={v.name} subLabel={v.unit || "Value"}>
                   <NumberField 
                     value={inputConfig[v.name] ?? 0} 
                     onChange={val => setInputConfig({...inputConfig, [v.name]: val})}
                     width={100}
                   />
                 </Field>
               ))}
               <Field label="实测得分" subLabel={target}>
                 <NumberField value={Number(inputScore)} onChange={val => setInputScore(val)} width={120} />
               </Field>
               <button 
                 className="run-btn" 
                 onClick={addObservation}
                 disabled={inputScore === ""}
                 style={{ padding: "10px 20px", background: LLM, color: "#020617", height: 42 }}
               >
                 <PlusIcon size={12} />
                 <span>录入 ADD</span>
               </button>
            </div>
          </div>
        </section>
      </div>

      {/* Right Column: Suggestions & Reasoning */}
      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

        {/* Landscape Visualization */}
        <LandscapeCanvas 
          variables={variables}
          history={history}
          suggestions={result?.suggestions ?? []}
          selectedSuggestion={inputConfig}
        />

        {/* Agent Config (Visible now) */}
        <section className="panel" style={{ padding: 24 }}>
          <h3 style={{ marginTop: 0, marginBottom: 20, fontSize: 14, fontWeight: 700, color: LLM, fontFamily: "var(--font-display)", letterSpacing: "0.02em" }}>
            3 · 决策引擎配置 AGENT CONFIG
          </h3>
          
          <div style={{ marginBottom: 24 }}>
            <AcquisitionConfig config={llmConfig} onChange={v => setLlmConfig(prev => ({...prev, ...v}))} accent={LLM} />
          </div>

          <div className="sub-panel">
            <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
              <Field label="模型引擎" subLabel="Model">
                <select 
                  className="field-input" 
                  style={{ width: 140, padding: '0 8px' }}
                  value={llmConfig.chat_engine} 
                  onChange={e => setLlmConfig({...llmConfig, chat_engine: e.target.value})}
                >
                  <option value="deepseek-v4-flash">DeepSeek V4</option>
                </select>
              </Field>

              <Field label="权衡系数" subLabel="Alpha α">
                <NumberField value={llmConfig.alpha} onChange={v => setLlmConfig({...llmConfig, alpha: v})} step={0.1} width={80} />
              </Field>
              <Field label="筛选规模" subLabel="Top-K">
                <NumberField value={llmConfig.top_k} onChange={v => setLlmConfig({...llmConfig, top_k: v})} width={80} />
              </Field>
              <Field label="建议数量" subLabel="Candidates">
                <NumberField value={llmConfig.n_candidates} onChange={v => setLlmConfig({...llmConfig, n_candidates: v})} width={80} />
              </Field>
            </div>
          </div>

          <button 
            className="run-btn"
            onClick={handleSuggest}
            disabled={loading}
            style={{ 
              width: "100%", 
              marginTop: 20,
              fontSize: 15, 
              background: `linear-gradient(135deg, ${LLM}, #3ef0b0)`, 
              color: "#020617",
              fontWeight: 800
            }}
          >
            <SparklesIcon size={16} />
            <span>{loading ? "正在计算..." : "咨询 AGENT 获取下一步方案"}</span>
          </button>
        </section>

        {/* Suggestion Result panels */}
        {result && (
          <div className="fade-rise" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
             
             {/* Reasoning Card */}
             <section className="panel" style={{ padding: 24 }}>
                <h3 style={{ marginTop: 0, marginBottom: 16, fontSize: 14, fontWeight: 700, color: LLM, fontFamily: "var(--font-display)", letterSpacing: "0.02em" }}>
                  AGENT 优化推理决策 (REASONING)
                </h3>
                <div 
                  style={{ 
                    fontSize: 13, 
                    lineHeight: 1.6, 
                    color: "var(--color-ink-100)", 
                    whiteSpace: "pre-wrap",
                    maxHeight: 350,
                    overflowY: "auto",
                    padding: 16,
                    background: "rgba(0,0,0,0.25)",
                    borderRadius: 12,
                    border: "1px solid rgba(255, 255, 255, 0.05)",
                    fontFamily: "var(--font-sans)"
                  }}
                >
                  {result.analysis}
                </div>
             </section>

             {/* Suggestions List */}
             <section className="panel" style={{ padding: 24 }}>
                <h3 style={{ marginTop: 0, marginBottom: 18, fontSize: 14, fontWeight: 700, color: LLM, fontFamily: "var(--font-display)", letterSpacing: "0.02em" }}>
                  建议配方 SUGGESTIONS
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  {result.suggestions.map((s, i) => (
                    <SuggestionCard 
                      key={i}
                      suggestion={s}
                      index={i}
                      isSelected={JSON.stringify(inputConfig) === JSON.stringify(s)}
                      onClick={() => setInputConfig(s)}
                    />
                  ))}
                </div>
             </section>
          </div>
        )}
      </div>
    </div>
  );
}

import type { AcquisitionType } from "../types";

interface FieldProps {
  label: string;
  subLabel?: string;
  hint?: string;
  children: React.ReactNode;
}

export function Field({ label, subLabel, hint, children }: FieldProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <div style={{ display: "flex", flexDirection: "column" }}>
        <label className="field-label" style={{ marginBottom: 0 }}>{label}</label>
        {subLabel && (
          <span style={{ fontSize: 10, color: "var(--color-ink-500)", fontWeight: 500, letterSpacing: "0.02em", marginTop: 1 }}>
            {subLabel}
          </span>
        )}
      </div>
      {children}
      {hint && (
        <p style={{ margin: "2px 0 0", fontSize: 11, color: "var(--color-ink-500)", opacity: 0.8 }}>{hint}</p>
      )}
    </div>
  );
}

const ACQ_DETAILS: Record<AcquisitionType, {
  label: string;
  nameCn: string;
  formula: string;
  description: string;
  tradeoff: { exploration: number; exploitation: number; label: string };
  scenarios: string[];
}> = {
  ei: {
    label: "Expected Improvement (EI)",
    nameCn: "期望改善",
    formula: "EI(x) = E[max(0, f(x) - f(x⁺))]",
    description: "计算目标函数在当前最佳值 x⁺ 基础上的期望改善量。通过评估可能取得的最大进展，自然平衡了探索与开发。",
    tradeoff: { exploration: 5, exploitation: 5, label: "均衡 (Balanced)" },
    scenarios: [
      "适用于大部分贝叶斯优化首选场景",
      "目标函数连续且噪声低时效果最佳",
      "智能平衡未知发掘与已知高值利用",
    ],
  },
  ucb: {
    label: "Upper Confidence Bound (UCB)",
    nameCn: "上置信界",
    formula: "UCB(x) = μ(x) + κ · σ(x)",
    description: "采用“面对不确定性时的乐观主义”。由预测均值 μ(x)（开发）加上预测标准差 σ(x)（探索）的 κ 倍组成，参数 κ 可控制偏向。",
    tradeoff: { exploration: 7, exploitation: 3, label: "偏向探索 / 可灵活调节 (Exploration-biased)" },
    scenarios: [
      "适合需要显式控制探索强度的项目",
      "防范陷入局部最优，发掘全局最值",
      "实验数据带有明显高噪声时",
    ],
  },
  pi: {
    label: "Probability of Improvement (PI)",
    nameCn: "改善概率",
    formula: "PI(x) = P(f(x) ≥ f(x⁺) + ξ)",
    description: "仅计算新评估点能够超越当前最佳值加上微小余长 ξ 的概率。相比 EI，它只关注超越概率而不关注超越幅度的多少。",
    tradeoff: { exploration: 2, exploitation: 8, label: "强开发性 (Exploitation-biased)" },
    scenarios: [
      "已知目标值临界点，需尽快突破阈值",
      "已知最优点周围的精细化局部寻优",
      "配合适当的偏置参数 ξ 快速收敛",
    ],
  },
};

interface AcqSelectProps {
  value: AcquisitionType;
  onChange: (v: AcquisitionType) => void;
  accent: string;
}

export function AcqSelect({ value, onChange, accent }: AcqSelectProps) {
  return (
    <div className="acq-segment-container" style={{ borderColor: accent }}>
      {Object.entries(ACQ_DETAILS).map(([key, details]) => {
        const typeKey = key as AcquisitionType;
        const isSelected = value === typeKey;
        return (
          <div key={typeKey} className="acq-segment-item">
            <button
              type="button"
              className={`acq-segment-btn ${isSelected ? "active" : ""}`}
              onClick={() => onChange(typeKey)}
              style={
                isSelected
                  ? {
                      backgroundColor: accent,
                      color: "#020617",
                      fontWeight: 700,
                      boxShadow: `0 2px 8px ${accent}40`,
                    }
                  : undefined
              }
            >
              {typeKey.toUpperCase()}
            </button>
            
            <div className="acq-tooltip-card" style={{ "--accent-glow": accent } as React.CSSProperties}>
              <div className="acq-tooltip-header">
                <span className="acq-tooltip-title">{details.label}</span>
                <span className="acq-tooltip-cn">{details.nameCn}</span>
              </div>
              <div className="acq-tooltip-formula" style={{ color: accent }}>
                {details.formula}
              </div>
              <div className="acq-tooltip-desc">{details.description}</div>
              
              <div className="acq-tooltip-tradeoff">
                <div className="acq-tradeoff-label">
                  <span>不确定平衡 (Trade-off)</span>
                  <span className="acq-tradeoff-val">{details.tradeoff.label}</span>
                </div>
                <div className="acq-tradeoff-bars">
                  <div className="acq-bar-container">
                    <span className="bar-label">探索 Exploration</span>
                    <div className="bar-bg">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${details.tradeoff.exploration * 10}%`,
                          backgroundColor: accent,
                          boxShadow: `0 0 6px ${accent}`,
                        }}
                      />
                    </div>
                  </div>
                  <div className="acq-bar-container">
                    <span className="bar-label">开发 Exploitation</span>
                    <div className="bar-bg">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${details.tradeoff.exploitation * 10}%`,
                          backgroundColor: accent,
                          boxShadow: `0 0 6px ${accent}`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="acq-tooltip-scenarios">
                <div className="scenarios-title">最佳适用场景 (Best Suited For):</div>
                <ul>
                  {details.scenarios.map((s, idx) => (
                    <li key={idx}>{s}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}


export function NumberField({
  value,
  onChange,
  step = 1,
  min,
  max,
  width = "100%",
  ...props
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  width?: string | number;
  [key: string]: any;
}) {
  return (
    <input
      className="field-input"
      type="number"
      value={value}
      step={step}
      min={min}
      max={max}
      style={{ width }}
      onChange={(e) => onChange(Number(e.target.value))}
      {...props}
    />
  );
}

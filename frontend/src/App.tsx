import { useState } from "react";
import { BenchMode } from "./BenchMode";
import { OperationalMode } from "./OperationalMode";

const TRAD = "#f59e0b"; // Amber-500
const LLM = "#10b981";  // Emerald-500

type Mode = "bench" | "operational";

function App() {
  const [mode, setMode] = useState<Mode>("bench");

  return (
    <div className="instrument-grid" style={{ minHeight: "100vh" }}>
      <div style={{ maxWidth: 1320, margin: "0 auto", padding: "32px 24px 64px" }}>
        {/* Header */}
        <header style={{ marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
              <h1
                style={{
                  margin: 0,
                  fontSize: 28,
                  fontWeight: 800,
                  letterSpacing: "-0.01em",
                  fontFamily: "var(--font-display)",
                  background: `linear-gradient(135deg, ${TRAD}, ${LLM})`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}
              >
                BO·AGENT
              </h1>
              <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-ink-500)", fontFamily: "var(--font-sans)" }}>
                LLM-driven Bayesian Optimization Analysis Tool · {mode === "bench" ? "性能评测模式 (BENCHMARK)" : "实验实操模式 (OPERATIONAL)"}
              </span>
            </div>
            
            {/* Mode Switcher */}
            <nav style={{ display: "flex", background: "rgba(15, 23, 42, 0.6)", padding: 4, borderRadius: 10, border: "1px solid rgba(255, 255, 255, 0.08)", backdropFilter: "blur(4px)" }}>
              <button 
                onClick={() => setMode("bench")}
                className={mode === "bench" ? "active" : ""}
                style={navButtonStyle(mode === "bench")}
              >
                性能评测 (BENCHMARK)
              </button>
              <button 
                onClick={() => setMode("operational")}
                className={mode === "operational" ? "active" : ""}
                style={navButtonStyle(mode === "operational")}
              >
                实验实操 (OPERATIONAL)
              </button>
            </nav>
          </div>
        </header>

        <div style={{ display: mode === "bench" ? "block" : "none" }}>
          <BenchMode />
        </div>
        <div style={{ display: mode === "operational" ? "block" : "none" }}>
          <OperationalMode />
        </div>
      </div>
    </div>
  );
}

function navButtonStyle(isActive: boolean): React.CSSProperties {
  return { 
    padding: "6px 18px", 
    fontSize: 11, 
    fontWeight: 700, 
    borderRadius: 8,
    border: "none",
    fontFamily: "var(--font-display)",
    cursor: "pointer",
    background: isActive ? "rgba(255, 255, 255, 0.08)" : "transparent",
    color: isActive ? LLM : "var(--color-ink-500)",
    boxShadow: isActive ? "0 4px 12px rgba(0, 0, 0, 0.2)" : "none",
    transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)"
  };
}

export default App;

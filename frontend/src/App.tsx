import { useCallback, useEffect, useRef, useState } from "react";
import { getTasks, streamComparison } from "./lib/api";
import { AcqSelect, Field, NumberField } from "./components/Field";
import { MetricReadout } from "./components/MetricReadout";
import { ChartLegend, ConvergenceChart } from "./components/ConvergenceChart";
import { OperationalMode } from "./OperationalMode";
import type {
  AggPoint,
  ChartPoint,
  CompareEvent,
  LLMBOConfig,
  Task,
  TraditionalConfig,
} from "./types";

const TRAD = "#f2a516";
const LLM = "#16d69b";

type Mode = "bench" | "operational";
type RunState = "idle" | "running" | "done" | "error";

interface MethodAgg {
  bestMean: number | null;
  bestStd: number | null;
  genMean: number | null;
}

const EMPTY_AGG: MethodAgg = { bestMean: null, bestStd: null, genMean: null };

/** Convert backend aggregate points into Recharts rows with [lower, upper] bands. */
function toChartData(points: AggPoint[]): ChartPoint[] {
  return points.map((p) => ({
    iteration: p.iteration,
    trad_best_mean: p.trad_best_mean,
    trad_best_band: [
      p.trad_best_mean - p.trad_best_std,
      p.trad_best_mean + p.trad_best_std,
    ],
    trad_gen_mean: p.trad_gen_mean,
    llm_best_mean: p.llm_best_mean,
    llm_best_band: [
      p.llm_best_mean - p.llm_best_std,
      p.llm_best_mean + p.llm_best_std,
    ],
    llm_gen_mean: p.llm_gen_mean,
  }));
}

/** Pull the final-iteration aggregate for a method into a headline readout. */
function finalAgg(points: AggPoint[], method: "trad" | "llm"): MethodAgg {
  if (points.length === 0) return EMPTY_AGG;
  const last = points[points.length - 1];
  return {
    bestMean: last[`${method}_best_mean`],
    bestStd: last[`${method}_best_std`],
    genMean: last[`${method}_gen_mean`],
  };
}


function App() {
  const [mode, setMode] = useState<Mode>("bench");

  return (
    <div className="instrument-grid" style={{ minHeight: "100vh" }}>
      <div style={{ maxWidth: 1320, margin: "0 auto", padding: "32px 28px 64px" }}>
        {/* Header */}
        <header style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 16, flexWrap: "wrap", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
              <h1
                style={{
                  margin: 0,
                  fontSize: 26,
                  fontWeight: 800,
                  letterSpacing: "-0.02em",
                  background: `linear-gradient(90deg, ${TRAD}, ${LLM})`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}
              >
                BO·AGENT
              </h1>
              <span style={{ fontSize: 13, color: "var(--color-ink-300)" }}>
                LLM-driven Bayesian Optimization Analysis Tool · {mode === "bench" ? "Benchmarking Engine (性能评测模式)" : "Operational Mode (实验实操模式)"}
              </span>
            </div>
            
            {/* Mode Switcher */}
            <nav style={{ display: "flex", background: "var(--color-graphite-880)", padding: 4, borderRadius: 8, border: "1px solid var(--color-graphite-700)" }}>
              <button 
                onClick={() => setMode("bench")}
                style={{ 
                  padding: "6px 16px", 
                  fontSize: 12, 
                  fontWeight: 700, 
                  borderRadius: 6,
                  border: "none",
                  background: mode === "bench" ? "var(--color-graphite-700)" : "transparent",
                  color: mode === "bench" ? LLM : "var(--color-ink-500)",
                  transition: "all 0.2s"
                }}
              >
                性能评测 (BENCHMARK)
              </button>
              <button 
                onClick={() => setMode("operational")}
                style={{ 
                  padding: "6px 16px", 
                  fontSize: 12, 
                  fontWeight: 700, 
                  borderRadius: 6,
                  border: "none",
                  background: mode === "operational" ? "var(--color-graphite-700)" : "transparent",
                  color: mode === "operational" ? LLM : "var(--color-ink-500)",
                  transition: "all 0.2s"
                }}
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

function BenchMode() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskId, setTaskId] = useState("band_alignment");
  const [nInitial, setNInitial] = useState(5);
  const [nTrials, setNTrials] = useState(20);
  const [nSeeds, setNSeeds] = useState(5);

  const [trad, setTrad] = useState<TraditionalConfig>({
    acquisition: "ei",
    xi: 0.01,
    kappa: 2.576,
  });
  const [llm, setLlm] = useState<LLMBOConfig>({
    acquisition: "ucb",
    xi: 0.01,
    kappa: 2.576,
    n_candidates: 5,
    n_templates: 2,
    top_k: 20,
    alpha: 0.1,
  });

  const [runState, setRunState] = useState<RunState>("idle");
  const [error, setError] = useState("");
  const [targetCol] = useState("eta");
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [tradAgg, setTradAgg] = useState<MethodAgg>(EMPTY_AGG);
  const [llmAgg, setLlmAgg] = useState<MethodAgg>(EMPTY_AGG);
  const [completedSeeds, setCompletedSeeds] = useState(0);
  const [totalSeeds, setTotalSeeds] = useState(0);
  const [tradBusy, setTradBusy] = useState(false);
  const [llmBusy, setLlmBusy] = useState(false);
  const abortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    getTasks()
      .then((list) => {
        setTasks(list);
        if (list.length > 0) setTaskId(list[0].task_id);
      })
      .catch(() => {});
    return () => abortRef.current?.();
  }, []);

  // Default seed pool; first `nSeeds` are used per run.
  const seedPool = [42, 7, 100, 1, 21, 13, 77, 2024, 5, 99];

  const handleEvent = useCallback(
    (ev: CompareEvent) => {
      if (ev.type === "meta") {
        setTotalSeeds(ev.total_seeds);
      } else if (ev.type === "seed_start") {
        // A new seed begins; both engines will work through it.
      } else if (ev.type === "step_start") {
        if (ev.method === "traditional") setTradBusy(true);
        else setLlmBusy(true);
      } else if (ev.type === "aggregate") {
        setChartData(toChartData(ev.points));
        setTradAgg(finalAgg(ev.points, "trad"));
        setLlmAgg(finalAgg(ev.points, "llm"));
        setCompletedSeeds(ev.completed_seeds);
        setTradBusy(false);
        setLlmBusy(false);
      } else if (ev.type === "done") {
        setTradBusy(false);
        setLlmBusy(false);
        setRunState("done");
      } else if (ev.type === "error") {
        setTradBusy(false);
        setLlmBusy(false);
        setError(ev.message);
        setRunState("error");
      }
    },
    [],
  );

  const handleRun = useCallback(() => {
    abortRef.current?.();
    setChartData([]);
    setTradAgg(EMPTY_AGG);
    setLlmAgg(EMPTY_AGG);
    setCompletedSeeds(0);
    setTotalSeeds(nSeeds);
    setTradBusy(false);
    setLlmBusy(false);
    setError("");
    setRunState("running");

    const seeds = seedPool.slice(0, Math.max(1, Math.min(nSeeds, seedPool.length)));
    abortRef.current = streamComparison(
      { task_id: taskId, n_initial: nInitial, n_trials: nTrials, seeds, traditional: trad, llmbo: llm },
      handleEvent,
      (msg) => {
        setError(msg);
        setRunState("error");
      },
    );
  }, [taskId, nInitial, nTrials, nSeeds, trad, llm, handleEvent]);

  const handleStop = useCallback(() => {
    abortRef.current?.();
    setTradBusy(false);
    setLlmBusy(false);
    setRunState("idle");
  }, []);

  const running = runState === "running";

  return (
    <>
        {/* Shared controls */}
        <div
          className="panel"
          style={{
            padding: "16px 20px",
            marginBottom: 20,
            display: "grid",
            gridTemplateColumns: "2fr 1fr 1fr 1fr auto",
            gap: 16,
            alignItems: "end",
          }}
        >
          <Field label="数据集 (Dataset)">
            <select className="field-input" value={taskId} onChange={(e) => setTaskId(e.target.value)}>
              {tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.name || t.task_id}
                </option>
              ))}
            </select>
          </Field>
          <Field label="初始点数 (Initial Points)">
            <NumberField value={nInitial} onChange={setNInitial} min={1} max={50} />
          </Field>
          <Field label="迭代次数 (Trials)">
            <NumberField value={nTrials} onChange={setNTrials} min={1} max={200} />
          </Field>
          <Field label="重复次数 (Seeds / Runs)">
            <NumberField value={nSeeds} onChange={setNSeeds} min={1} max={10} />
          </Field>
          <div style={{ display: "flex", gap: 10 }}>
            <button
              className="run-btn"
              onClick={handleRun}
              disabled={running}
              style={{
                background: `linear-gradient(90deg, ${TRAD}, ${LLM})`,
                color: "#07090d",
                opacity: running ? 0.5 : 1,
              }}
            >
              {running ? "Running..." : "▶ Start Comparison (开启对比实验)"}
            </button>
            {running && (
              <button
                className="run-btn"
                onClick={handleStop}
                style={{ background: "var(--color-graphite-700)", color: "var(--color-ink-100)" }}
              >
                Stop (停止)
              </button>
            )}
          </div>
        </div>

        {/* Dual config panels */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
          {/* Traditional */}
          <div className="panel panel-trad" style={{ padding: 20 }}>
            <PanelHeader accent={TRAD} title="A · Traditional Bayesian Optimization" subtitle="Gaussian Process Surrogate Model + 解析式 Acquisition Function" />
            <div style={{ display: "grid", gap: 14, marginTop: 16 }}>
              <Field label="Acquisition Function (采集函数)">
                <AcqSelect
                  value={trad.acquisition}
                  onChange={(v) => setTrad({ ...trad, acquisition: v })}
                  accent="var(--color-graphite-700)"
                />
              </Field>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <Field label="ξ (EI/PI 探索因子)" hint="exploration margin">
                  <NumberField value={trad.xi} onChange={(v) => setTrad({ ...trad, xi: v })} step={0.01} min={0} />
                </Field>
                <Field label="κ (UCB 探索因子)" hint="confidence width">
                  <NumberField value={trad.kappa} onChange={(v) => setTrad({ ...trad, kappa: v })} step={0.1} min={0} />
                </Field>
              </div>
            </div>
            <div style={{ marginTop: 16 }}>
              <MetricReadout
                accent={TRAD}
                bestMean={tradAgg.bestMean}
                bestStd={tradAgg.bestStd}
                genMean={tradAgg.genMean}
                completedSeeds={completedSeeds}
                totalSeeds={totalSeeds}
                busy={tradBusy}
                busyLabel="Gaussian Process Optimizing..."
              />
            </div>
          </div>

          {/* LLMBO */}
          <div className="panel panel-llm" style={{ padding: 20 }}>
            <PanelHeader accent={LLM} title="B · LLMBO" subtitle="Gaussian Process Pre-screening + LLM Acquisition Function" />
            <div style={{ display: "grid", gap: 14, marginTop: 16 }}>
              <Field label="GP Pre-screening Acquisition Function">
                <AcqSelect
                  value={llm.acquisition}
                  onChange={(v) => setLlm({ ...llm, acquisition: v })}
                  accent="var(--color-graphite-700)"
                />
              </Field>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                <Field label="Top-K">
                  <NumberField value={llm.top_k} onChange={(v) => setLlm({ ...llm, top_k: v })} min={1} max={100} />
                </Field>
                <Field label="Candidates (候选点数)">
                  <NumberField value={llm.n_candidates} onChange={(v) => setLlm({ ...llm, n_candidates: v })} min={1} max={50} />
                </Field>
                <Field label="α">
                  <NumberField value={llm.alpha} onChange={(v) => setLlm({ ...llm, alpha: v })} step={0.1} />
                </Field>
              </div>
            </div>
            <div style={{ marginTop: 16 }}>
              <MetricReadout
                accent={LLM}
                bestMean={llmAgg.bestMean}
                bestStd={llmAgg.bestStd}
                genMean={llmAgg.genMean}
                completedSeeds={completedSeeds}
                totalSeeds={totalSeeds}
                busy={llmBusy}
                busyLabel="LLM Inferring..."
              />
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="panel" style={{ padding: 20 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 16,
              flexWrap: "wrap",
              gap: 12,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h2 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>收敛曲线 (Convergence)</h2>
              {running && (
                <span
                  className="live-dot"
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: LLM,
                    display: "inline-block",
                  }}
                />
              )}
              {running && (
                <span style={{ fontSize: 11, color: LLM, fontFamily: "var(--font-mono)" }}>实时 (LIVE)</span>
              )}
            </div>
            <ChartLegend />
          </div>

          {chartData.length === 0 ? (
            <div
              style={{
                height: 420,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--color-ink-500)",
                fontFamily: "var(--font-mono)",
                fontSize: 13,
              }}
            >
              {runState === "idle"
                ? "Please configure parameters and click 'Start Comparison' (请配置对比实验参数并开启)"
                : "Waiting for first iteration results... (等待迭代结果)"}
            </div>
          ) : (
            <ConvergenceChart data={chartData} targetCol={targetCol} />
          )}
        </div>

        {/* Final mean ± std summary */}
        {(tradAgg.bestMean !== null || llmAgg.bestMean !== null) && (
          <div className="fade-rise" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 20 }}>
            <SummaryCard
              accent={TRAD}
              title="Traditional BO · 最终结果 (Final)"
              mean={tradAgg.bestMean}
              std={tradAgg.bestStd}
              completedSeeds={completedSeeds}
            />
            <SummaryCard
              accent={LLM}
              title="LLMBO · 最终结果 (Final)"
              mean={llmAgg.bestMean}
              std={llmAgg.bestStd}
              completedSeeds={completedSeeds}
            />
          </div>
        )}

        {error && (
          <div
            style={{
              marginTop: 20,
              padding: 16,
              border: "1px solid var(--color-fault-400)",
              borderRadius: 10,
              background: "rgb(255 93 108 / 0.08)",
              color: "var(--color-fault-400)",
              fontFamily: "var(--font-mono)",
              fontSize: 13,
            }}
          >
            ⚠ {error}
          </div>
        )}
    </>
  );
}

function PanelHeader({ accent, title, subtitle }: { accent: string; title: string; subtitle: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ width: 10, height: 10, borderRadius: 3, background: accent, boxShadow: `0 0 12px ${accent}` }} />
      <div>
        <div style={{ fontSize: 13, fontWeight: 800, letterSpacing: "0.04em", color: accent }}>{title}</div>
        <div style={{ fontSize: 11, color: "var(--color-ink-500)" }}>{subtitle}</div>
      </div>
    </div>
  );
}

function SummaryCard({
  accent,
  title,
  mean,
  std,
  completedSeeds,
}: {
  accent: string;
  title: string;
  mean: number | null;
  std: number | null;
  completedSeeds: number;
}) {
  return (
    <div className="panel" style={{ padding: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: "0.04em" }}>{title}</span>
        <span style={{ fontSize: 11, color: "var(--color-ink-500)", fontFamily: "var(--font-mono)" }}>
          {completedSeeds} 种子均值 (seeds)
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 30, fontWeight: 800, color: accent, lineHeight: 1 }}>
          {mean === null ? "—" : mean.toFixed(3)}
        </span>
        {mean !== null && std !== null && (
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 16, color: "var(--color-ink-300)" }}>
            ± {std.toFixed(3)}
          </span>
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: "var(--color-ink-500)" }}>
        Best Score 最终收敛值 (均值 ± 标准差)
      </div>
    </div>
  );
}

export default App;

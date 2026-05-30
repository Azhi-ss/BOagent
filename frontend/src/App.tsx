import { useCallback, useEffect, useRef, useState } from "react";
import { getTasks, streamComparison } from "./lib/api";
import { AcqSelect, Field, NumberField } from "./components/Field";
import { MetricReadout } from "./components/MetricReadout";
import { ChartLegend, ConvergenceChart } from "./components/ConvergenceChart";
import type {
  ChartPoint,
  CompareEvent,
  IterationEvent,
  LLMBOConfig,
  Method,
  Task,
  TraditionalConfig,
} from "./types";

const TRAD = "#f2a516";
const LLM = "#16d69b";

type RunState = "idle" | "running" | "done" | "error";

interface MethodSnapshot {
  best: number | null;
  gen: number | null;
  candidate: number | null;
  iteration: number;
  bestConfig: Record<string, number> | null;
}

const EMPTY_SNAPSHOT: MethodSnapshot = {
  best: null,
  gen: null,
  candidate: null,
  iteration: 0,
  bestConfig: null,
};

function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskId, setTaskId] = useState("band_alignment");
  const [nInitial, setNInitial] = useState(5);
  const [nTrials, setNTrials] = useState(20);
  const [seed, setSeed] = useState(42);

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
  const [targetCol, setTargetCol] = useState("eta");
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [tradSnap, setTradSnap] = useState<MethodSnapshot>(EMPTY_SNAPSHOT);
  const [llmSnap, setLlmSnap] = useState<MethodSnapshot>(EMPTY_SNAPSHOT);
  const [tradBusy, setTradBusy] = useState(false);
  const [llmBusy, setLlmBusy] = useState(false);
  const abortRef = useRef<(() => void) | null>(null);
  const chartRef = useRef<Map<number, ChartPoint>>(new Map());

  useEffect(() => {
    getTasks()
      .then((list) => {
        setTasks(list);
        if (list.length > 0) setTaskId(list[0].task_id);
      })
      .catch(() => {});
    return () => abortRef.current?.();
  }, []);

  const applyIteration = useCallback((ev: IterationEvent) => {
    const map = chartRef.current;
    const row = map.get(ev.iteration) ?? { iteration: ev.iteration };
    if (ev.method === "traditional") {
      row.trad_best = ev.best_score;
      row.trad_gen = ev.generalization_score;
      row.trad_cand = ev.candidate_score;
    } else {
      row.llm_best = ev.best_score;
      row.llm_gen = ev.generalization_score;
      row.llm_cand = ev.candidate_score;
    }
    map.set(ev.iteration, row);
    setChartData(Array.from(map.values()).sort((a, b) => a.iteration - b.iteration));

    const snap: MethodSnapshot = {
      best: ev.best_score,
      gen: ev.generalization_score,
      candidate: ev.candidate_score,
      iteration: ev.iteration,
      bestConfig: ev.best_config,
    };
    if (ev.method === "traditional") {
      setTradSnap(snap);
      setTradBusy(false);
    } else {
      setLlmSnap(snap);
      setLlmBusy(false);
    }
  }, []);

  const handleEvent = useCallback(
    (ev: CompareEvent) => {
      if (ev.type === "meta") {
        setTargetCol(ev.target_col);
      } else if (ev.type === "step_start") {
        if (ev.method === "traditional") setTradBusy(true);
        else setLlmBusy(true);
      } else if (ev.type === "iteration") {
        applyIteration(ev);
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
    [applyIteration],
  );

  const handleRun = useCallback(() => {
    abortRef.current?.();
    chartRef.current = new Map();
    setChartData([]);
    setTradSnap(EMPTY_SNAPSHOT);
    setLlmSnap(EMPTY_SNAPSHOT);
    setTradBusy(false);
    setLlmBusy(false);
    setError("");
    setRunState("running");

    abortRef.current = streamComparison(
      { task_id: taskId, n_initial: nInitial, n_trials: nTrials, seed, traditional: trad, llmbo: llm },
      handleEvent,
      (msg) => {
        setError(msg);
        setRunState("error");
      },
    );
  }, [taskId, nInitial, nTrials, seed, trad, llm, handleEvent]);

  const handleStop = useCallback(() => {
    abortRef.current?.();
    setTradBusy(false);
    setLlmBusy(false);
    setRunState("idle");
  }, []);

  const running = runState === "running";

  return (
    <div className="instrument-grid" style={{ minHeight: "100vh" }}>
      <div style={{ maxWidth: 1320, margin: "0 auto", padding: "32px 28px 64px" }}>
        {/* Header */}
        <header style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 16, flexWrap: "wrap" }}>
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
              BO·BENCH
            </h1>
            <span style={{ fontSize: 13, color: "var(--color-ink-300)" }}>
              Traditional BO vs LLMBO · real-time convergence over PVK Excel datasets
            </span>
          </div>
        </header>

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
          <Field label="Dataset (shared train/test split)">
            <select className="field-input" value={taskId} onChange={(e) => setTaskId(e.target.value)}>
              {tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.name || t.task_id}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Initial Points">
            <NumberField value={nInitial} onChange={setNInitial} min={1} max={50} />
          </Field>
          <Field label="Trials">
            <NumberField value={nTrials} onChange={setNTrials} min={1} max={200} />
          </Field>
          <Field label="Seed">
            <NumberField value={seed} onChange={setSeed} min={0} />
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
              {running ? "RUNNING…" : "▶ RUN COMPARISON"}
            </button>
            {running && (
              <button
                className="run-btn"
                onClick={handleStop}
                style={{ background: "var(--color-graphite-700)", color: "var(--color-ink-100)" }}
              >
                STOP
              </button>
            )}
          </div>
        </div>

        {/* Dual config panels */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
          {/* Traditional */}
          <div className="panel panel-trad" style={{ padding: 20 }}>
            <PanelHeader accent={TRAD} title="A · TRADITIONAL BO" subtitle="GP surrogate + analytic acquisition" />
            <div style={{ display: "grid", gap: 14, marginTop: 16 }}>
              <Field label="Acquisition Function">
                <AcqSelect
                  value={trad.acquisition}
                  onChange={(v) => setTrad({ ...trad, acquisition: v })}
                  accent="var(--color-graphite-700)"
                />
              </Field>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <Field label="ξ (EI/PI explore)" hint="exploration margin">
                  <NumberField value={trad.xi} onChange={(v) => setTrad({ ...trad, xi: v })} step={0.01} min={0} />
                </Field>
                <Field label="κ (UCB explore)" hint="confidence width">
                  <NumberField value={trad.kappa} onChange={(v) => setTrad({ ...trad, kappa: v })} step={0.1} min={0} />
                </Field>
              </div>
            </div>
            <div style={{ marginTop: 16 }}>
              <MetricReadout
                accent={TRAD}
                best={tradSnap.best}
                gen={tradSnap.gen}
                candidate={tradSnap.candidate}
                iteration={tradSnap.iteration}
                totalTrials={nTrials}
                busy={tradBusy}
                busyLabel="GP optimizing…"
              />
            </div>
          </div>

          {/* LLMBO */}
          <div className="panel panel-llm" style={{ padding: 20 }}>
            <PanelHeader accent={LLM} title="B · LLMBO" subtitle="GP pre-filter + LLM acquisition" />
            <div style={{ display: "grid", gap: 14, marginTop: 16 }}>
              <Field label="GP Pre-filter Acquisition">
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
                <Field label="Candidates">
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
                best={llmSnap.best}
                gen={llmSnap.gen}
                candidate={llmSnap.candidate}
                iteration={llmSnap.iteration}
                totalTrials={nTrials}
                busy={llmBusy}
                busyLabel="LLM reasoning…"
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
              <h2 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>Convergence</h2>
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
                <span style={{ fontSize: 11, color: LLM, fontFamily: "var(--font-mono)" }}>LIVE</span>
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
                ? "Configure both methods and press RUN COMPARISON"
                : "Awaiting first iteration…"}
            </div>
          ) : (
            <ConvergenceChart data={chartData} targetCol={targetCol} />
          )}
        </div>

        {/* Best config comparison */}
        {(tradSnap.bestConfig || llmSnap.bestConfig) && (
          <div className="fade-rise" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 20 }}>
            <BestConfigCard accent={TRAD} title="Traditional · Best Recipe" config={tradSnap.bestConfig} score={tradSnap.best} />
            <BestConfigCard accent={LLM} title="LLMBO · Best Recipe" config={llmSnap.bestConfig} score={llmSnap.best} />
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
      </div>
    </div>
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

function BestConfigCard({
  accent,
  title,
  config,
  score,
}: {
  accent: string;
  title: string;
  config: Record<string, number> | null;
  score: number | null;
}) {
  return (
    <div className="panel" style={{ padding: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: "0.04em" }}>{title}</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 18, fontWeight: 700, color: accent }}>
          {score === null ? "—" : score.toFixed(4)}
        </span>
      </div>
      {config ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {Object.entries(config).map(([k, v]) => (
            <div
              key={k}
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                padding: "6px 10px",
                background: "var(--color-graphite-880)",
                borderRadius: 6,
                border: "1px solid var(--color-graphite-700)",
              }}
            >
              <span style={{ color: "var(--color-ink-500)" }}>{k}</span>
              <span style={{ color: "var(--color-ink-100)" }}>{v.toFixed(3)}</span>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ color: "var(--color-ink-500)", fontSize: 12 }}>No data yet</div>
      )}
    </div>
  );
}

export default App;

import { useCallback, useEffect, useRef, useState } from "react";
import { getTasks, streamComparison } from "./lib/api";
import { Field, NumberField } from "./components/Field";
import { AcquisitionConfig } from "./components/AcquisitionConfig";
import { MetricReadout } from "./components/MetricReadout";
import { ConvergenceChart } from "./components/ConvergenceChart";
import { PlayIcon, StopIcon } from "./components/Icons";
import { PanelHeader, SummaryCard } from "./components/Layout";
import type {
  AggPoint,
  ChartPoint,
  CompareEvent,
  LLMBOConfig,
  Task,
  TraditionalConfig,
} from "./types";

const TRAD = "var(--color-amber-500)";
const LLM = "var(--color-signal-500)";

type RunState = "idle" | "running" | "done" | "error";

interface MethodAgg {
  bestMean: number | null;
  bestStd: number | null;
  genMean: number | null;
}

const EMPTY_AGG: MethodAgg = { bestMean: null, bestStd: null, genMean: null };

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

function finalAgg(points: AggPoint[], method: "trad" | "llm"): MethodAgg {
  if (points.length === 0) return EMPTY_AGG;
  const last = points[points.length - 1];
  return {
    bestMean: last[`${method}_best_mean`],
    bestStd: last[`${method}_best_std`],
    genMean: last[`${method}_gen_mean`],
  };
}

export function BenchMode() {
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
    chat_engine: "deepseek-v4-flash",
  });

  const [runState, setRunState] = useState<RunState>("idle");
  const [error, setError] = useState("");
  const [targetCol] = useState("eta");
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [tradAgg, setTradAgg] = useState<MethodAgg>(EMPTY_AGG);
  const [llmAgg, setLlmAgg] = useState<MethodAgg>(EMPTY_AGG);
  const [completedSeeds, setCompletedSeeds] = useState(0);
  const [totalSeeds, setTotalSeeds] = useState(0);
  // Fine-grained iteration progress: each step_start increments by 1.
  // totalIters = nTrials * nSeeds * 2 (traditional + llmbo per step).
  const [completedIters, setCompletedIters] = useState(0);
  const [totalIters, setTotalIters] = useState(0);
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

  const seedPool = [42, 7, 100, 1, 21, 13, 77, 2024, 5, 99];

  const handleEvent = useCallback(
    (ev: CompareEvent) => {
      if (ev.type === "meta") {
        setTotalSeeds(ev.total_seeds);
        // 2 methods (trad + llmbo) × n_trials × n_seeds = total engine calls
        setTotalIters(ev.n_trials * ev.total_seeds * 2);
        setCompletedIters(0);
      } else if (ev.type === "step_start") {
        // Each step_start means one iteration of one method has begun → count it done.
        setCompletedIters((n) => n + 1);
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
    setCompletedIters(0);
    setTotalIters(nTrials * nSeeds * 2);
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
      <div className="panel" style={{ padding: "20px 24px", marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--font-display)", color: "var(--color-ink-300)" }}>全局配置 GLOBAL CONFIG</span>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => { setNInitial(3); setNTrials(10); setNSeeds(2); }}
              style={{ padding: "4px 10px", fontSize: 10, borderRadius: 4, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "var(--color-ink-300)" }}
            >
              快速预览 (Fast Preview)
            </button>
            <button
              onClick={() => { setNInitial(5); setNTrials(20); setNSeeds(5); }}
              style={{ padding: "4px 10px", fontSize: 10, borderRadius: 4, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "var(--color-ink-300)" }}
            >
              标准评测 (Standard)
            </button>
          </div>
        </div>
        
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "2fr 1fr 1fr 1fr auto",
            gap: 16,
            alignItems: "end",
          }}
        >
          <Field label="数据集 (Dataset)">
            <select className="field-input" data-testid="task-selector" value={taskId} onChange={(e) => setTaskId(e.target.value)}>
              {tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.name || t.task_id}
                </option>
              ))}
            </select>
          </Field>
          <Field label="初始点数 (Initial Points)">
            <NumberField value={nInitial} onChange={setNInitial} min={1} max={50} data-testid="input-n-initial" />
          </Field>
          <Field label="迭代次数 (Trials)">
            <NumberField value={nTrials} onChange={setNTrials} min={1} max={200} data-testid="input-n-trials" />
          </Field>
          <Field label="重复次数 (Seeds / Runs)">
            <NumberField value={nSeeds} onChange={setNSeeds} min={1} max={10} data-testid="input-n-seeds" />
          </Field>
          
          <div style={{ display: "flex", gap: 12 }}>
            <button
              className="run-btn"
              data-testid="run-bench-btn"
              onClick={handleRun}
              disabled={running}
              style={{
                background: `linear-gradient(90deg, ${TRAD}, ${LLM})`,
                color: "#020617",
                opacity: running ? 0.6 : 1,
                boxShadow: running ? "none" : `0 4px 15px -3px ${LLM}40`,
              }}
            >
              <PlayIcon size={14} />
              <span>{running ? "运行中..." : "开启对比实验 (Run)"}</span>
            </button>
            {running && (
              <button
                className="run-btn"
                data-testid="stop-bench-btn"
                onClick={handleStop}
                style={{
                  background: "rgba(244, 63, 94, 0.15)",
                  border: "1px solid rgba(244, 63, 94, 0.4)",
                  color: "var(--color-fault-400)",
                  boxShadow: "none"
                }}
              >
                <StopIcon size={14} />
                <span>停止</span>
              </button>
            )}
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
        <div className="panel panel-trad" style={{ padding: 24 }}>
          <PanelHeader accent={TRAD} title="A · 传统贝叶斯优化 TRADITIONAL BO" subtitle="Gaussian Process Surrogate Model + Analytic Acquisition Function" />
          <div className="sub-panel" style={{ marginTop: 20 }}>
            <span className="sub-panel-title">核心策略 CORE STRATEGY</span>
            <AcquisitionConfig config={trad} onChange={v => setTrad({...trad, ...v})} accent={TRAD} />
          </div>
          <div style={{ marginTop: 20 }}>
            <MetricReadout
              accent={TRAD}
              bestMean={tradAgg.bestMean}
              bestStd={tradAgg.bestStd}
              genMean={tradAgg.genMean}
              completedSeeds={completedSeeds}
              totalSeeds={totalSeeds}
              completedIters={completedIters}
              totalIters={totalIters}
              busy={tradBusy}
              busyLabel="GP Optimization In Progress..."
            />
          </div>
        </div>

        <div className="panel panel-llm" style={{ padding: 24 }}>
          <PanelHeader accent={LLM} title="B · 大模型驱动优化 LLMBO" subtitle="Gaussian Process Pre-screening + LLM Acquisition Function" />
          <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 20 }}>
            <div className="sub-panel" style={{ borderColor: `${LLM}20` }}>
              <span className="sub-panel-title" style={{ color: LLM }}>1. GP 预筛选阶段 GP PRE-SCREENING</span>
              <AcquisitionConfig config={llm} onChange={v => setLlm({...llm, ...v})} accent={LLM} />
              <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
                <Field label="筛选数量" subLabel="Top-K Selection" hint="进入决赛圈的点数">
                  <NumberField value={llm.top_k} onChange={(v) => setLlm({ ...llm, top_k: v })} min={1} max={100} width={100} />
                </Field>
              </div>
            </div>
            <div className="sub-panel" style={{ borderColor: `${LLM}20` }}>
              <span className="sub-panel-title" style={{ color: LLM }}>2. 大模型决策阶段 LLM REFINEMENT</span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
                <Field label="模型引擎" subLabel="Model">
                  <select 
                    className="field-input" 
                    style={{ width: 140, padding: '0 8px' }}
                    value={llm.chat_engine} 
                    onChange={(e) => setLlm({ ...llm, chat_engine: e.target.value })}
                  >
                    <option value="deepseek-v4-flash">DeepSeek V4</option>
                  </select>
                </Field>
                <Field label="权衡系数" subLabel="Refinement Alpha α" hint="LLM 决策话语权">
                  <NumberField value={llm.alpha} onChange={(v) => setLlm({ ...llm, alpha: v })} step={0.1} width={100} />
                </Field>
                <Field label="精选建议" subLabel="Candidates Count" hint="最终建议配方数">
                  <NumberField value={llm.n_candidates} onChange={(v) => setLlm({ ...llm, n_candidates: v })} min={1} max={50} width={100} />
                </Field>
              </div>
            </div>
          </div>
          <div style={{ marginTop: 24 }}>
            <MetricReadout
              accent={LLM}
              bestMean={llmAgg.bestMean}
              bestStd={llmAgg.bestStd}
              genMean={llmAgg.genMean}
              completedSeeds={completedSeeds}
              totalSeeds={totalSeeds}
              completedIters={completedIters}
              totalIters={totalIters}
              busy={llmBusy}
              busyLabel="LLM Multi-Agent Inferring..."
            />
          </div>
        </div>
      </div>

      <div className="panel" style={{ padding: 24, marginBottom: 24, display: "flex", flexDirection: "column" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20, flexShrink: 0 }}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, fontFamily: "var(--font-display)" }}>收敛曲线 (Convergence Analytics)</h2>
          {running && <span className="live-dot" style={{ width: 8, height: 8, borderRadius: "50%", background: LLM }} />}
          {running && <span style={{ fontSize: 11, color: LLM, fontFamily: "var(--font-mono)", fontWeight: 600 }}>实时 (LIVE)</span>}
        </div>
        {chartData.length === 0 ? (
          <div style={{ flex: 1, minHeight: 420, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-ink-500)", fontFamily: "var(--font-mono)", fontSize: 13, background: "rgba(0, 0, 0, 0.15)", borderRadius: 12, border: "1px dashed rgba(255, 255, 255, 0.05)" }}>
            {runState === "idle" ? "Please configure parameters and click 'Start Comparison' (请配置对比实验参数并开启)" : "Waiting for first iteration results... (等待迭代结果)"}
          </div>
        ) : <ConvergenceChart data={chartData} targetCol={targetCol} />}
      </div>

      {(tradAgg.bestMean !== null || llmAgg.bestMean !== null) && (
        <div className="fade-rise" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginTop: 24 }}>
          <SummaryCard accent={TRAD} title="Traditional BO · 最终结果 (Final Score)" mean={tradAgg.bestMean} std={tradAgg.bestStd} completedSeeds={completedSeeds} testId="summary-trad" />
          <SummaryCard accent={LLM} title="LLMBO · 最终结果 (Final Score)" mean={llmAgg.bestMean} std={llmAgg.bestStd} completedSeeds={completedSeeds} testId="summary-llm" />
        </div>
      )}

      {error && (
        <div style={{ marginTop: 24, padding: 18, border: "1px solid rgba(244, 63, 94, 0.3)", borderRadius: 12, background: "rgba(244, 63, 94, 0.06)", color: "var(--color-fault-400)", fontFamily: "var(--font-mono)", fontSize: 13, backdropFilter: "blur(4px)", boxShadow: "0 4px 15px rgba(0, 0, 0, 0.2)" }}>
          <span style={{ fontWeight: 700, marginRight: 6 }}>⚠ System Failure:</span> {error}
        </div>
      )}
    </>
  );
}

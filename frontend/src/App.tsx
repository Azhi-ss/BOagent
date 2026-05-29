import { useCallback, useEffect, useState } from "react";
import { createBenchmarkRun, getTasks } from "./lib/api";
import type { BenchmarkResponse, Task } from "./types";

type RunState = "idle" | "running" | "done" | "error";

function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskId, setTaskId] = useState("band_alignment");
  const [nTrials, setNTrials] = useState(20);
  const [nInitial, setNInitial] = useState(5);
  const [seed, setSeed] = useState(42);
  const [seedsText, setSeedsText] = useState("");
  const [smMode, setSmMode] = useState<"discriminative" | "generative">("discriminative");
  const [nCandidates, setNCandidates] = useState(10);
  const [topK, setTopK] = useState(20);
  const [runState, setRunState] = useState<RunState>("idle");
  const [result, setResult] = useState<BenchmarkResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getTasks()
      .then((list) => {
        setTasks(list);
        if (list.length > 0) setTaskId(list[0].task_id);
      })
      .catch(() => {});
  }, []);

  const handleRun = useCallback(async () => {
    setRunState("running");
    setError("");
    setResult(null);
    try {
      const seeds = seedsText
        .split(",")
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !isNaN(n));
      const res = await createBenchmarkRun({
        task_id: taskId,
        n_trials: nTrials,
        n_initial: nInitial,
        seed,
        seeds: seeds.length > 0 ? seeds : undefined,
        sm_mode: smMode,
        n_candidates: nCandidates,
        top_k: topK,
      });
      setResult(res);
      setRunState("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setRunState("error");
    }
  }, [taskId, nTrials, nInitial, seed, seedsText, smMode, nCandidates, topK]);

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "40px 24px", fontFamily: "system-ui" }}>
      <h1 style={{ margin: "0 0 8px", fontSize: 28 }}>BOagent Benchmark</h1>
      <p style={{ margin: "0 0 32px", color: "#6b7280", fontSize: 14 }}>
        GP+LLM acquisition function evaluation over PVK-LLM datasets
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 16,
          marginBottom: 24,
        }}
      >
        <label>
          Task
          <select value={taskId} onChange={(e) => setTaskId(e.target.value)}>
            {tasks.map((t) => (
              <option key={t.task_id} value={t.task_id}>
                {t.task_id}
              </option>
            ))}
          </select>
        </label>
        <label>
          n_trials
          <input type="number" value={nTrials} onChange={(e) => setNTrials(Number(e.target.value))} />
        </label>
        <label>
          n_initial
          <input type="number" value={nInitial} onChange={(e) => setNInitial(Number(e.target.value))} />
        </label>
        <label>
          seed
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </label>
        <label>
          seeds (multi, comma)
          <input
            type="text"
            value={seedsText}
            onChange={(e) => setSeedsText(e.target.value)}
            placeholder="42,123,456"
          />
        </label>
        <label>
          sm_mode
          <select value={smMode} onChange={(e) => setSmMode(e.target.value as "discriminative" | "generative")}>
            <option value="discriminative">discriminative</option>
            <option value="generative">generative</option>
          </select>
        </label>
        <label>
          n_candidates
          <input type="number" value={nCandidates} onChange={(e) => setNCandidates(Number(e.target.value))} />
        </label>
        <label>
          top_k
          <input type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={runState === "running"}
        style={{
          padding: "12px 32px",
          fontSize: 16,
          fontWeight: 600,
          cursor: runState === "running" ? "not-allowed" : "pointer",
          opacity: runState === "running" ? 0.6 : 1,
          marginBottom: 24,
        }}
      >
        {runState === "running" ? "Running..." : "Run Benchmark"}
      </button>

      {runState === "running" && (
        <p style={{ color: "#6b7280" }}>Benchmark running — this may take several minutes...</p>
      )}

      {error && (
        <div style={{ padding: 16, background: "#fee2e2", borderRadius: 8, marginBottom: 16 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div>
          <h2 style={{ fontSize: 18, margin: "24px 0 12px" }}>
            Results — {result.task_id} ({result.runs} run{result.runs !== 1 ? "s" : ""})
          </h2>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "2px solid #e5e7eb" }}>
                <th style={{ padding: "8px 12px" }}>Seed</th>
                <th style={{ padding: "8px 12px" }}>Best Score</th>
                <th style={{ padding: "8px 12px" }}>Generalization Score</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r) => (
                <tr key={r.seed} style={{ borderBottom: "1px solid #f3f4f6" }}>
                  <td style={{ padding: "8px 12px" }}>{r.seed}</td>
                  <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>{r.best_score.toFixed(4)}</td>
                  <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>
                    {r.best_generalization_score.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ marginTop: 12, color: "#6b7280", fontSize: 13 }}>Output: {result.output_dir}</p>
        </div>
      )}
    </div>
  );
}

export default App;

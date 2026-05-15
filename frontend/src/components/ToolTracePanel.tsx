import { statusTone, stringifyCompact } from "../lib/format";
import type { SessionArtifact, ToolTraceEntry } from "../types";

interface ToolTracePanelProps {
  trace: ToolTraceEntry[];
  artifacts: SessionArtifact[] | string[];
  isLoading: boolean;
}

export function ToolTracePanel({ trace, artifacts, isLoading }: ToolTracePanelProps) {
  return (
    <section className="panel-surface rounded-3xl p-5">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.32em] text-slate-500">
            Tool Trace
          </p>
          <h2 className="mt-2 text-xl font-semibold text-white">工具执行轨迹</h2>
        </div>
        <span className="rounded-full border border-slate-600/40 px-3 py-1 font-mono text-xs text-slate-400">
          {trace.length} events
        </span>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[0, 1].map((item) => (
            <div className="h-28 animate-pulse rounded-2xl bg-slate-700/20" key={item} />
          ))}
        </div>
      ) : trace.length > 0 ? (
        <div className="space-y-3">
          {trace.map((entry, index) => (
            <details
              className="group rounded-2xl border border-slate-700/50 bg-black/20 p-4 open:border-signal-400/35"
              key={entry.id || `${entry.tool}-${index}`}
            >
              <summary className="flex cursor-pointer list-none items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-xs text-slate-500">
                    {entry.tool || entry.name || `tool-${index + 1}`}
                  </p>
                  <h3 className="mt-1 text-sm font-semibold text-white">
                    {entry.message || "查看输入输出详情"}
                  </h3>
                  {entry.duration_ms ? (
                    <p className="mt-1 font-mono text-xs text-slate-500">{entry.duration_ms} ms</p>
                  ) : null}
                </div>
                <span
                  className={`rounded-full border px-2.5 py-1 font-mono text-[11px] ${statusTone(
                    entry.status,
                  )}`}
                >
                  {entry.status || "event"}
                </span>
              </summary>
              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                <TraceBlock title="input" value={entry.input} />
                <TraceBlock title="output" value={entry.output} />
              </div>
            </details>
          ))}
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-slate-700/60 p-6 text-sm leading-6 text-slate-500">
          tool_trace 为空。后端返回工具调用后会显示每一步输入、输出和耗时。
        </div>
      )}

      <div className="mt-5 border-t border-slate-700/50 pt-4">
        <p className="font-mono text-xs uppercase tracking-[0.24em] text-slate-500">Artifacts</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {artifacts.length > 0 ? (
            artifacts.map((artifact, index) => {
              const label = typeof artifact === "string" ? artifact : artifact.name;
              return (
                <span
                  className="rounded-full border border-slate-600/40 bg-slate-500/10 px-3 py-1 font-mono text-xs text-slate-300"
                  key={`${label}-${index}`}
                >
                  {label}
                </span>
              );
            })
          ) : (
            <span className="text-sm text-slate-500">暂无 artifacts</span>
          )}
        </div>
      </div>
    </section>
  );
}

function TraceBlock({ title, value }: { title: string; value: unknown }) {
  return (
    <div className="rounded-2xl bg-graphite-950/80 p-3">
      <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
        {title}
      </p>
      <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-5 text-slate-300">
        {stringifyCompact(value)}
      </pre>
    </div>
  );
}

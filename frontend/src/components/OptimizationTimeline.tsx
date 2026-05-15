import { formatNumber } from "../lib/format";
import type { ObservedHistoryPoint } from "../types";

interface OptimizationTimelineProps {
  history: ObservedHistoryPoint[];
  isLoading: boolean;
}

export function OptimizationTimeline({ history, isLoading }: OptimizationTimelineProps) {
  return (
    <section className="panel-surface rounded-3xl p-5">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.32em] text-slate-500">
            Optimization Timeline
          </p>
          <h2 className="mt-2 text-xl font-semibold text-white">观测历史</h2>
        </div>
        <span className="rounded-full border border-slate-600/40 px-3 py-1 font-mono text-xs text-slate-400">
          {history.length} observations
        </span>
      </div>

      {isLoading ? (
        <div className="h-56 animate-pulse rounded-2xl bg-slate-700/20" />
      ) : history.length > 0 ? (
        <ol className="relative space-y-5 border-l border-slate-700/70 pl-6">
          {history.map((point, index) => (
            <li className="relative" key={`${point.candidate_id || "obs"}-${index}`}>
              <span className="absolute -left-[31px] top-1 h-3 w-3 rounded-full border border-signal-400 bg-graphite-950 shadow-[0_0_18px_rgb(57_255_182_/_0.55)]" />
              <div className="rounded-2xl border border-slate-700/50 bg-black/20 p-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="font-mono text-xs text-slate-500">
                      STEP {point.iteration ?? index + 1}
                    </p>
                    <h3 className="mt-1 text-sm font-semibold text-white">
                      {point.candidate_id || "observed candidate"}
                    </h3>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-sm text-signal-400">
                      {formatNumber(point.objective ?? point.best, 4)}
                    </p>
                    <p className="mt-1 text-xs text-slate-500">objective</p>
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  {Object.entries(point.metrics || {}).map(([key, value]) => (
                    <div className="rounded-xl bg-slate-500/10 px-3 py-2" key={key}>
                      <p className="font-mono text-[10px] uppercase text-slate-500">{key}</p>
                      <p className="mt-1 text-sm text-slate-200">{String(value)}</p>
                    </div>
                  ))}
                </div>
                {point.note || point.timestamp ? (
                  <p className="mt-3 text-xs text-slate-500">{point.note || point.timestamp}</p>
                ) : null}
              </div>
            </li>
          ))}
        </ol>
      ) : (
        <div className="rounded-2xl border border-dashed border-slate-700/60 p-6 text-sm leading-6 text-slate-500">
          observed history 为空。运行 step 后将按迭代顺序记录目标值、指标和候选点。
        </div>
      )}
    </section>
  );
}

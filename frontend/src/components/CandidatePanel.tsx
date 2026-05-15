import { entriesOf, formatNumber, formatPrimitive, statusTone } from "../lib/format";
import type { CandidatePoint } from "../types";

interface CandidatePanelProps {
  candidates: CandidatePoint[];
  isLoading: boolean;
}

export function CandidatePanel({ candidates, isLoading }: CandidatePanelProps) {
  return (
    <section className="panel-surface rounded-3xl p-5">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.32em] text-slate-500">
            Candidate Panel
          </p>
          <h2 className="mt-2 text-xl font-semibold text-white">推荐实验点</h2>
        </div>
        <span className="rounded-full border border-slate-600/40 px-3 py-1 font-mono text-xs text-slate-400">
          {candidates.length} points
        </span>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[0, 1, 2].map((item) => (
            <div className="h-24 animate-pulse rounded-2xl bg-slate-700/20" key={item} />
          ))}
        </div>
      ) : candidates.length > 0 ? (
        <div className="space-y-3">
          {candidates.map((candidate, index) => (
            <article
              className="rounded-2xl border border-slate-700/50 bg-black/20 p-4 transition hover:border-signal-400/40"
              key={candidate.id || candidate.label || index}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-xs text-slate-500">候选 #{index + 1}</p>
                  <h3 className="mt-1 font-semibold text-white">
                    {candidate.label || candidate.id || "未命名配方"}
                  </h3>
                </div>
                <span
                  className={`rounded-full border px-2.5 py-1 font-mono text-[11px] ${statusTone(
                    candidate.status,
                  )}`}
                >
                  {candidate.status || "pending"}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-2">
                <Stat label="score" value={formatNumber(candidate.score, 4)} />
                <Stat label="EI" value={formatNumber(candidate.expected_improvement, 4)} />
                <Stat label="uncertainty" value={formatNumber(candidate.uncertainty, 4)} />
              </div>

              <dl className="mt-4 grid gap-2 text-sm">
                {entriesOf(candidate.parameters).map(([key, value]) => (
                  <div className="flex justify-between gap-4 border-t border-slate-700/40 pt-2" key={key}>
                    <dt className="font-mono text-xs text-slate-500">{key}</dt>
                    <dd className="text-right text-slate-200">{formatPrimitive(value)}</dd>
                  </div>
                ))}
              </dl>

              {candidate.rationale ? (
                <p className="mt-4 rounded-xl bg-slate-500/10 p-3 text-sm leading-6 text-slate-400">
                  {candidate.rationale}
                </p>
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-slate-700/60 p-6 text-sm leading-6 text-slate-500">
          尚无候选点。创建 session 后运行下一步，候选配方会在这里出现。
        </div>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-slate-500/10 p-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-sm text-slate-100">{value}</p>
    </div>
  );
}

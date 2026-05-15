import { formatNumber } from "../lib/format";
import type { OptimizationSession } from "../types";

interface MetricCardsProps {
  session: OptimizationSession | null;
  isLoading: boolean;
}

export function MetricCards({ session, isLoading }: MetricCardsProps) {
  const best = session?.best_result;
  const observed = session?.observed_history || session?.observed || session?.history || [];
  const candidates = session?.candidate_points || [];
  const objective = best?.objective ?? best?.score;

  const metrics = [
    {
      label: "Best Result",
      value: formatNumber(objective, 4),
      helper: best?.summary || best?.target || "等待优化结果",
      accent: "from-signal-400/20",
    },
    {
      label: "候选点",
      value: String(candidates.length),
      helper: "candidate_points",
      accent: "from-cyan-400/16",
    },
    {
      label: "已观测历史",
      value: String(observed.length),
      helper: "observed history",
      accent: "from-caution-400/16",
    },
    {
      label: "迭代步",
      value: String(session?.iteration ?? observed.length ?? 0),
      helper: session?.updated_at ? `更新 ${session.updated_at}` : "等待 session",
      accent: "from-slate-400/12",
    },
  ];

  return (
    <section className="grid gap-3 lg:grid-cols-4">
      {metrics.map((metric) => (
        <article
          className={`panel-surface rounded-3xl bg-gradient-to-br ${metric.accent} to-transparent p-5`}
          key={metric.label}
        >
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-slate-500">
            {metric.label}
          </p>
          <div className="mt-4 flex items-end justify-between gap-4">
            <p className="text-3xl font-semibold text-white">
              {isLoading ? <span className="animate-pulse text-slate-500">...</span> : metric.value}
            </p>
            <div className="h-10 w-1 rounded-full bg-signal-400/70 shadow-[0_0_20px_rgb(57_255_182_/_0.45)]" />
          </div>
          <p className="mt-4 line-clamp-2 min-h-10 text-sm leading-5 text-slate-400">
            {metric.helper}
          </p>
        </article>
      ))}
    </section>
  );
}

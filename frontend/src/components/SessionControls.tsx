import { API_BASE_URL, getSessionId } from "../lib/api";
import { statusTone } from "../lib/format";
import type { OptimizationSession } from "../types";

interface SessionControlsProps {
  selectedTaskId: string;
  session: OptimizationSession | null;
  isCreating: boolean;
  isStepping: boolean;
  error: string | null;
  onCreate: () => void;
  onStep: () => void;
  onRefreshSession: () => void;
}

export function SessionControls({
  selectedTaskId,
  session,
  isCreating,
  isStepping,
  error,
  onCreate,
  onStep,
  onRefreshSession,
}: SessionControlsProps) {
  const sessionId = getSessionId(session);
  const status = session?.status || "idle";

  return (
    <section className="panel-surface rounded-3xl p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.32em] text-slate-500">
            Session Controls
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-white">实验运行控制</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
            API Base: <span className="font-mono text-slate-200">{API_BASE_URL}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className={`rounded-full border px-3 py-1 font-mono text-xs ${statusTone(status)}`}>
            {status}
          </span>
          {sessionId ? (
            <span className="rounded-full border border-slate-600/40 bg-black/20 px-3 py-1 font-mono text-xs text-slate-400">
              {sessionId}
            </span>
          ) : null}
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <button
          className="rounded-2xl bg-signal-400 px-4 py-3 text-sm font-semibold text-graphite-950 transition hover:bg-signal-500 disabled:cursor-not-allowed disabled:opacity-45"
          type="button"
          disabled={!selectedTaskId || isCreating}
          onClick={onCreate}
        >
          {isCreating ? "创建中..." : "创建 Session"}
        </button>
        <button
          className="rounded-2xl border border-signal-400/40 bg-signal-400/10 px-4 py-3 text-sm font-semibold text-signal-400 transition hover:border-signal-400 disabled:cursor-not-allowed disabled:opacity-45"
          type="button"
          disabled={!sessionId || isStepping}
          onClick={onStep}
        >
          {isStepping ? "运行中..." : "运行下一步"}
        </button>
        <button
          className="rounded-2xl border border-slate-600/40 bg-slate-500/10 px-4 py-3 text-sm font-semibold text-slate-200 transition hover:border-slate-400/50 disabled:cursor-not-allowed disabled:opacity-45"
          type="button"
          disabled={!sessionId || isCreating || isStepping}
          onClick={onRefreshSession}
        >
          同步状态
        </button>
      </div>

      {error ? (
        <div className="mt-4 rounded-2xl border border-fault-400/30 bg-fault-400/10 p-3 text-sm text-fault-400">
          Session 操作失败：{error}
        </div>
      ) : null}
    </section>
  );
}

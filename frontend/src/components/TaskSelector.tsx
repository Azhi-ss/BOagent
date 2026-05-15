import type { Task } from "../types";

interface TaskSelectorProps {
  tasks: Task[];
  selectedTaskId: string;
  isLoading: boolean;
  error: string | null;
  onSelect: (taskId: string) => void;
  onRefresh: () => void;
}

function taskLabel(task: Task) {
  return task.title || task.name || task.id;
}

export function TaskSelector({
  tasks,
  selectedTaskId,
  isLoading,
  error,
  onSelect,
  onRefresh,
}: TaskSelectorProps) {
  const selectedTask = tasks.find((task) => task.id === selectedTaskId);

  return (
    <section className="panel-surface rounded-3xl p-5">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.32em] text-signal-400">
            Task Selector
          </p>
          <h2 className="mt-2 text-xl font-semibold text-white">选择 PVK 优化任务</h2>
        </div>
        <button
          className="rounded-full border border-slate-500/30 px-3 py-1.5 text-xs text-slate-300 transition hover:border-signal-400/50 hover:text-white"
          type="button"
          onClick={onRefresh}
          disabled={isLoading}
        >
          {isLoading ? "同步中" : "刷新任务"}
        </button>
      </div>

      <label className="block text-sm text-slate-400" htmlFor="task-select">
        后端任务列表
      </label>
      <select
        id="task-select"
        className="mt-2 w-full rounded-2xl border border-slate-600/40 bg-graphite-950 px-4 py-3 text-sm text-slate-100 transition hover:border-slate-400/40 disabled:opacity-60"
        value={selectedTaskId}
        onChange={(event) => onSelect(event.target.value)}
        disabled={isLoading || tasks.length === 0}
      >
        <option value="">选择一个任务</option>
        {tasks.map((task) => (
          <option key={task.id} value={task.id}>
            {taskLabel(task)}
          </option>
        ))}
      </select>

      {error ? (
        <div className="mt-4 rounded-2xl border border-fault-400/30 bg-fault-400/10 p-3 text-sm text-fault-400">
          任务加载失败：{error}
        </div>
      ) : null}

      <div className="mt-5 min-h-24 rounded-2xl border border-slate-700/40 bg-black/20 p-4">
        {isLoading ? (
          <p className="animate-pulse text-sm text-slate-400">正在读取任务契约...</p>
        ) : selectedTask ? (
          <div className="space-y-3">
            <div>
              <p className="text-sm font-medium text-white">{taskLabel(selectedTask)}</p>
              <p className="mt-1 text-sm leading-6 text-slate-400">
                {selectedTask.description || selectedTask.objective || "后端未返回任务说明。"}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {(selectedTask.parameters || []).map((parameter) => (
                <span
                  className="rounded-full border border-slate-600/40 bg-slate-500/10 px-3 py-1 font-mono text-xs text-slate-300"
                  key={parameter}
                >
                  {parameter}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-sm text-slate-500">等待选择任务后创建 optimization session。</p>
        )}
      </div>
    </section>
  );
}

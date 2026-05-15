import type { DataBoundary } from "../types";

interface DataBoundaryBannerProps {
  boundary: DataBoundary | null | undefined;
  isLoading: boolean;
}

export function DataBoundaryBanner({ boundary, isLoading }: DataBoundaryBannerProps) {
  const warnings = boundary?.warnings || [];
  const constraints = boundary?.constraints || [];

  return (
    <section className="relative overflow-hidden rounded-3xl border border-caution-400/25 bg-caution-400/10 p-5">
      <div className="absolute inset-y-0 left-0 w-1 bg-caution-400" />
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.32em] text-caution-400">
            Data Boundary
          </p>
          <h2 className="mt-2 text-xl font-semibold text-white">数据边界与实验假设</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
            {isLoading
              ? "正在同步 data_boundary..."
              : boundary?.notes ||
                "后端尚未返回 data_boundary。这里会标记训练数据来源、约束、警告和可解释范围。"}
          </p>
        </div>
        <div className="grid min-w-64 grid-cols-3 gap-2 text-center">
          <MiniStat label="rows" value={boundary?.rows} />
          <MiniStat label="train" value={boundary?.train_rows} />
          <MiniStat label="valid" value={boundary?.valid_rows} />
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        <BoundaryCell label="dataset" value={boundary?.dataset || "未声明"} />
        <BoundaryCell label="source" value={boundary?.source || "未声明"} />
        <BoundaryCell label="updated" value={boundary?.last_updated || "未知"} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <ListBlock title="约束条件" emptyText="暂无约束" items={constraints} />
        <ListBlock title="风险提示" emptyText="暂无警告" items={warnings} tone="warning" />
      </div>
    </section>
  );
}

function MiniStat({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="rounded-2xl border border-caution-400/20 bg-black/20 p-3">
      <p className="font-mono text-[10px] uppercase text-caution-400/80">{label}</p>
      <p className="mt-1 font-mono text-lg text-white">{value ?? "--"}</p>
    </div>
  );
}

function BoundaryCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-caution-400/15 bg-black/20 p-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-caution-400/80">
        {label}
      </p>
      <p className="mt-1 text-sm text-slate-100">{value}</p>
    </div>
  );
}

function ListBlock({
  title,
  items,
  emptyText,
  tone = "default",
}: {
  title: string;
  items: string[];
  emptyText: string;
  tone?: "default" | "warning";
}) {
  return (
    <div className="rounded-2xl border border-caution-400/15 bg-black/20 p-4">
      <p className="mb-3 text-sm font-semibold text-white">{title}</p>
      {items.length > 0 ? (
        <ul className="space-y-2 text-sm leading-6 text-slate-300">
          {items.map((item) => (
            <li className="flex gap-2" key={item}>
              <span className={tone === "warning" ? "text-caution-400" : "text-signal-400"}>
                •
              </span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-slate-500">{emptyText}</p>
      )}
    </div>
  );
}

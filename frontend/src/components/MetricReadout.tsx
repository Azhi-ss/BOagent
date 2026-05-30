import { ActivityIcon } from "./Icons";

interface MetricReadoutProps {
  accent: string;
  bestMean: number | null;
  bestStd: number | null;
  genMean: number | null;
  completedSeeds: number;
  totalSeeds: number;
  /** Fine-grained progress: total engine calls completed across all seeds. */
  completedIters?: number;
  /** Fine-grained total: nTrials * nSeeds * 2 (trad + llmbo). */
  totalIters?: number;
  busy?: boolean;
  busyLabel?: string;
}

function Stat({
  label,
  value,
  accent,
  mono = true,
}: {
  label: string;
  value: string;
  accent?: string;
  mono?: boolean;
}) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 10,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "var(--color-ink-500)",
          marginBottom: 6,
          fontWeight: 600,
        }}
      >
        {label}
      </div>
      <div
        key={value}
        className="value-flash"
        style={{
          fontFamily: mono ? "var(--font-mono)" : "inherit",
          fontSize: 22,
          fontWeight: 700,
          color: accent ?? "var(--color-ink-100)",
          lineHeight: 1.1,
          textShadow: accent ? `0 0 16px ${accent}40` : "none",
        }}
      >
        {value}
      </div>
    </div>
  );
}

export function MetricReadout({
  accent,
  bestMean,
  bestStd,
  genMean,
  completedSeeds,
  totalSeeds,
  completedIters,
  totalIters,
  busy = false,
  busyLabel = "计算中 (computing)...",
}: MetricReadoutProps) {
  const fmt = (v: number | null) => (v === null ? "—" : v.toFixed(4));
  const bestText =
    bestMean === null
      ? "—"
      : bestStd !== null
        ? `${bestMean.toFixed(4)} ± ${bestStd.toFixed(3)}`
        : bestMean.toFixed(4);
  // Use fine-grained iter progress when available; fall back to seed-level.
  const progress =
    totalIters && totalIters > 0
      ? Math.min(100, ((completedIters ?? 0) / totalIters) * 100)
      : totalSeeds > 0
        ? Math.min(100, (completedSeeds / totalSeeds) * 100)
        : 0;

  return (
    <div
      style={{
        border: `1px solid ${busy ? accent : "rgba(255, 255, 255, 0.08)"}`,
        borderRadius: 14,
        padding: "16px 18px",
        background: "rgba(15, 23, 42, 0.4)",
        boxShadow: busy ? `0 0 15px -3px ${accent}20` : "none",
        transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        backdropFilter: "blur(8px)",
      }}
    >
      <div style={{ display: "flex", gap: 20 }}>
        <Stat label="Best Score (最优均值 ± 标准差)" value={bestText} accent={accent} />
        <Stat label="Generalization (泛化性能均值)" value={fmt(genMean)} />
      </div>
      
      <div style={{ marginTop: 16 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: 11,
            color: "var(--color-ink-500)",
            marginBottom: 6,
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 500 }}>
            {busy ? (
              <>
                <ActivityIcon size={14} style={{ color: accent }} className="live-dot" />
                <span style={{ color: accent, fontWeight: 600, letterSpacing: "0.02em" }}>{busyLabel}</span>
              </>
            ) : (
              <span>已完成 {completedSeeds}/{totalSeeds} 组种子 (Seeds)</span>
            )}
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
            {totalIters && totalIters > 0
              ? `${completedIters ?? 0}/${totalIters} steps · ${progress.toFixed(0)}%`
              : `${progress.toFixed(0)}%`}
          </span>
        </div>
        
        {/* Styled Progress Bar Container */}
        <div
          style={{
            height: 6,
            borderRadius: 99,
            background: "rgba(255, 255, 255, 0.05)",
            border: "1px solid rgba(255, 255, 255, 0.03)",
            overflow: "hidden",
            position: "relative"
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progress}%`,
              background: `linear-gradient(90deg, ${accent}cc, ${accent})`,
              boxShadow: `0 0 8px ${accent}`,
              borderRadius: 99,
              transition: "width 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          />
        </div>
      </div>
    </div>
  );
}

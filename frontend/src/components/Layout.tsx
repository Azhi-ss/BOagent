export function PanelHeader({ accent, title, subtitle }: { accent: string; title: string; subtitle: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: accent, boxShadow: `0 0 14px ${accent}`, display: "inline-block" }} />
      <div>
        <div style={{ fontSize: 13, fontWeight: 800, letterSpacing: "0.04em", color: accent, fontFamily: "var(--font-display)" }}>{title}</div>
        <div style={{ fontSize: 11, color: "var(--color-ink-500)", marginTop: 2 }}>{subtitle}</div>
      </div>
    </div>
  );
}

export function SummaryCard({
  accent,
  title,
  mean,
  std,
  completedSeeds,
  testId,
}: {
  accent: string;
  title: string;
  mean: number | null;
  std: number | null;
  completedSeeds: number;
  testId?: string;
}) {
  return (
    <div className="panel" data-testid={testId} style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 16 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: "0.02em", fontFamily: "var(--font-display)" }}>{title}</span>
        <span style={{ fontSize: 11, color: "var(--color-ink-500)", fontFamily: "var(--font-mono)" }}>
          {completedSeeds} 个种子均值 (seeds)
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span data-testid="score-value" style={{ fontFamily: "var(--font-mono)", fontSize: 32, fontWeight: 800, color: accent, lineHeight: 1 }}>
          {mean === null ? "—" : mean.toFixed(4)}
        </span>
        {mean !== null && std !== null && (
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 18, color: "var(--color-ink-300)" }}>
            ± {std.toFixed(3)}
          </span>
        )}
      </div>
      <div style={{ marginTop: 12, fontSize: 11, color: "var(--color-ink-500)" }}>
        Best Score 最终收敛值 (均值 ± 标准差)
      </div>
    </div>
  );
}

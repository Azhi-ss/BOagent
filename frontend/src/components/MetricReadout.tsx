interface MetricReadoutProps {
  accent: string;
  best: number | null;
  gen: number | null;
  candidate: number | null;
  iteration: number;
  totalTrials: number;
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
    <div style={{ flex: 1 }}>
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--color-ink-500)",
          marginBottom: 4,
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
          fontSize: 20,
          fontWeight: 700,
          color: accent ?? "var(--color-ink-100)",
          lineHeight: 1.1,
        }}
      >
        {value}
      </div>
    </div>
  );
}

export function MetricReadout({
  accent,
  best,
  gen,
  candidate,
  iteration,
  totalTrials,
}: MetricReadoutProps) {
  const fmt = (v: number | null) => (v === null ? "—" : v.toFixed(4));
  const progress = totalTrials > 0 ? Math.min(100, (iteration / totalTrials) * 100) : 0;

  return (
    <div
      style={{
        border: "1px solid var(--color-graphite-700)",
        borderRadius: 10,
        padding: "12px 14px",
        background: "var(--color-graphite-880)",
      }}
    >
      <div style={{ display: "flex", gap: 12 }}>
        <Stat label="Best Score" value={fmt(best)} accent={accent} />
        <Stat label="Generalization" value={fmt(gen)} />
        <Stat label="Candidate" value={fmt(candidate)} />
      </div>
      <div style={{ marginTop: 12 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 10,
            color: "var(--color-ink-500)",
            marginBottom: 4,
            fontFamily: "var(--font-mono)",
          }}
        >
          <span>ITER {iteration}</span>
          <span>{totalTrials} TRIALS</span>
        </div>
        <div
          style={{
            height: 4,
            borderRadius: 4,
            background: "var(--color-graphite-700)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progress}%`,
              background: accent,
              transition: "width 0.4s ease-out",
            }}
          />
        </div>
      </div>
    </div>
  );
}

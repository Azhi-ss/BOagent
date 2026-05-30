import type { AcquisitionType } from "../types";

interface FieldProps {
  label: string;
  hint?: string;
  children: React.ReactNode;
}

export function Field({ label, hint, children }: FieldProps) {
  return (
    <div>
      <label className="field-label">{label}</label>
      {children}
      {hint && (
        <p style={{ margin: "4px 0 0", fontSize: 11, color: "var(--color-ink-500)" }}>{hint}</p>
      )}
    </div>
  );
}

interface AcqSelectProps {
  value: AcquisitionType;
  onChange: (v: AcquisitionType) => void;
  accent: string;
}

const ACQ_OPTIONS: { value: AcquisitionType; label: string }[] = [
  { value: "ei", label: "Expected Improvement (EI)" },
  { value: "ucb", label: "Upper Confidence Bound (UCB)" },
  { value: "pi", label: "Probability of Improvement (PI)" },
];

export function AcqSelect({ value, onChange, accent }: AcqSelectProps) {
  return (
    <select
      className="field-input"
      value={value}
      onChange={(e) => onChange(e.target.value as AcquisitionType)}
      style={{ borderColor: accent }}
    >
      {ACQ_OPTIONS.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function NumberField({
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <input
      className="field-input"
      type="number"
      value={value}
      step={step}
      min={min}
      max={max}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

import type { JsonMap, Primitive } from "../types";

export function formatNumber(value: number | undefined, digits = 3) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }

  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatPrimitive(value: Primitive | undefined) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }

  if (typeof value === "number") {
    return formatNumber(value, 4);
  }

  return String(value);
}

export function stringifyCompact(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "无";
  }

  if (typeof value === "string") {
    return value;
  }

  return JSON.stringify(value, null, 2);
}

export function entriesOf(map: JsonMap | Record<string, Primitive> | undefined) {
  return Object.entries(map || {}).filter(([, value]) => value !== undefined) as Array<
    [string, Primitive]
  >;
}

export function statusTone(status?: string) {
  const normalized = status?.toLowerCase();

  if (normalized === "failed" || normalized === "error") {
    return "text-fault-400 bg-fault-400/10 border-fault-400/25";
  }

  if (normalized === "running" || normalized === "queued") {
    return "text-caution-400 bg-caution-400/10 border-caution-400/25";
  }

  if (normalized === "success" || normalized === "completed" || normalized === "selected") {
    return "text-signal-400 bg-signal-400/10 border-signal-400/25";
  }

  return "text-slate-300 bg-slate-400/10 border-slate-400/20";
}

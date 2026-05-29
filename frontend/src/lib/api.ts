import type { BenchmarkRequest, BenchmarkResponse, Task } from "../types";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/$/, "");

type Envelope<T> = T | { data: T };
type TasksResponse = Task[] | { tasks?: Task[] };

function unwrap<T>(payload: Envelope<T>): T {
  if (
    payload &&
    typeof payload === "object" &&
    "data" in payload &&
    Object.keys(payload).length <= 2
  ) {
    return (payload as { data: T }).data;
  }
  return payload as T;
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  return text ? (JSON.parse(text) as unknown) : null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  const payload = await readJson(response);
  if (!response.ok) {
    const errorPayload = payload as { error?: { message?: string }; detail?: string } | null;
    throw new Error(
      errorPayload?.error?.message ||
        errorPayload?.detail ||
        `API ${response.status}: ${response.statusText || "request failed"}`,
    );
  }
  return unwrap<T>(payload as Envelope<T>);
}

export async function getTasks() {
  const payload = await request<TasksResponse>("/api/v1/tasks");
  return Array.isArray(payload) ? payload : payload.tasks || [];
}

export function createBenchmarkRun(payload: BenchmarkRequest) {
  return request<BenchmarkResponse>("/api/v1/benchmark", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

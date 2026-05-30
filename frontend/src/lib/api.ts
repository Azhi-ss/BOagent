import type {
  BenchmarkRequest,
  BenchmarkResponse,
  CompareEvent,
  CompareRequest,
  Task,
} from "../types";

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

/**
 * Stream a dual-method comparison via SSE-over-POST.
 *
 * Uses fetch + ReadableStream (not EventSource, which is GET-only) so the
 * full comparison config can be sent in the request body. Invokes `onEvent`
 * for every parsed event. Returns an abort function to cancel the run.
 */
export function streamComparison(
  payload: CompareRequest,
  onEvent: (event: CompareEvent) => void,
  onError: (message: string) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/benchmark/compare/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`Stream failed: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          const jsonText = line.slice(5).trim();
          if (!jsonText) continue;
          try {
            onEvent(JSON.parse(jsonText) as CompareEvent);
          } catch {
            // ignore malformed frame
          }
        }
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      onError(e instanceof Error ? e.message : String(e));
    }
  })();

  return () => controller.abort();
}

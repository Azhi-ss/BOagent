import type {
  ApiErrorPayload,
  BoCurveResponse,
  ChatAgentRequest,
  ChatAgentResponse,
  ChatRequest,
  ChatResponse,
  OptimizationSession,
  PassivationTarget,
  SessionArtifact,
  Task,
} from "../types";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/$/, "");

type Envelope<T> = T | { data: T };
type TasksResponse = Task[] | { tasks?: Task[] };
type ArtifactsResponse =
  | SessionArtifact[]
  | string[]
  | { artifacts?: SessionArtifact[] | string[] }
  | Record<string, unknown>;

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
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });
  const payload = await readJson(response);

  if (!response.ok) {
    const errorPayload = payload as ApiErrorPayload | null;
    throw new Error(
      errorPayload?.error?.message ||
        errorPayload?.detail ||
        `API ${response.status}: ${response.statusText || "请求失败"}`,
    );
  }

  return unwrap<T>(payload as Envelope<T>);
}

export async function getTasks() {
  const payload = await request<TasksResponse>("/api/v1/tasks");
  return Array.isArray(payload) ? payload : payload.tasks || [];
}

export function createSession(taskId: string) {
  return request<OptimizationSession>("/api/v1/sessions", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId }),
  });
}

export function getSession(sessionId: string) {
  return request<OptimizationSession>(`/api/v1/sessions/${sessionId}`);
}

export function runNextStep(sessionId: string) {
  return request<OptimizationSession>(`/api/v1/sessions/${sessionId}/steps`, {
    method: "POST",
  });
}

export async function getArtifacts(sessionId: string) {
  const payload = await request<ArtifactsResponse>(`/api/v1/sessions/${sessionId}/artifacts`);
  if (Array.isArray(payload)) {
    return payload;
  }
  if ("artifacts" in payload && Array.isArray(payload.artifacts)) {
    return payload.artifacts;
  }
  return payload;
}

export function getPassivationTarget(sessionId: string) {
  return request<PassivationTarget>(`/api/v1/sessions/${sessionId}/passivation-target`);
}

export function getBoCurve(sessionId: string) {
  return request<BoCurveResponse | BoCurveResponse["points"]>(
    `/api/v1/sessions/${sessionId}/bo-curve`,
  );
}

export function sendChatMessage(sessionId: string, payload: ChatRequest) {
  return request<ChatResponse>(`/api/v1/sessions/${sessionId}/chat`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function sendAgentChatMessage(payload: ChatAgentRequest) {
  return request<ChatAgentResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSessionId(session: OptimizationSession | null) {
  return session?.session_id || session?.id || "";
}

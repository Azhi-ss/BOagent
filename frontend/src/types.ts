export type Primitive = string | number | boolean | null;

export type JsonMap = Record<string, unknown>;

export interface Task {
  id: string;
  task_id?: string;
  name?: string;
  title?: string;
  description?: string;
  objective?: string;
  metric?: string;
  parameters?: string[];
  metadata?: JsonMap;
  data_available?: boolean;
  data_boundary?: DataBoundary | string;
  source_path?: string;
  record_count?: number | null;
}

export interface BestResult {
  summary?: string;
  objective?: number;
  score?: number;
  target?: string;
  parameters?: Record<string, Primitive>;
  metrics?: Record<string, number | string>;
  iteration?: number;
}

export interface CandidatePoint {
  id?: string;
  label?: string;
  status?: "pending" | "running" | "selected" | "observed" | "rejected" | string;
  score?: number;
  expected_improvement?: number;
  uncertainty?: number;
  parameters?: Record<string, Primitive>;
  rationale?: string;
}

export interface ObservedHistoryPoint {
  iteration?: number;
  candidate_id?: string;
  objective?: number;
  best?: number;
  timestamp?: string;
  metrics?: Record<string, number | string>;
  note?: string;
}

export interface ToolTraceEntry {
  id?: string;
  tool?: string;
  name?: string;
  status?: "queued" | "running" | "success" | "failed" | string;
  duration_ms?: number;
  timestamp?: string;
  input?: unknown;
  output?: unknown;
  message?: string;
}

export interface DataBoundary {
  dataset?: string;
  source?: string;
  rows?: number;
  train_rows?: number;
  valid_rows?: number;
  last_updated?: string;
  constraints?: string[];
  warnings?: string[];
  notes?: string;
}

export interface SessionArtifact {
  name: string;
  type?: string;
  url?: string;
  size?: number;
  metadata?: JsonMap;
}

export interface PassivationCandidate {
  name: string;
  role?: string;
  ratio?: number | string;
  evidence?: string;
  risk?: string;
  evidence_level?: string;
}

export interface PassivationTarget {
  title?: string;
  target?: string;
  objective?: string;
  passivation_ratio?: number;
  candidates?: PassivationCandidate[];
  passivators?: Record<string, Omit<PassivationCandidate, "name">>;
  composition?: Record<string, Primitive>;
  note?: string;
  recommended_strategy?: string;
  data_boundary?: string;
  champion_threshold?: {
    metric?: string;
    operator?: string;
    value?: number;
    unit?: string;
    note?: string;
  };
}

export interface BoCurvePoint {
  iteration: number;
  pvk_bo?: number;
}

export interface BoCurveResponse {
  points?: BoCurvePoint[];
  iterations?: number[];
  pvk_bo?: number[];
  series?:
    | Record<
        "pvk_bo" | string,
        {
          label?: string;
          points?: Array<{ iteration: number; pce: number }>;
          boundary?: string;
        }
      >
    | Array<{ iteration: number; objective?: number; best?: number; pce?: number }>;
  curve_boundary?: string;
}

export type ChatRole = "experimenter" | "agent" | "user" | "assistant" | "system";

export interface ChatMessage {
  role: ChatRole;
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  message: string;
  history?: ChatMessage[];
}

export interface ChatResponse {
  reply?: string;
  assistant_message?: string;
  phase?: string;
  tool_calls?: JsonMap[];
  artifacts?: JsonMap;
  message?: ChatMessage;
  messages?: ChatMessage[];
}

export type ChatAgentState =
  | "idle"
  | "awaiting_data_choice"
  | "awaiting_demo_confirm"
  | "awaiting_goal_confirm"
  | "ready_to_run"
  | "running_bo"
  | "reporting"
  | (string & {});

export type ChatAgentActionType =
  | "none"
  | "select_demo_data"
  | "confirm_demo_data"
  | "confirm_goal"
  | "create_bo_session"
  | "run_bo_step"
  | "explain_result"
  | (string & {});

export interface ChatAgentAction {
  type: ChatAgentActionType;
  args?: JsonMap;
}

export interface ChatAgentRequest {
  conversation_id?: string | null;
  message: string;
  history?: ChatMessage[];
  language?: "zh" | "en";
}

export interface ChatAgentResponse {
  conversation_id: string;
  state: ChatAgentState;
  session_id?: string | null;
  assistant_message?: string;
  messages?: ChatMessage[];
  message?: ChatMessage;
  intent?: string;
  action?: ChatAgentAction;
  ui_hints?: string[];
  tool_calls?: JsonMap[];
  artifacts?: JsonMap;
}

export interface OptimizationSession {
  session_id?: string;
  id?: string;
  task_id?: string;
  status?: "idle" | "running" | "completed" | "failed" | string;
  iteration?: number;
  created_at?: string;
  updated_at?: string;
  best_result?: BestResult | null;
  candidate_points?: CandidatePoint[];
  observed_history?: ObservedHistoryPoint[];
  observed?: ObservedHistoryPoint[];
  history?: ObservedHistoryPoint[];
  tool_trace?: ToolTraceEntry[];
  data_boundary?: DataBoundary | null;
  artifacts?: SessionArtifact[] | string[];
  guardrails?: JsonMap;
  task?: JsonMap;
}

export interface ApiErrorPayload {
  detail?: string;
  error?: {
    code?: string;
    message?: string;
  };
}

export type Primitive = string | number | boolean | null;

export type JsonMap = Record<string, unknown>;

export interface Task {
  id?: string;
  task_id: string;
  name?: string;
  title?: string;
  objective?: string;
  data_source?: string;
  data_available?: boolean;
  data_boundary?: DataBoundary;
  source_path?: string;
  record_count?: number | null;
}

export interface DataBoundary {
  notes?: string;
  dataset?: string;
  source?: string;
  constraints?: string[];
  warnings?: string[];
}

export interface BenchmarkRequest {
  task_id: string;
  n_initial?: number;
  n_trials?: number;
  seed?: number;
  seeds?: number[];
  sm_mode?: "discriminative" | "generative";
  n_candidates?: number;
  n_templates?: number;
  n_gens?: number;
  alpha?: number;
  top_k?: number;
  output_dir?: string;
}

export interface BenchmarkRunResult {
  seed: number;
  best_score: number;
  best_generalization_score: number;
}

export interface BenchmarkResponse {
  task_id: string;
  runs: number;
  results: BenchmarkRunResult[];
  output_dir: string;
}

// --- Comparison (dual-panel real-time) ---

export type AcquisitionType = "ei" | "ucb" | "pi";
export type Method = "traditional" | "llmbo";

export interface TraditionalConfig {
  acquisition: AcquisitionType;
  xi: number;
  kappa: number;
}

export interface LLMBOConfig {
  acquisition: AcquisitionType;
  xi: number;
  kappa: number;
  n_candidates: number;
  n_templates: number;
  top_k: number;
  alpha: number;
}

export interface CompareRequest {
  task_id: string;
  n_initial: number;
  n_trials: number;
  seed: number;
  traditional: TraditionalConfig;
  llmbo: LLMBOConfig;
}

export interface MetaEvent {
  type: "meta";
  task_id: string;
  n_trials: number;
  n_initial: number;
  seed: number;
  feature_cols: string[];
  target_col: string;
}

export interface IterationEvent {
  type: "iteration";
  method: Method;
  iteration: number;
  best_score: number;
  generalization_score: number;
  candidate_score: number | null;
  best_config: Record<string, number>;
  completed: boolean;
}

export interface StepStartEvent {
  type: "step_start";
  method: Method;
  iteration: number;
}

export interface DoneEvent {
  type: "done";
  traditional: Omit<IterationEvent, "type" | "method">;
  llmbo: Omit<IterationEvent, "type" | "method">;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type CompareEvent = MetaEvent | IterationEvent | StepStartEvent | DoneEvent | ErrorEvent;

/** One row of the merged chart series, keyed by iteration. */
export interface ChartPoint {
  iteration: number;
  trad_best?: number;
  trad_gen?: number;
  trad_cand?: number | null;
  llm_best?: number;
  llm_gen?: number;
  llm_cand?: number | null;
}

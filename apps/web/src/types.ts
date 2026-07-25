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
  chat_engine: string;
  use_llm_heuristic?: boolean;
  use_direct_full_pool?: boolean;
  heuristic_weight?: number;
}

export interface CompareRequest {
  task_id: string;
  n_initial: number;
  n_trials: number;
  seeds: number[];
  traditional: TraditionalConfig;
  llmbo: LLMBOConfig;
}

export interface MetaEvent {
  type: "meta";
  task_id: string;
  n_trials: number;
  n_initial: number;
  seeds: number[];
  total_seeds: number;
}

export interface SeedStartEvent {
  type: "seed_start";
  seed: number;
  seed_index: number;
  total_seeds: number;
}

export interface StepStartEvent {
  type: "step_start";
  method: Method;
  seed_index: number;
  total_seeds: number;
  iteration: number;
}

/** One aggregated chart row: per-iteration mean/std across completed seeds. */
export interface AggPoint {
  iteration: number;
  trad_best_mean: number;
  trad_best_std: number;
  trad_gen_mean: number;
  trad_gen_std: number;
  llm_best_mean: number;
  llm_best_std: number;
  llm_gen_mean: number;
  llm_gen_std: number;
}

export interface AggregateEvent {
  type: "aggregate";
  completed_seeds: number;
  total_seeds: number;
  points: AggPoint[];
}

export interface MethodSummary {
  best_mean: number;
  best_std: number;
}

export interface DoneEvent {
  type: "done";
  total_seeds: number;
  summary: {
    traditional: MethodSummary;
    llmbo: MethodSummary;
  };
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type CompareEvent =
  | MetaEvent
  | SeedStartEvent
  | StepStartEvent
  | AggregateEvent
  | DoneEvent
  | ErrorEvent;

/**
 * One row of the chart series for Recharts. Carries the mean lines plus the
 * [lower, upper] band tuples (mean ∓ std) that Area renders as a variance band.
 */
export interface ChartPoint {
  iteration: number;
  trad_best_mean: number;
  trad_best_band: [number, number];
  trad_gen_mean: number;
  llm_best_mean: number;
  llm_best_band: [number, number];
  llm_gen_mean: number;
}

// --- Operational (Human-in-the-loop) ---

export interface OperationalVariable {
  name: string;
  min: number;
  max: number;
  unit: string;
}

export interface OperationalObservation {
  config: Record<string, number>;
  score: number;
}

export interface OperationalSuggestRequest {
  target: string;
  variables: OperationalVariable[];
  history: OperationalObservation[];
  llm_config: LLMBOConfig;
  n_sample?: number;
  seed?: number;
}

export interface OperationalSuggestResponse {
  suggestions: Record<string, number>[];
  analysis: string;
  prompt: string;
}

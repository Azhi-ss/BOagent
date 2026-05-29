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

#!/usr/bin/env python
"""A/B/C benchmark experiment: compare three viability prompt variants.

Variant A (baseline):  Top-5 history only (current code)
Variant B (full_hist): Full observation history
Variant C (full_gp):   Full history + GP mean/std per candidate

Each variant runs 3 seeds (42, 7, 100) on band_alignment task.
Results are saved to backend/tmp_results/prompt_ablation/
"""
from __future__ import annotations

import json
import os
import sys
import time
import copy
from pathlib import Path

# Ensure backend/ is importable
_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from benchmark.runner import BenchmarkRunner

SEEDS = [42, 7, 100]
TASK = "band_alignment"
N_TRIALS = 15
N_INITIAL = 5
OUTPUT_ROOT = _backend / "tmp_results" / "prompt_ablation"


def patch_variant_a():
    """Baseline: Top-5 history (current default). No patch needed."""
    pass


def patch_variant_b():
    """Full history: remove the [:5] slice."""
    from optimization import knowledge as k

    original_build = k.KnowledgeEngine.build_system_prompt_for_viability

    def patched_build(self, target_name, feature_cols, observed_data):
        physical_context, feature_hints = self._get_physical_context_and_hints(
            feature_cols, include_task_prefix=False
        )

        prompt = f"""You are a senior materials scientist specializing in device physics for Perovskite Solar Cells.
Your goal is to evaluate if a candidate formulation is physically viable and likely to achieve high power conversion efficiency ({target_name}).

### Device Physics Rules
{physical_context}

### Parameter Definitions"""
        for col in feature_cols:
            hint = feature_hints.get(col, "Feature parameter")
            prompt += f"\n- {col}: {hint}"

        # ===== VARIANT B: Full history sorted by score descending =====
        sorted_obs = sorted(observed_data, key=lambda x: x[1], reverse=True)
        if sorted_obs:
            prompt += "\n\n### All Observed Formulations (sorted by performance)"
            for i, (obs_values, obs_score) in enumerate(sorted_obs):
                feats_str = ", ".join(
                    f"{k_}={obs_values.get(k_, 0.0):.4f}" for k_ in feature_cols
                )
                prompt += f"\n [{i+1}] {feats_str} -> {target_name}={obs_score:.4f}"

        if len(self.memory) > 0:
            memory_block = self.memory.format_all_for_prompt(max_items=3)
            if memory_block:
                prompt += f"\n\n{memory_block}"

        prompt += "\n\nBased on these rules and guidelines, evaluate the candidate formulation provided in the user message. Answer strictly with either 'Yes' or 'No'."
        return prompt

    k.KnowledgeEngine.build_system_prompt_for_viability = patched_build


def patch_variant_c():
    """Full history + GP predictions injected into user prompt."""
    from optimization import knowledge as k

    # First apply variant B's system prompt patch (full history)
    patch_variant_b()

    # Then patch evaluate_candidate_viability to include GP mean/std
    original_eval = k.KnowledgeEngine.evaluate_candidate_viability

    def patched_eval(self, candidate, system_prompt, feature_cols, gp_mean=None, gp_std=None):
        if not self._client.is_configured():
            return 0.0

        user_prompt = "Candidate formulation:\n"
        for col in feature_cols:
            user_prompt += f"- {col}: {candidate.get(col, 0.0):.4f}\n"

        if "CHI_PVK" in candidate and "CHI_ETL" in candidate:
            cbo = candidate["CHI_PVK"] - candidate["CHI_ETL"]
            user_prompt += f"- Calculated CBO: {cbo:.4f} eV\n"
        if all(kk in candidate for kk in ["CHI_HTL", "Eg_HTL", "CHI_PVK"]):
            vbo = (candidate["CHI_HTL"] + candidate["Eg_HTL"]) - candidate["CHI_PVK"]
            user_prompt += f"- Calculated VBO: {vbo:.4f} eV\n"

        # ===== VARIANT C: inject GP predictions =====
        if gp_mean is not None and gp_std is not None:
            user_prompt += f"\nGP Surrogate Predictions:\n"
            user_prompt += f"- Predicted {self._client.model or 'PCE'} (mean): {gp_mean:.4f}\n"
            user_prompt += f"- Prediction uncertainty (std): {gp_std:.4f}\n"

        user_prompt += "\nIs this candidate formulation physically viable and likely to achieve high PCE? Answer 'Yes' or 'No'."

        from llm_client import LlmCallResult
        result: LlmCallResult = self._client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1,
            extra_body={
                "thinking": {"type": "disabled"},
                "logprobs": True,
                "top_logprobs": 5,
            },
        )

        if result.status != "success" or not result.logprobs:
            return -2.0

        return k.extract_yes_logprob(result.logprobs)

    k.KnowledgeEngine.evaluate_candidate_viability = patched_eval

    # Also patch optimizer.suggest to pass GP mean/std to evaluate_candidate_viability
    from optimization import optimizer as opt

    original_suggest = opt.BayesianOptimizer.suggest

    def patched_suggest(self, top_k=20, n_candidates=5, acquisition="ucb",
                        kappa=2.576, xi=0.01, use_llm=True, gamma=0.1, use_logprobs=True):
        pool_df = self.space.get_unobserved(self.observed_configs)
        scored_df = self._score_candidates(pool_df, acquisition, kappa, xi)
        top_candidates = scored_df.sort_values("score", ascending=False).head(top_k)

        observed_data = []
        for i, (_, row) in enumerate(self.observed_configs.iterrows()):
            observed_data.append((row.to_dict(), self.observed_scores[i]))

        if use_llm and use_logprobs and self.knowledge_engine._client.is_configured() and not top_candidates.empty:
            system_prompt = self.knowledge_engine.build_system_prompt_for_viability(
                self.target_name, self.space.feature_cols, observed_data
            )
            prompt = system_prompt

            candidates_list = [row[self.space.feature_cols].to_dict() for _, row in top_candidates.iterrows()]
            gp_means = top_candidates["mean"].values if "mean" in top_candidates.columns else [None] * len(candidates_list)
            gp_stds = top_candidates["std"].values if "std" in top_candidates.columns else [None] * len(candidates_list)

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates_list)) as executor:
                futures = [
                    executor.submit(
                        self.knowledge_engine.evaluate_candidate_viability,
                        cand, system_prompt, self.space.feature_cols,
                        gp_mean=float(gp_means[idx]) if gp_means[idx] is not None else None,
                        gp_std=float(gp_stds[idx]) if gp_stds[idx] is not None else None,
                    )
                    for idx, cand in enumerate(candidates_list)
                ]
                log_probs = []
                for f in futures:
                    try:
                        log_probs.append(f.result(timeout=30.0))
                    except Exception:
                        log_probs.append(-2.0)

            import numpy as np
            gp_scores = top_candidates["score"].values
            gp_std_val = float(np.std(gp_scores)) if len(gp_scores) > 1 else 1.0
            if gp_std_val == 0:
                gp_std_val = 1.0
            lambda_t = gamma * gp_std_val

            hybrid_scores = gp_scores + lambda_t * np.array(log_probs)

            top_candidates_copy = top_candidates.copy()
            top_candidates_copy["hybrid_score"] = hybrid_scores
            top_candidates_copy["log_prob"] = log_probs
            sorted_candidates = top_candidates_copy.sort_values("hybrid_score", ascending=False)

            from optimization.knowledge import SuggestionResult
            suggestions = [
                row[self.space.feature_cols].to_dict()
                for _, row in sorted_candidates.head(n_candidates).iterrows()
            ]

            analysis_lines = ["Log-probs Hybrid Selection Analysis (with GP context):"]
            for idx2, (_, row) in enumerate(sorted_candidates.head(n_candidates).iterrows()):
                cbo_str = ""
                if "CHI_PVK" in row and "CHI_ETL" in row:
                    cbo = row["CHI_PVK"] - row["CHI_ETL"]
                    cbo_str = f", CBO={cbo:.3f}eV"
                analysis_lines.append(
                    f"Selected Candidate {idx2+1}: GP Score={row['score']:.4f}, LLM Log-prob={row['log_prob']:.4f}, Hybrid Score={row['hybrid_score']:.4f}{cbo_str}"
                )
            analysis = "\n".join(analysis_lines)

            if not suggestions and not top_candidates.empty:
                suggestions = [
                    row[self.space.feature_cols].to_dict()
                    for _, row in top_candidates.head(n_candidates).iterrows()
                ]

            return SuggestionResult(suggestions=suggestions, analysis=analysis, prompt=prompt)
        else:
            # Fallback to original for non-logprob paths
            return original_suggest(self, top_k=top_k, n_candidates=n_candidates,
                                    acquisition=acquisition, kappa=kappa, xi=xi,
                                    use_llm=use_llm, gamma=gamma, use_logprobs=False)

    opt.BayesianOptimizer.suggest = patched_suggest


def run_variant(variant_name: str, patch_fn, output_dir: Path):
    """Run a single variant across all seeds."""
    import importlib
    # Reload modules to reset any patches from previous variant
    from optimization import knowledge, optimizer
    importlib.reload(knowledge)
    importlib.reload(optimizer)

    # Apply this variant's patch
    patch_fn()

    results = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Variant={variant_name}  Seed={seed}")
        print(f"{'='*60}")
        runner = BenchmarkRunner(
            task_id=TASK,
            seed=seed,
            n_initial=N_INITIAL,
            n_trials=N_TRIALS,
            output_dir=str(output_dir / variant_name),
        )
        try:
            result = runner.run()
            runner.save_results(result)
            results.append({
                "seed": seed,
                "best_score": result["best_score"],
                "best_generalization_score": result["best_generalization_score"],
                "convergence_curve": result["fvals"]["score"].tolist(),
                "generalization_curve": result["fvals"]["generalization_score"].tolist(),
            })
            print(f"  -> best_score={result['best_score']:.4f}, gen={result['best_generalization_score']:.4f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({
                "seed": seed,
                "best_score": -1.0,
                "best_generalization_score": -1.0,
                "error": str(e),
            })

    return results


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = {}
    start = time.time()

    # ---- Variant A: Baseline (Top 5 history) ----
    print("\n" + "#" * 70)
    print("  VARIANT A — Baseline (Top 5 history)")
    print("#" * 70)
    all_results["A_baseline_top5"] = run_variant("A_baseline_top5", patch_variant_a, OUTPUT_ROOT)

    # ---- Variant B: Full history ----
    print("\n" + "#" * 70)
    print("  VARIANT B — Full history")
    print("#" * 70)
    all_results["B_full_history"] = run_variant("B_full_history", patch_variant_b, OUTPUT_ROOT)

    # ---- Variant C: Full history + GP predictions ----
    print("\n" + "#" * 70)
    print("  VARIANT C — Full history + GP predictions")
    print("#" * 70)
    all_results["C_full_gp"] = run_variant("C_full_gp", patch_variant_c, OUTPUT_ROOT)

    elapsed = time.time() - start

    # ---- Summary ----
    summary_path = OUTPUT_ROOT / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("  ABLATION EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {summary_path}")

    # Print summary table
    for variant, results in all_results.items():
        scores = [r["best_score"] for r in results if r["best_score"] > 0]
        gens = [r["best_generalization_score"] for r in results if r["best_generalization_score"] > 0]
        if scores:
            import numpy as np
            print(f"\n  {variant}:")
            print(f"    Best Score:  mean={np.mean(scores):.4f} ± {np.std(scores):.4f}  ({scores})")
            print(f"    Gen Score:   mean={np.mean(gens):.4f} ± {np.std(gens):.4f}  ({gens})")


if __name__ == "__main__":
    main()

"""benchmark_agent_team.py

An automated agent team test runner for PVK-BO optimization evaluation.
Coordinates subagents to:
1. [BaselineAgent] Run baseline evaluation and save metrics.
2. [EvaluationAgent] Run evaluation on the improved codebase and save metrics.
3. [AnalyzerAgent] Parse results, compute stats, and generate a markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure backend/ is on sys.path
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from benchmark.comparison import ComparisonRunner

class AgentTeamRunner:
    def __init__(self, task_id: str, seeds: list[int], n_trials: int, n_initial: int) -> None:
        self.task_id = task_id
        self.seeds = seeds
        self.n_trials = n_trials
        self.n_initial = n_initial
        self.baseline_file = _backend_dir / "baseline_results.json"
        self.improved_file = _backend_dir / "improved_results.json"
        self.report_file = _backend_dir / "benchmark_results_comparison.md"

    def run_baseline_agent(self) -> None:
        print("\n=== [BaselineAgent] Starting Baseline Evaluation ===")
        print(f"Task: {self.task_id} | Seeds: {self.seeds} | Trials: {self.n_trials} | Initial: {self.n_initial}")
        
        runner = ComparisonRunner(
            task_id=self.task_id,
            n_initial=self.n_initial,
            n_trials=self.n_trials,
            seeds=self.seeds,
            traditional={},
            llmbo={"n_candidates": 5, "top_k": 20, "alpha": 0.1},
        )
        
        results = self._run_comparison(runner)
        
        with open(self.baseline_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"=== [BaselineAgent] Baseline complete. Saved to {self.baseline_file} ===\n")

    def run_evaluation_agent(self) -> None:
        print("\n=== [EvaluationAgent] Starting Improved Evaluation ===")
        print(f"Task: {self.task_id} | Seeds: {self.seeds} | Trials: {self.n_trials} | Initial: {self.n_initial}")
        
        runner = ComparisonRunner(
            task_id=self.task_id,
            n_initial=self.n_initial,
            n_trials=self.n_trials,
            seeds=self.seeds,
            traditional={},
            llmbo={"n_candidates": 5, "top_k": 20, "alpha": 0.1},
        )
        
        results = self._run_comparison(runner)
        
        with open(self.improved_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"=== [EvaluationAgent] Improved evaluation complete. Saved to {self.improved_file} ===\n")

    def run_analyzer_agent(self) -> None:
        print("\n=== [AnalyzerAgent] Comparing Baseline and Improved Results ===")
        if not self.baseline_file.exists():
            print(f"Error: Baseline results file {self.baseline_file} not found. Run baseline first.")
            return
        if not self.improved_file.exists():
            print(f"Error: Improved results file {self.improved_file} not found. Run improved first.")
            return

        with open(self.baseline_file) as f:
            base = json.load(f)
        with open(self.improved_file) as f:
            imp = json.load(f)

        print("\n--- Summary Comparison Matrix ---")
        print(f"{'Metric':<25} | {'Baseline Traditional':<20} | {'Baseline LLMBO':<20} | {'Improved LLMBO':<20}")
        print("-" * 95)
        
        base_trad_mean = base["summary"]["traditional"]["best_mean"]
        base_trad_std = base["summary"]["traditional"]["best_std"]
        base_llm_mean = base["summary"]["llmbo"]["best_mean"]
        base_llm_std = base["summary"]["llmbo"]["best_std"]
        
        imp_llm_mean = imp["summary"]["llmbo"]["best_mean"]
        imp_llm_std = imp["summary"]["llmbo"]["best_std"]
        
        print(f"{'Best PCE Mean':<25} | {base_trad_mean:20.4f} | {base_llm_mean:20.4f} | {imp_llm_mean:20.4f}")
        print(f"{'Best PCE Std':<25} | {base_trad_std:20.4f} | {base_llm_std:20.4f} | {imp_llm_std:20.4f}")

        # Compute relative improvement of LLMBO over Traditional
        base_lift = ((base_llm_mean - base_trad_mean) / base_trad_mean * 100) if base_trad_mean > 0 else 0
        imp_lift = ((imp_llm_mean - base_trad_mean) / base_trad_mean * 100) if base_trad_mean > 0 else 0
        llmbo_improvement = ((imp_llm_mean - base_llm_mean) / base_llm_mean * 100) if base_llm_mean > 0 else 0
        
        print(f"{'Lift over Trad (%)':<25} | {'-':^20} | {base_lift:19.2f}% | {imp_lift:19.2f}%")
        print("-" * 95)
        print(f"PCE improvement of Improved LLMBO over Baseline LLMBO: {llmbo_improvement:.2f}%")

        # Generate Markdown Report
        markdown_content = f"""# Perovskite Solar Cell Optimization: Comparative Analysis Report

## Executive Summary
This report analyzes the optimization of Perovskite solar cell band alignment under the **baseline** configurations and the **improved** system featuring physics-informed prompts, explicit thinking process reasoning loops, and dynamic scientific memory.

## Multi-seed Comparison (Task: {self.task_id})
- **Seeds evaluated**: {self.seeds}
- **BO Trials**: {self.n_trials}
- **Initial sample size**: {self.n_initial}

### Performance Metrics Table

| Metric | Baseline Traditional BO | Baseline LLMBO | Improved LLMBO (Ours) | Relative LLMBO Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Best PCE Mean** | {base_trad_mean:.4f} | {base_llm_mean:.4f} | {imp_llm_mean:.4f} | **+{llmbo_improvement:.2f}%** |
| **Best PCE Std** | {base_trad_std:.4f} | {base_llm_std:.4f} | {imp_llm_std:.4f} | - |
| **Lift over Traditional** | - | {base_lift:.2f}% | {imp_lift:.2f}% | - |

### Convergence History Analysis
Below is the convergence trajectory (Power Conversion Efficiency / PCE) across seeds.

- **Baseline LLMBO Curve**: {base.get('aggregate_points', [])}
- **Improved LLMBO Curve**: {imp.get('aggregate_points', [])}

## Key Improvements Implemented
1. **Physics-informed Prompts**: Exposes band edge energy computations ($CBM = -\\chi_{{PVK}}$, $VBM = -\\chi_{{PVK}} - 1.6$, etc.) and optimal offsets to guide the LLM's selection.
2. **Structured Reasoning Loop**: Enforces `Thinking Process` generation, making LLM predictions self-reflective and grounded in physical semiconductor constraints.
3. **Dynamic Scientific Memory**: Evaluates historical data points and inputs summaries of high-performing ranges as lessons learned into consecutive iterations.
"""
        with open(self.report_file, "w") as f:
            f.write(markdown_content)
        print(f"=== [AnalyzerAgent] Comparative report written to {self.report_file} ===\n")

    def _run_comparison(self, runner: ComparisonRunner) -> dict[str, Any]:
        results: dict[str, Any] = {
            "runs": {},
            "summary": {},
            "aggregate_points": []
        }
        
        # We collect events from runner.events()
        for event in runner.events():
            t = event["type"]
            if t == "seed_start":
                print(f"  -> Running seed {event['seed']}...")
            elif t == "step_start":
                # Print progress dots
                sys.stdout.write(".")
                sys.stdout.flush()
            elif t == "aggregate":
                print(f"\n  -> Seed completed. Completed: {event['completed_seeds']}/{event['total_seeds']}")
                results["aggregate_points"] = event["points"]
            elif t == "done":
                results["summary"] = event["summary"]
                print(f"  -> All seeds done! Summary: {event['summary']}")
        return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Team Test Coordinator")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "improved", "analyze"])
    parser.add_argument("--seeds", type=str, default="42,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--task", type=str, default="band_alignment")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    runner = AgentTeamRunner(task_id=args.task, seeds=seeds, n_trials=args.trials, n_initial=args.initial)
    
    if args.mode == "baseline":
        runner.run_baseline_agent()
    elif args.mode == "improved":
        runner.run_evaluation_agent()
    elif args.mode == "analyze":
        runner.run_analyzer_agent()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publication-quality Benchmark Visualization Tool for BOagent.

Generates:
1. Convergence Curves with Standard Error / Standard Deviation Shaded Bands.
2. t95 Iteration Distribution Histograms / Violin plots.
3. Combined 1x2 Publication Dashboard (300 DPI PNG/SVG).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# Premium Tech Palette (Tailwind HSL / Hex Palette)
COLOR_PALETTE = {
    "LGBO": {"main": "#10B981", "shade": "#10B98126", "label": "LGBO (Ours)"},
    "GPBO": {"main": "#F59E0B", "shade": "#F59E0B26", "label": "Standard GPBO"},
    "Random": {"main": "#6B7280", "shade": "#6B728026", "label": "Random Search"},
}


def plot_benchmark_dashboard(
    convergence_data: Dict[str, np.ndarray],
    t95_data: Dict[str, List[int]],
    global_best: float,
    title_prefix: str = "Benchmark",
    output_path: str | Path = "benchmark_dashboard.png"
):
    """Generate a 1x2 publication-grade dashboard figure.
    
    Args:
        convergence_data: Dict mapping method name (e.g. 'LGBO') to 2D array of shape (n_seeds, n_rounds).
        t95_data: Dict mapping method name to list of t95 steps for each seed.
        global_best: Scalar value representing the theoretical/dataset global best.
        title_prefix: Dataset or benchmark name for the title.
        output_path: Path to save PNG/SVG figure.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    
    # ----------------------------------------------------
    # Subplot 1: Convergence Trajectory (Mean + Shaded Std)
    # ----------------------------------------------------
    for method, seeds_matrix in convergence_data.items():
        style = COLOR_PALETTE.get(method, {"main": "#3B82F6", "shade": "#3B82F626", "label": method})
        mean_trajectory = np.mean(seeds_matrix, axis=0)
        std_trajectory = np.std(seeds_matrix, axis=0)
        rounds = np.arange(1, len(mean_trajectory) + 1)
        
        ax1.plot(
            rounds, mean_trajectory, 
            color=style["main"], label=style["label"], 
            linewidth=2.5
        )
        ax1.fill_between(
            rounds, 
            mean_trajectory - std_trajectory, 
            mean_trajectory + std_trajectory, 
            color=style["main"], alpha=0.15
        )
        
    ax1.axhline(global_best, color="#EF4444", linestyle="--", alpha=0.85, label=f"Global Best ({global_best})")
    ax1.set_xlabel("Optimization Round", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Target Yield (%)", fontsize=11, fontweight='bold')
    ax1.set_title(f"{title_prefix} - A. Convergence Trajectory & Stability", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # ----------------------------------------------------
    # Subplot 2: t95 Distribution Comparison (Bar/Violin)
    # ----------------------------------------------------
    methods = list(t95_data.keys())
    x_positions = np.arange(len(methods))
    t95_means = [np.mean(t95_data[m]) for m in methods]
    t95_stds = [np.std(t95_data[m]) for m in methods]
    bar_colors = [COLOR_PALETTE.get(m, {}).get("main", "#3B82F6") for m in methods]
    
    bars = ax2.bar(
        x_positions, t95_means, yerr=t95_stds, 
        capsize=5, color=bar_colors, alpha=0.85, width=0.5,
        edgecolor='black', linewidth=1.2
    )
    
    # Value annotations on top of bars
    for bar, mean_val in zip(bars, t95_means):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, height + 1.0, 
            f"Mean: {mean_val:.1f}", 
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
        
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([COLOR_PALETTE.get(m, {}).get("label", m) for m in methods], fontsize=11, fontweight='bold')
    ax2.set_ylabel("Rounds to 95% Global Best (t95)", fontsize=11, fontweight='bold')
    ax2.set_title(f"{title_prefix} - B. t95 Efficiency (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_p, bbox_inches='tight')
    plt.close()
    print(f"[Plotter] Dashboard successfully saved to: {output_p.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison charts")
    parser.add_argument("--json", type=str, help="Path to benchmark results JSON file")
    parser.add_argument("--out", type=str, default="benchmark_dashboard.png", help="Output PNG/SVG file path")
    parser.add_argument("--title", type=str, default="Benchmark", help="Dataset/Benchmark title")
    parser.add_argument("--global_best", type=float, default=99.90, help="Theoretical global best value")
    args = parser.parse_args()

    if args.json and Path(args.json).exists():
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Extract matrices if provided
        conv_data = {}
        t95_data = {}
        for method, res in data.items():
            if "trajectories" in res:
                conv_data[method] = np.array(res["trajectories"])
            if "t95_list" in res:
                t95_data[method] = res["t95_list"]
                
        plot_benchmark_dashboard(conv_data, t95_data, args.global_best, args.title, args.out)
    else:
        print("[Plotter] Template mode executed. Call plot_benchmark_dashboard() programmatically or supply --json.")


if __name__ == "__main__":
    main()

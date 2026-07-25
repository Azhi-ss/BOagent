#!/usr/bin/env python3
"""Rational & Scalable 1x2 Benchmark Dashboard Plotter for BOagent.

CLI & Programmatic entry point for multi-algorithm benchmark visualizations.
Uses Matplotlib, Seaborn, and adjustText.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

COLOR_PRIMARY = "#10B981"     # Vivid Emerald Green for LGBO (Ours)
COLOR_SECONDARY = ["#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6"]
COLOR_BASELINE = "#9CA3AF"    # Cool Gray for Random Search / Simple Baselines


def render_sample_rational_dashboard(
    output_path: str | Path = "scripts/rational_benchmark_dashboard.png"
) -> Path:
    """Generate simulated 5-algorithm benchmark data to demonstrate plot layout."""
    np.random.seed(42)
    n_seeds = 20
    n_rounds = 40
    global_best = 99.90
    t95_target = global_best * 0.95

    algorithms = [
        ("LGBO (Ours)", 25, 74, True),
        ("LGBO-NoPhysics", 28, 68, False),
        ("GPBO (Standard)", 30, 65, False),
        ("TuRBO-BO", 32, 62, False),
        ("Random Search", 40, 50, False),
    ]

    conv_data = {}
    t95_data = {}

    for name, beta_b, offset, is_ours in algorithms:
        matrix = np.zeros((n_seeds, n_rounds))
        t95_list = []
        for s in range(n_seeds):
            base = np.sort(np.random.beta(a=2 if is_ours else 1.2, b=1.5, size=n_rounds)) * beta_b + offset
            noise = np.cumsum(np.random.normal(0, 0.4 if is_ours else 0.8, size=n_rounds))
            traj = np.minimum(global_best, base + noise)
            traj = np.maximum.accumulate(traj)
            matrix[s] = traj
            hit = np.where(traj >= t95_target)[0]
            step = hit[0] + 1 if len(hit) > 0 else 41
            t95_list.append(step)
        conv_data[name] = matrix
        t95_data[name] = t95_list

    return plot_rational_dashboard(
        conv_data, t95_data, global_best, title_prefix="Suzuki-Miyaura Perovskite Optimization", output_path=output_path
    )


def plot_rational_dashboard(
    convergence_data: Dict[str, np.ndarray],
    t95_data: Dict[str, List[int]],
    global_best: float,
    title_prefix: str = "Benchmark",
    output_path: str | Path = "benchmark_dashboard.png"
) -> Path:
    """Render a clean 1x2 dashboard for multi-algorithm comparisons.
    
    Panel A: Dynamic Convergence Trajectories (Mean ± Std).
    Panel B: t95 Efficiency Horizontal Bar Chart (Rounds to 95%).
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=300)
    t95_target = global_best * 0.95

    texts = []
    sec_idx = 0
    method_colors = {}

    # Panel A: Trajectory
    for name, seeds_matrix in convergence_data.items():
        mean_traj = np.mean(seeds_matrix, axis=0)
        std_traj = np.std(seeds_matrix, axis=0)
        rounds = np.arange(1, len(mean_traj) + 1)

        is_ours = "Ours" in name or ("LGBO" in name and "No" not in name)
        is_baseline = "Random" in name

        if is_ours:
            color = COLOR_PRIMARY
            lw, ls, alpha_shade, zorder = 3.2, "-", 0.18, 10
        elif is_baseline:
            color = COLOR_BASELINE
            lw, ls, alpha_shade, zorder = 1.6, ":", 0.0, 2
        else:
            color = COLOR_SECONDARY[sec_idx % len(COLOR_SECONDARY)]
            sec_idx += 1
            lw, ls, alpha_shade, zorder = 2.0, "-", 0.06, 5

        method_colors[name] = color

        ax1.plot(rounds, mean_traj, color=color, linestyle=ls, label=name, linewidth=lw, zorder=zorder)
        if alpha_shade > 0:
            ax1.fill_between(rounds, mean_traj - std_traj, mean_traj + std_traj, color=color, alpha=alpha_shade, zorder=zorder-1)

        end_val = mean_traj[-1]
        txt = ax1.text(rounds[-1], end_val, f" {name} ({end_val:.1f}%)", color=color, fontweight='bold' if is_ours else 'normal', fontsize=9)
        texts.append(txt)

    ax1.axhline(global_best, color="#EF4444", linestyle="--", linewidth=1.2, alpha=0.85, label=f"Global Best ({global_best}%)")
    ax1.axhline(t95_target, color="#F59E0B", linestyle=":", linewidth=1.2, alpha=0.8, label=f"95% Target ({t95_target:.1f}%)")

    ax1.set_xlabel("Optimization Round", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Yield (%)", fontsize=11, fontweight='bold')
    ax1.set_title("A. Optimization Convergence Trajectories (Mean ± Std)", fontsize=12, fontweight='bold', pad=12)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right', fontsize=8.5)
    ax1.set_ylim(40, 103)
    adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    # Panel B: Metric Summary
    methods = list(convergence_data.keys())
    y_pos = np.arange(len(methods))
    t95_means = [np.mean(t95_data[m]) for m in methods]
    t95_stds = [np.std(t95_data[m]) for m in methods]
    palette_list = [method_colors[m] for m in methods]

    bars = ax2.barh(y_pos, t95_means, xerr=t95_stds, capsize=4, color=palette_list, alpha=0.85, edgecolor='black', linewidth=1.0, height=0.55)
    
    for bar, mean_val, std_val in zip(bars, t95_means, t95_stds):
        width = bar.get_width()
        top_pos = width + std_val + 0.8
        ax2.text(top_pos, bar.get_y() + bar.get_height()/2.0, f"{mean_val:.1f} rounds", va='center', fontweight='bold', fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=10, fontweight='bold')
    ax2.set_xlabel("Average Rounds to 95% Best (t95, Lower is Better)", fontsize=11, fontweight='bold')
    ax2.set_title("B. Convergence Efficiency (t95 Steps ± Std)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlim(0, 48)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f"{title_prefix} Benchmark Suite", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f"[Plotter] Dashboard successfully saved to: {out_p}")
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Rational 1x2 Multi-Algorithm Benchmark Plotter")
    parser.add_argument("--json", type=str, help="Path to benchmark results JSON file")
    parser.add_argument("--sample", action="store_true", help="Generate sample benchmark dashboard")
    parser.add_argument("--out", type=str, default="scripts/rational_benchmark_dashboard.png", help="Output PNG file path")
    parser.add_argument("--title", type=str, default="Benchmark", help="Dataset/Benchmark title")
    parser.add_argument("--global_best", type=float, default=99.90, help="Theoretical global best value")
    args = parser.parse_args()

    if args.sample or not args.json:
        out_path = render_sample_rational_dashboard(args.out)
    else:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        conv_data = {m: np.array(v["trajectories"]) for m, v in data.items() if "trajectories" in v}
        t95_data = {m: v["t95_list"] for m, v in data.items() if "t95_list" in v}
        out_path = plot_rational_dashboard(conv_data, t95_data, args.global_best, args.title, args.out)

    artifact_dir = Path("/home/dministrator/.gemini/antigravity-ide/brain/70e521f9-1fa3-4822-acad-65432d1caf08")
    if artifact_dir.exists():
        dest = artifact_dir / out_path.name
        shutil.copy(out_path, dest)
        print(f"[Plotter] Copied figure to artifact dir: {dest}")


if __name__ == "__main__":
    main()

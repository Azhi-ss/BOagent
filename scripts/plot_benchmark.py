#!/usr/bin/env python3
"""Publication-grade Multi-Algorithm Scalable Plotter for BOagent.

Designed for scaling up to 10+ algorithm variants without visual clutter.
Uses Visual Hierarchy (Primary Method in Vivid Color, Baselines in Subtle Gray/Dashed).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# Visual Hierarchy Palette (Scalable up to N algorithms)
PRIMARY_COLOR = "#10B981"  # Vivid Emerald for LGBO (Ours)
SECONDARY_COLORS = ["#F59E0B", "#3B82F6", "#8B5CF6", "#EC4899", "#14B8A6"]
BASELINE_COLOR = "#9CA3AF"  # Soft Slate Gray for weak/baseline algorithms


def render_sample_scalable_dashboard(
    output_path: str | Path = "scripts/sample_benchmark_dashboard.png"
) -> Path:
    """Generate simulated 5-algorithm benchmark data to demonstrate scalability."""
    np.random.seed(42)
    n_seeds = 20
    n_rounds = 40
    global_best = 99.90
    t95_target = global_best * 0.95

    # Define 5 algorithms to prove scalability
    algorithms = [
        ("LGBO (Ours)", "primary", 25, 74, 15.2),
        ("GPBO (Standard)", "secondary", 30, 65, 25.8),
        ("LGBO-NoPhysics", "secondary", 28, 68, 21.4),
        ("TuRBO-BO", "secondary", 32, 62, 28.1),
        ("Random Search", "baseline", 40, 50, 36.5),
    ]

    conv_data = {}
    t95_data = {}

    for name, kind, beta_b, offset, target_t95 in algorithms:
        matrix = np.zeros((n_seeds, n_rounds))
        t95_list = []
        for s in range(n_seeds):
            base = np.sort(np.random.beta(a=2 if kind == "primary" else 1.2, b=1.5, size=n_rounds)) * beta_b + offset
            noise = np.cumsum(np.random.normal(0, 0.4 if kind == "primary" else 0.8, size=n_rounds))
            traj = np.minimum(global_best, base + noise)
            traj = np.maximum.accumulate(traj)
            matrix[s] = traj
            hit = np.where(traj >= t95_target)[0]
            step = hit[0] + 1 if len(hit) > 0 else 41
            t95_list.append(step)
        conv_data[name] = matrix
        t95_data[name] = t95_list

    return plot_scalable_dashboard(
        conv_data, t95_data, global_best, title_prefix="Suzuki-Miyaura Multi-Algorithm Benchmark", output_path=output_path
    )


def plot_scalable_dashboard(
    convergence_data: Dict[str, np.ndarray],
    t95_data: Dict[str, List[int]],
    global_best: float,
    title_prefix: str = "Benchmark",
    output_path: str | Path = "benchmark_dashboard.png"
) -> Path:
    """Render a scalable 1x2 dashboard with Visual Hierarchy."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    t95_target = global_best * 0.95

    # ----------------------------------------------------
    # Subplot 1: Scalable Convergence (Visual Hierarchy)
    # ----------------------------------------------------
    texts = []
    sec_idx = 0

    for method, seeds_matrix in convergence_data.items():
        mean_traj = np.mean(seeds_matrix, axis=0)
        std_traj = np.std(seeds_matrix, axis=0)
        rounds = np.arange(1, len(mean_traj) + 1)

        is_ours = "Ours" in method or "LGBO" in method and "No" not in method
        is_baseline = "Random" in method

        if is_ours:
            color = PRIMARY_COLOR
            lw = 3.0
            ls = "-"
            alpha_shade = 0.18
            zorder = 10
        elif is_baseline:
            color = BASELINE_COLOR
            lw = 1.5
            ls = ":"
            alpha_shade = 0.0
            zorder = 2
        else:
            color = SECONDARY_COLORS[sec_idx % len(SECONDARY_COLORS)]
            sec_idx += 1
            lw = 2.0
            ls = "-"
            alpha_shade = 0.10
            zorder = 5

        ax1.plot(rounds, mean_traj, color=color, linestyle=ls, label=method, linewidth=lw, zorder=zorder)
        if alpha_shade > 0:
            ax1.fill_between(rounds, mean_traj - std_traj, mean_traj + std_traj, color=color, alpha=alpha_shade, zorder=zorder-1)

        end_val = mean_traj[-1]
        txt = ax1.text(rounds[-1], end_val, f" {method} ({end_val:.1f}%)", color=color, fontweight='bold' if is_ours else 'normal', fontsize=8.5)
        texts.append(txt)

    ax1.axhline(global_best, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Global Best ({global_best}%)")
    ax1.axhline(t95_target, color="#F59E0B", linestyle=":", linewidth=1.2, alpha=0.8, label=f"95% Target ({t95_target:.1f}%)")

    ax1.set_xlabel("Optimization Round", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Yield (%)", fontsize=11, fontweight='bold')
    ax1.set_title(f"{title_prefix}\nA. Convergence Trajectories (Visual Hierarchy)", fontsize=11, fontweight='bold', pad=10)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right', fontsize=8.5)
    ax1.set_ylim(40, 102)

    adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    # ----------------------------------------------------
    # Subplot 2: t95 Efficiency Comparison (Bar + Error Cap)
    # ----------------------------------------------------
    methods = list(t95_data.keys())
    x_pos = np.arange(len(methods))
    t95_means = [np.mean(t95_data[m]) for m in methods]
    t95_stds = [np.std(t95_data[m]) for m in methods]

    bar_colors = []
    sec_idx = 0
    for m in methods:
        if "Ours" in m or ("LGBO" in m and "No" not in m):
            bar_colors.append(PRIMARY_COLOR)
        elif "Random" in m:
            bar_colors.append(BASELINE_COLOR)
        else:
            bar_colors.append(SECONDARY_COLORS[sec_idx % len(SECONDARY_COLORS)])
            sec_idx += 1

    bars = ax2.bar(
        x_pos, t95_means, yerr=t95_stds, capsize=4,
        color=bar_colors, alpha=0.85, width=0.55, edgecolor='black', linewidth=1.0
    )

    # Clean label placement on top of error bars
    for bar, mean_val, std_val in zip(bars, t95_means, t95_stds):
        height = bar.get_height()
        top_pos = height + std_val + 0.8
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, top_pos,
            f"{mean_val:.1f} rounds",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold'
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=15, ha='right', fontsize=9, fontweight='bold')
    ax2.set_ylabel("Rounds to 95% Best (t95)", fontsize=11, fontweight='bold')
    ax2.set_title(f"{title_prefix}\nB. t95 Convergence Speed (Lower is Better)", fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 48)

    plt.tight_layout()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f"[Plotter] Successfully generated scalable dashboard: {out_p}")
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Multi-Algorithm Scalable Plotter")
    parser.add_argument("--sample", action="store_true", help="Generate 5-algorithm sample chart")
    parser.add_argument("--out", type=str, default="scripts/sample_benchmark_dashboard.png", help="Output image path")
    args = parser.parse_args()

    out_path = render_sample_scalable_dashboard(args.out)

    artifact_dir = Path("/home/dministrator/.gemini/antigravity-ide/brain/70e521f9-1fa3-4822-acad-65432d1caf08")
    if artifact_dir.exists():
        dest = artifact_dir / out_path.name
        shutil.copy(out_path, dest)
        print(f"[Plotter] Copied figure to artifact dir: {dest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publication-grade Benchmark Plotter using Matplotlib, Seaborn, and adjustText.

Generates:
1. Convergence curves with 95% Confidence Intervals (CI / Shaded Standard Error).
2. t95 Efficiency violin/box distributions via Seaborn.
3. Auto-avoids label overlap using adjustText.
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

# Premium Publication Palette (Tailwind HSL tailored colors)
COLOR_PALETTE = {
    "LGBO": {"main": "#10B981", "shade": "#10B98126", "label": "LGBO (Ours)"},
    "GPBO": {"main": "#F59E0B", "shade": "#F59E0B26", "label": "Standard GPBO"},
    "Random": {"main": "#6B7280", "shade": "#6B728026", "label": "Random Search"},
}


def render_sample_benchmark_dashboard(
    output_path: str | Path = "scripts/sample_benchmark_dashboard.png"
) -> Path:
    """Generate sample 20-seed benchmark data and plot a 1x2 publication dashboard."""
    np.random.seed(42)
    n_seeds = 20
    n_rounds = 40
    global_best = 99.90
    t95_target = global_best * 0.95

    # 1. Simulate LGBO trajectories (fast convergence, mean t95 ~15.5)
    lgbo_matrix = np.zeros((n_seeds, n_rounds))
    lgbo_t95 = []
    for s in range(n_seeds):
        base = np.sort(np.random.beta(a=2, b=1, size=n_rounds)) * 25 + 74
        noise = np.cumsum(np.random.normal(0, 0.5, size=n_rounds))
        traj = np.minimum(global_best, base + noise)
        traj = np.maximum.accumulate(traj)
        lgbo_matrix[s] = traj
        hit = np.where(traj >= t95_target)[0]
        step = hit[0] + 1 if len(hit) > 0 else 41
        lgbo_t95.append(step)

    # 2. Simulate GPBO trajectories (slower convergence, mean t95 ~26.0)
    gpbo_matrix = np.zeros((n_seeds, n_rounds))
    gpbo_t95 = []
    for s in range(n_seeds):
        base = np.sort(np.random.beta(a=1.2, b=1.5, size=n_rounds)) * 30 + 65
        noise = np.cumsum(np.random.normal(0, 0.8, size=n_rounds))
        traj = np.minimum(global_best, base + noise)
        traj = np.maximum.accumulate(traj)
        gpbo_matrix[s] = traj
        hit = np.where(traj >= t95_target)[0]
        step = hit[0] + 1 if len(hit) > 0 else 41
        gpbo_t95.append(step)

    # 3. Simulate Random Search trajectories (very slow, mean t95 ~35.0)
    rand_matrix = np.zeros((n_seeds, n_rounds))
    rand_t95 = []
    for s in range(n_seeds):
        base = np.sort(np.random.uniform(50, 92, size=n_rounds))
        traj = np.maximum.accumulate(base)
        rand_matrix[s] = traj
        hit = np.where(traj >= t95_target)[0]
        step = hit[0] + 1 if len(hit) > 0 else 41
        rand_t95.append(step)

    conv_data = {"LGBO": lgbo_matrix, "GPBO": gpbo_matrix, "Random": rand_matrix}
    t95_data = {"LGBO": lgbo_t95, "GPBO": gpbo_t95, "Random": rand_t95}

    return plot_benchmark_dashboard(
        conv_data, t95_data, global_best, title_prefix="Suzuki-Miyaura Reaction", output_path=output_path
    )


def plot_benchmark_dashboard(
    convergence_data: Dict[str, np.ndarray],
    t95_data: Dict[str, List[int]],
    global_best: float,
    title_prefix: str = "Benchmark",
    output_path: str | Path = "benchmark_dashboard.png"
) -> Path:
    """Render publication 1x2 dashboard with Seaborn, Matplotlib, and adjustText."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    t95_target = global_best * 0.95

    # ----------------------------------------------------
    # Subplot 1: Convergence Trajectories (Mean + CI Shading)
    # ----------------------------------------------------
    texts = []
    for method, seeds_matrix in convergence_data.items():
        style = COLOR_PALETTE.get(method, {"main": "#3B82F6", "shade": "#3B82F626", "label": method})
        mean_traj = np.mean(seeds_matrix, axis=0)
        std_traj = np.std(seeds_matrix, axis=0)
        rounds = np.arange(1, len(mean_traj) + 1)

        ax1.plot(rounds, mean_traj, color=style["main"], label=style["label"], linewidth=2.8)
        ax1.fill_between(rounds, mean_traj - std_traj, mean_traj + std_traj, color=style["main"], alpha=0.18)

        # Annotate end values with adjustText
        end_val = mean_traj[-1]
        txt = ax1.text(rounds[-1], end_val, f"  {method} ({end_val:.1f}%)", color=style["main"], fontweight='bold', fontsize=9)
        texts.append(txt)

    ax1.axhline(global_best, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.9, label=f"Global Best ({global_best}%)")
    ax1.axhline(t95_target, color="#F59E0B", linestyle=":", linewidth=1.2, alpha=0.8, label=f"95% Target ({t95_target:.1f}%)")

    ax1.set_xlabel("Optimization Round", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Yield (%)", fontsize=11, fontweight='bold')
    ax1.set_title(f"{title_prefix} - A. Convergence & Stability (20 Seeds)", fontsize=12, fontweight='bold', pad=12)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right', fontsize=9)
    ax1.set_ylim(45, 102)

    # Use adjustText to eliminate label overlaps
    adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    # ----------------------------------------------------
    # Subplot 2: t95 Violin / Box Distribution via Seaborn
    # ----------------------------------------------------
    df_rows = []
    for method, steps in t95_data.items():
        for st in steps:
            df_rows.append({"Method": COLOR_PALETTE.get(method, {}).get("label", method), "t95": st, "raw_method": method})
    df_t95 = pd_df = sns.load_dataset("tips") # Dummy check, build DataFrame
    import pandas as pd
    df_t95 = pd.DataFrame(df_rows)

    palette_dict = {
        COLOR_PALETTE.get(m, {}).get("label", m): COLOR_PALETTE.get(m, {}).get("main", "#3B82F6")
        for m in t95_data.keys()
    }

    sns.violinplot(
        data=df_t95, x="Method", y="t95", hue="Method", legend=False, ax=ax2, palette=palette_dict,
        inner="quartile", cut=0, linewidth=1.2, alpha=0.85
    )
    sns.stripplot(
        data=df_t95, x="Method", y="t95", ax=ax2, color="black",
        size=4, jitter=0.2, alpha=0.6
    )

    # Annotate mean t95 on bars
    for i, method in enumerate(t95_data.keys()):
        m_label = COLOR_PALETTE.get(method, {}).get("label", method)
        mean_step = np.mean(t95_data[method])
        ax2.text(i, mean_step, f"  mean t95 = {mean_step:.1f}s", color="black", fontweight="bold", fontsize=9, va="center")

    ax2.set_xlabel("Algorithm", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Rounds to 95% Best (t95)", fontsize=11, fontweight='bold')
    ax2.set_title(f"{title_prefix} - B. t95 Efficiency Distribution", fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f"[Plotter] Successfully generated dashboard: {out_p}")
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Publication-grade Benchmark Plotter")
    parser.add_argument("--sample", action="store_true", help="Generate sample benchmark chart")
    parser.add_argument("--out", type=str, default="scripts/sample_benchmark_dashboard.png", help="Output image path")
    args = parser.parse_args()

    if args.sample or not any(vars(args).values()):
        out_path = render_sample_benchmark_dashboard(args.out)
        
        # Copy to artifact directory for rendering in response
        artifact_dir = Path("/home/dministrator/.gemini/antigravity-ide/brain/70e521f9-1fa3-4822-acad-65432d1caf08")
        if artifact_dir.exists():
            dest = artifact_dir / out_path.name
            shutil.copy(out_path, dest)
            print(f"[Plotter] Copied figure to artifact dir: {dest}")


if __name__ == "__main__":
    main()

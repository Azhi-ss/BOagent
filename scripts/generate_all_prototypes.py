#!/usr/bin/env python3
"""Generate 3 visual prototypes for Multi-Algorithm Benchmark Comparison:

Prototype 1: Visual Hierarchy (Primary Method Vivid + Shaded, Baselines Subtle)
Prototype 2: Small Multiples / Facet Grid (2x2 Grid comparing Ours vs Each Baseline)
Prototype 3: Relative Boost & Speedup Dashboard (Normalized Speedup + Final Yield Boxplot)
"""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

ARTIFACT_DIR = Path("/home/dministrator/.gemini/antigravity-ide/brain/70e521f9-1fa3-4822-acad-65432d1caf08")
OUT_DIR = Path("scripts")

# Simulated Data Generator (5 algorithms, 20 seeds, 40 rounds)
def generate_simulated_benchmark_data():
    np.random.seed(42)
    n_seeds = 20
    n_rounds = 40
    global_best = 99.90
    t95_target = global_best * 0.95

    algorithms = [
        ("LGBO (Ours)", 25, 74, 15.2),
        ("GPBO (Standard)", 30, 65, 25.8),
        ("LGBO-NoPhysics", 28, 68, 21.4),
        ("TuRBO-BO", 32, 62, 28.1),
        ("Random Search", 40, 50, 36.5),
    ]

    conv_data = {}
    t95_data = {}

    for name, beta_b, offset, target_t95 in algorithms:
        matrix = np.zeros((n_seeds, n_rounds))
        t95_list = []
        is_ours = "Ours" in name
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

    return conv_data, t95_data, global_best


# ==============================================================================
# PROTOTYPE 1: Visual Hierarchy (Primary Vivid, Baselines Subtle)
# ==============================================================================
def render_prototype_1(conv_data, t95_data, global_best):
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    t95_target = global_best * 0.95

    primary_color = "#10B981"
    secondary_colors = ["#F59E0B", "#3B82F6", "#8B5CF6"]
    baseline_color = "#9CA3AF"

    texts = []
    sec_idx = 0
    bar_colors = []

    for method, seeds_matrix in conv_data.items():
        mean_traj = np.mean(seeds_matrix, axis=0)
        std_traj = np.std(seeds_matrix, axis=0)
        rounds = np.arange(1, len(mean_traj) + 1)

        if "Ours" in method:
            color = primary_color
            lw = 3.2
            ls = "-"
            alpha_shade = 0.20
            zorder = 10
        elif "Random" in method:
            color = baseline_color
            lw = 1.5
            ls = ":"
            alpha_shade = 0.0
            zorder = 2
        else:
            color = secondary_colors[sec_idx % len(secondary_colors)]
            sec_idx += 1
            lw = 2.0
            ls = "-"
            alpha_shade = 0.08
            zorder = 5

        bar_colors.append(color)

        ax1.plot(rounds, mean_traj, color=color, linestyle=ls, label=method, linewidth=lw, zorder=zorder)
        if alpha_shade > 0:
            ax1.fill_between(rounds, mean_traj - std_traj, mean_traj + std_traj, color=color, alpha=alpha_shade, zorder=zorder-1)

        end_val = mean_traj[-1]
        txt = ax1.text(rounds[-1], end_val, f" {method} ({end_val:.1f}%)", color=color, fontweight='bold' if "Ours" in method else 'normal', fontsize=8.5)
        texts.append(txt)

    ax1.axhline(global_best, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Global Best ({global_best}%)")
    ax1.axhline(t95_target, color="#F59E0B", linestyle=":", linewidth=1.2, alpha=0.8, label=f"95% Target ({t95_target:.1f}%)")

    ax1.set_xlabel("Optimization Round", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Yield (%)", fontsize=11, fontweight='bold')
    ax1.set_title("Prototype 1: Visual Hierarchy\nA. Convergence Speed & Stability (Primary vs Baselines)", fontsize=11, fontweight='bold', pad=10)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right', fontsize=8.5)
    ax1.set_ylim(40, 102)

    adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    methods = list(t95_data.keys())
    x_pos = np.arange(len(methods))
    t95_means = [np.mean(t95_data[m]) for m in methods]
    t95_stds = [np.std(t95_data[m]) for m in methods]

    bars = ax2.bar(
        x_pos, t95_means, yerr=t95_stds, capsize=4,
        color=bar_colors, alpha=0.85, width=0.55, edgecolor='black', linewidth=1.0
    )

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
    ax2.set_title("Prototype 1: Visual Hierarchy\nB. t95 Convergence Speed (Lower is Better)", fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 48)

    plt.tight_layout()
    out_p = OUT_DIR / "prototype_1_visual_hierarchy.png"
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    return out_p


# ==============================================================================
# PROTOTYPE 2: Small Multiples / Facet Grid (2x2 Comparison Grid)
# ==============================================================================
def render_prototype_2(conv_data, t95_data, global_best):
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=300)
    axes_flat = axes.flatten()

    ours_matrix = conv_data["LGBO (Ours)"]
    ours_mean = np.mean(ours_matrix, axis=0)
    ours_std = np.std(ours_matrix, axis=0)
    rounds = np.arange(1, len(ours_mean) + 1)

    baselines = [m for m in conv_data.keys() if m != "LGBO (Ours)"]
    baseline_colors = ["#F59E0B", "#3B82F6", "#8B5CF6", "#9CA3AF"]

    for i, b_name in enumerate(baselines):
        ax = axes_flat[i]
        b_matrix = conv_data[b_name]
        b_mean = np.mean(b_matrix, axis=0)
        b_std = np.std(b_matrix, axis=0)
        b_color = baseline_colors[i]

        # Draw Ours (LGBO) as the benchmark in Emerald
        ax.plot(rounds, ours_mean, color="#10B981", label="LGBO (Ours)", linewidth=2.8, zorder=10)
        ax.fill_between(rounds, ours_mean - ours_std, ours_mean + ours_std, color="#10B981", alpha=0.15)

        # Draw Target Baseline
        ax.plot(rounds, b_mean, color=b_color, linestyle="--", label=b_name, linewidth=2.2, zorder=5)
        ax.fill_between(rounds, b_mean - b_std, b_mean + b_std, color=b_color, alpha=0.10)

        ax.axhline(global_best, color="#EF4444", linestyle=":", alpha=0.7, label="Global Best")

        # Calculate speedup
        ours_t95 = np.mean(t95_data["LGBO (Ours)"])
        b_t95 = np.mean(t95_data[b_name])
        speedup = ((b_t95 - ours_t95) / b_t95) * 100

        ax.set_title(f"LGBO (Ours) vs {b_name}\n({speedup:+.1f}% Speedup to t95)", fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel("Yield (%)", fontsize=9)
        ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=8)
        ax.set_ylim(40, 102)

    plt.suptitle("Prototype 2: Small Multiples Grid (Zero-Overlap Comparison)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_p = OUT_DIR / "prototype_2_small_multiples.png"
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    return out_p


# ==============================================================================
# PROTOTYPE 3: Relative Boost & Speedup Summary (Normalized Speedup + Yield Boxplot)
# ==============================================================================
def render_prototype_3(conv_data, t95_data, global_best):
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # Left Subplot: Relative Speedup % compared to GPBO Standard
    gpbo_t95_mean = np.mean(t95_data["GPBO (Standard)"])
    methods = list(t95_data.keys())
    speedups = [((gpbo_t95_mean - np.mean(t95_data[m])) / gpbo_t95_mean) * 100 for m in methods]

    y_pos = np.arange(len(methods))
    bar_colors = ["#10B981" if "Ours" in m else "#F59E0B" if "Standard" in m else "#3B82F6" if "No" in m else "#8B5CF6" if "TuRBO" in m else "#9CA3AF" for m in methods]

    bars = ax1.barh(y_pos, speedups, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.0, height=0.55)
    ax1.axvline(0, color="black", linestyle="-", linewidth=1.0)

    for bar, val in zip(bars, speedups):
        width = bar.get_width()
        x_text = width + (1.5 if width >= 0 else -6.0)
        ax1.text(x_text, bar.get_y() + bar.get_height()/2.0, f"{val:+.1f}%", va='center', fontweight='bold', fontsize=9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=10, fontweight='bold')
    ax1.set_xlabel("Relative t95 Speedup (%) vs Standard GPBO", fontsize=11, fontweight='bold')
    ax1.set_title("Prototype 3: Relative Boost Dashboard\nA. Speedup Advantage vs Standard GPBO Baseline", fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='x')

    # Right Subplot: Seaborn Boxplot for Final Yield Distribution
    import pandas as pd
    box_rows = []
    for method, matrix in conv_data.items():
        final_yields = matrix[:, -1]
        for y in final_yields:
            box_rows.append({"Method": method, "Final Yield (%)": y})
    df_box = pd.DataFrame(box_rows)

    sns.boxplot(
        data=df_box, y="Method", x="Final Yield (%)", hue="Method", legend=False, ax=ax2,
        palette={m: c for m, c in zip(methods, bar_colors)},
        width=0.45, linewidth=1.2, fliersize=3
    )

    ax2.set_ylabel("", fontsize=11)
    ax2.set_xlabel("Final Round Yield (%)", fontsize=11, fontweight='bold')
    ax2.set_title("Prototype 3: Relative Boost Dashboard\nB. Final Yield Reliability Distribution", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlim(80, 101)

    plt.tight_layout()
    out_p = OUT_DIR / "prototype_3_relative_boost.png"
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    return out_p


def main():
    conv_data, t95_data, global_best = generate_simulated_benchmark_data()

    p1 = render_prototype_1(conv_data, t95_data, global_best)
    p2 = render_prototype_2(conv_data, t95_data, global_best)
    p3 = render_prototype_3(conv_data, t95_data, global_best)

    print(f"[Prototypes] Generated:\n 1. {p1}\n 2. {p2}\n 3. {p3}")

    if ARTIFACT_DIR.exists():
        for p in [p1, p2, p3]:
            shutil.copy(p, ARTIFACT_DIR / p.name)
            print(f"[Prototypes] Copied to artifact dir: {ARTIFACT_DIR / p.name}")


if __name__ == "__main__":
    main()

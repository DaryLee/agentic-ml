"""
Post-hoc analysis of experiment results.
Reads results.tsv and generates progress plots.
Run after (or during) the agent loop: python analyze.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_PATH = BASE_DIR / "results.tsv"
PLOTS_DIR = BASE_DIR / "plots"


def load_results():
    if not RESULTS_PATH.exists():
        print("No results.tsv found. Run main.py first.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_PATH, sep="\t")
    print(f"Loaded {len(df)} experiments from results.tsv")
    return df


def plot_progress(df):
    """Main progress plot: F1 over experiments with keep/discard markers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Color by status
    colors = {"keep": "#2ecc71", "discard": "#95a5a6", "crash": "#e74c3c"}
    for status, group in df.groupby("status"):
        ax.scatter(group["step"], group["f1_macro"], c=colors.get(status, "#95a5a6"),
                   s=80, label=status.upper(), edgecolors="white", linewidths=0.5, zorder=3)

    # Best-so-far line
    best_so_far = df["f1_macro"].cummax()
    ax.plot(df["step"], best_so_far, "k--", linewidth=1.5, alpha=0.7, label="Best so far")

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("AutoResearch Progress: Macro F1 Over Experiments", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "progress.png", dpi=150)
    plt.close()
    print("  -> progress.png")


def plot_improvement_trajectory(df):
    """Only kept experiments, showing monotonic improvement."""
    kept = df[df["status"] == "keep"].copy()
    if len(kept) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(kept["step"], kept["f1_macro"], "o-", color="#2ecc71", markersize=8, linewidth=2)

    for _, row in kept.iterrows():
        desc = str(row.get("description", ""))[:20]
        ax.annotate(desc, (row["step"], row["f1_macro"]),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=7, alpha=0.8, rotation=15)

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Improvement Trajectory (Kept Experiments)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "improvement_trajectory.png", dpi=150)
    plt.close()
    print("  -> improvement_trajectory.png")


def plot_status_distribution(df):
    """Pie chart of keep/discard/crash rates."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Status distribution
    status_counts = df["status"].value_counts()
    colors = {"keep": "#2ecc71", "discard": "#95a5a6", "crash": "#e74c3c"}
    pie_colors = [colors.get(s, "#95a5a6") for s in status_counts.index]
    axes[0].pie(status_counts, labels=status_counts.index.str.upper(),
                colors=pie_colors, autopct="%1.0f%%", startangle=90)
    axes[0].set_title("Experiment Outcomes")

    # F1 distribution histogram
    valid = df[df["f1_macro"] > 0]["f1_macro"]
    if len(valid) > 0:
        axes[1].hist(valid, bins=min(15, len(valid)), color="#3498db", alpha=0.7, edgecolor="white")
        axes[1].axvline(valid.max(), color="#2ecc71", linestyle="--", label=f"Best: {valid.max():.4f}")
        axes[1].set_xlabel("Macro F1")
        axes[1].set_ylabel("Count")
        axes[1].set_title("F1 Score Distribution")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "analysis.png", dpi=150)
    plt.close()
    print("  -> analysis.png")


def print_summary(df):
    """Print a text summary of the run."""
    print("\n" + "=" * 50)
    print("  RUN SUMMARY")
    print("=" * 50)
    print(f"  Total experiments: {len(df)}")
    print(f"  Kept:    {len(df[df['status'] == 'keep'])}")
    print(f"  Discard: {len(df[df['status'] == 'discard'])}")
    print(f"  Crash:   {len(df[df['status'] == 'crash'])}")
    print(f"  Best F1: {df['f1_macro'].max():.4f}")
    print(f"  Baseline F1: {df.iloc[0]['f1_macro']:.4f}")
    improvement = df["f1_macro"].max() - df.iloc[0]["f1_macro"]
    print(f"  Improvement: +{improvement:.4f} ({improvement/max(df.iloc[0]['f1_macro'], 0.001)*100:.1f}%)")
    print("=" * 50)

    # Best experiment
    best_row = df.loc[df["f1_macro"].idxmax()]
    print(f"\n  Best experiment (step {int(best_row['step'])}):")
    print(f"    {best_row.get('description', 'N/A')}")
    print()


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_results()
    print_summary(df)

    print("\nGenerating plots...")
    plot_progress(df)
    plot_improvement_trajectory(df)
    plot_status_distribution(df)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()

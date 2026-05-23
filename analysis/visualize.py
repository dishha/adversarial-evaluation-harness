"""
Visualization module for adversarial evaluation experiment results.

Usage:
    python analysis/visualize.py --input results.json --output-dir analysis/charts/

Generates Graphs A–D and Table E from the Jira deliverable spec.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("experiments", [data])  # support both single and multi-run files


def build_summary_df(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        s = exp["summary"]
        rows.append({
            "model": s.get("model_label", "unknown"),
            "budget": s.get("budget_label", 0),
            "scenario_type": s.get("scenario_type", "toxicity"),
            "failure_rate": s.get("failure_rate", s.get("toxic_failure_rate", 0)),
            "tokens_per_failure": s.get("tokens_per_failure", s.get("tokens_per_toxic_failure")),
            "sessions_per_failure": s.get("sessions_per_failure", s.get("sessions_per_toxic_failure")),
            "avg_turns_to_failure": s.get("avg_turns_to_failure"),
            "total_sessions": s.get("total_sessions", 0),
            "failed_sessions": s.get("failed_sessions", s.get("toxic_failure_sessions", 0)),
            "tokens_used": s.get("tokens_used_total", 0),
        })
    return pd.DataFrame(rows)


def build_turn_depth_series(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        model = exp["summary"].get("model_label", "unknown")
        for sess in exp.get("sessions", []):
            if sess.get("failed", sess.get("toxic_failure")):
                rows.append({
                    "model": model,
                    "turns_to_failure": sess["turn_count"],
                })
    return pd.DataFrame(rows)


def build_turn_score_series(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        model = exp["summary"].get("model_label", "unknown")
        for sess in exp.get("sessions", []):
            failed = sess.get("failed", False)
            for turn in sess.get("turns", []):
                jr = turn.get("judge_result", {})
                rows.append({
                    "model": model,
                    "session_id": sess["session_id"],
                    "turn_id": turn["turn_id"],
                    "failure_score": jr.get("failure_score", 0),
                    "harm_potential": jr.get("harm_potential", 0),
                    "failed_session": failed,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Graph A — Toxic Failure Rate vs Token Budget
# ---------------------------------------------------------------------------

def graph_a(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for model, grp in df.groupby("model"):
        grp_sorted = grp.sort_values("budget")
        ax.plot(
            grp_sorted["budget"],
            grp_sorted["failure_rate"] * 100,
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.set_xlabel("Token Budget", fontsize=12)
    ax.set_ylabel("Failure Rate (%)", fontsize=12)
    ax.set_title("Graph A — Failure Rate vs Token Budget", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_ylim(0, 105)
    ax.legend(title="Target Model", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir / "graph_a_failure_rate_vs_budget.png")


# ---------------------------------------------------------------------------
# Graph B — Tokens Required per Toxic Failure
# ---------------------------------------------------------------------------

def graph_b(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.dropna(subset=["tokens_per_failure"]).copy()
    if plot_df.empty:
        print("  [Graph B] no toxic failures in data — skipping")
        return

    # Pivot for grouped bar chart: x=budget, hue=model
    pivot = plot_df.pivot_table(
        index="budget", columns="model", values="tokens_per_failure", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)

    for i, col in enumerate(pivot.columns):
        offset = (i - len(pivot.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, pivot[col] / 1000, width=width * 0.9, label=col)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}k", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b):,}" for b in pivot.index])
    ax.set_xlabel("Token Budget", fontsize=12)
    ax.set_ylabel("Tokens per Failure (×1,000)", fontsize=12)
    ax.set_title("Graph B — Tokens Required per Failure", fontsize=13, fontweight="bold")
    ax.legend(title="Target Model", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir / "graph_b_tokens_per_failure.png")


# ---------------------------------------------------------------------------
# Graph C — Sessions Required per Toxic Failure
# ---------------------------------------------------------------------------

def graph_c(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.dropna(subset=["sessions_per_failure"]).copy()
    if plot_df.empty:
        print("  [Graph C] no toxic failures in data — skipping")
        return

    models = sorted(plot_df["model"].unique())
    data_by_model = [plot_df.loc[plot_df["model"] == m, "sessions_per_failure"].values for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data_by_model, labels=models, patch_artist=True, notch=False)

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Target Model", fontsize=12)
    ax.set_ylabel("Sessions per Failure", fontsize=12)
    ax.set_title("Graph C — Sessions Required per Failure", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir / "graph_c_sessions_per_failure.png")


# ---------------------------------------------------------------------------
# Graph D — Turn Depth Before Failure
# ---------------------------------------------------------------------------

def graph_d(turns_df: pd.DataFrame, output_dir: Path) -> None:
    if turns_df.empty:
        print("  [Graph D] no toxic failure sessions — skipping")
        return

    models = sorted(turns_df["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models:
        vals = turns_df.loc[turns_df["model"] == model, "turns_to_failure"]
        ax.hist(vals, bins=range(1, 12), alpha=0.6, label=model, edgecolor="white")

    ax.set_xlabel("Turns Before Failure", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Graph D — Turn Depth Before Failure", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(title="Target Model", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir / "graph_d_turn_depth.png")


# ---------------------------------------------------------------------------
# Graph F — Failure Score Progression by Turn
# ---------------------------------------------------------------------------

def graph_f(score_df: pd.DataFrame, output_dir: Path) -> None:
    if score_df.empty:
        print("  [Graph F] no turn-level data — skipping")
        return

    models = sorted(score_df["model"].unique())
    max_turn = int(score_df["turn_id"].max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, col, ylabel, title_suffix in [
        (axes[0], "failure_score", "Avg Failure Score (0–4)", "Failure Score"),
        (axes[1], "harm_potential", "Avg Harm Potential (0–4)", "Harm Potential"),
    ]:
        colors = plt.cm.Set1(np.linspace(0, 0.8, len(models)))
        for model, color in zip(models, colors):
            sub = score_df[score_df["model"] == model]
            stats = (
                sub.groupby("turn_id")[col]
                .agg(["mean", "std"])
                .reindex(range(1, max_turn + 1))
            )
            stats["std"] = stats["std"].fillna(0)
            turns = stats.index.values
            means = stats["mean"].values
            stds = stats["std"].values

            ax.plot(turns, means, marker="o", linewidth=2, label=model, color=color)
            ax.fill_between(turns, means - stds, means + stds, alpha=0.15, color=color)

        # Trend line (pooled across all models)
        pooled = score_df.groupby("turn_id")[col].mean().reindex(range(1, max_turn + 1))
        z = np.polyfit(
            pooled.index[pooled.notna()],
            pooled[pooled.notna()].values,
            1,
        )
        trend = np.poly1d(z)
        ax.plot(
            range(1, max_turn + 1),
            trend(range(1, max_turn + 1)),
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.5,
            label=f"Trend (slope={z[0]:+.3f})",
        )

        ax.set_xlabel("Turn Number", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Graph F — {title_suffix} vs Turn", fontsize=12, fontweight="bold")
        ax.set_xlim(0.5, max_turn + 0.5)
        ax.set_ylim(-0.1, 4.2)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Graph F — Does Failure Score Escalate with Retries?",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, output_dir / "graph_f_score_progression.png")


# ---------------------------------------------------------------------------
# Table E — Comparative Model Resilience
# ---------------------------------------------------------------------------

def table_e(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        df.groupby("model")
        .agg(
            Budgets=("budget", lambda x: ", ".join(f"{int(v):,}" for v in sorted(x))),
            Failure_Rate=("failure_rate", lambda x: f"{x.mean():.1%}"),
            Tokens_per_Failure=("tokens_per_failure", lambda x: f"{x.dropna().mean():,.0f}" if x.dropna().any() else "—"),
            Avg_Turns=("avg_turns_to_failure", lambda x: f"{x.dropna().mean():.1f}" if x.dropna().any() else "—"),
            Sessions_per_Failure=("sessions_per_failure", lambda x: f"{x.dropna().mean():.1f}" if x.dropna().any() else "—"),
        )
        .reset_index()
        .rename(columns={
            "model": "Model",
            "Budgets": "Budgets Tested",
            "Failure_Rate": "Avg Failure Rate",
            "Tokens_per_Failure": "Avg Tokens/Failure",
            "Avg_Turns": "Avg Turns/Failure",
            "Sessions_per_Failure": "Avg Sessions/Failure",
        })
    )

    csv_path = output_dir / "table_e_model_resilience.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Table E saved to {csv_path}")
    print(summary.to_string(index=False))

    # Also render as a matplotlib table figure
    fig, ax = plt.subplots(figsize=(12, max(2, len(summary) * 0.6 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(summary.columns))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    ax.set_title("Table E — Comparative Model Resilience", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, output_dir / "table_e_model_resilience.png")
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def generate_all(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} …")
    experiments = load_results(input_path)
    df = build_summary_df(experiments)
    turns_df = build_turn_depth_series(experiments)
    score_df = build_turn_score_series(experiments)

    print(f"  {len(experiments)} experiment(s), {len(df)} summary rows, {len(score_df)} turn records")

    print("\nGraph A …")
    graph_a(df, out)

    print("Graph B …")
    graph_b(df, out)

    print("Graph C …")
    graph_c(df, out)

    print("Graph D …")
    graph_d(turns_df, out)

    print("Table E …")
    table_e(df, out)

    print("Graph F …")
    graph_f(score_df, out)

    print("\nDone. Charts saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analytics charts from experiment results")
    parser.add_argument("--input", default="results.json", help="Experiment results JSON")
    parser.add_argument("--output-dir", default="analysis/charts", help="Directory for output files")
    args = parser.parse_args()
    generate_all(args.input, args.output_dir)

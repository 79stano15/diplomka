from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MODELS = (
    "medium",
    "compact_quantum",
    "compact_frozen",
)

LABELS = {
    "medium": "Medium",
    "compact_quantum": "Compact quantum",
    "compact_frozen": "Compact frozen",
}

COLORS = {
    "medium": "#3366cc",
    "compact_quantum": "#dd7700",
    "compact_frozen": "#228844",
}

BUDGETS = (500, 1000, 2000, 5000, 10000)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "model",
        "seed",
        "train_episodes",
        "train_policy_calls",
        "train_action_evals",
        "train_seconds",
        "total_parameters",
        "trainable_parameters",
        "matches",
        "wins",
        "win_rate",
        "nonwinning_decisions",
        "bank_rate",
        "mean_p_bank",
        "winning_opportunities",
        "missed_wins",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    unknown = set(frame["model"]).difference(MODELS)

    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    keys = ["model", "seed", "train_episodes"]

    if frame.duplicated(keys).any():
        raise ValueError("Duplicate model/seed/budget rows.")

    numeric = sorted(required.difference({"model"}))

    for column in numeric:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    if frame[numeric].isna().any().any():
        raise ValueError("Missing numeric values.")

    if (frame["matches"] <= 0).any():
        raise ValueError("Evaluation matches must be positive.")

    if (
        (frame["wins"] < 0)
        | (frame["wins"] > frame["matches"])
    ).any():
        raise ValueError("Invalid evaluation win counts.")

    expected_rate = frame["wins"] / frame["matches"]

    if ((frame["win_rate"] - expected_rate).abs() > 1e-9).any():
        raise ValueError("win_rate disagrees with wins/matches.")

    seed_sets = []

    for model, model_rows in frame.groupby("model"):
        seed_sets.append(set(model_rows["seed"]))

        for column in (
            "total_parameters",
            "trainable_parameters",
        ):
            if model_rows[column].nunique() != 1:
                raise ValueError(
                    f"Inconsistent {column} for {model}."
                )

    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError("Models do not have the same training seeds.")

    for (model, seed), run in frame.groupby(["model", "seed"]):
        run = run.sort_values("train_episodes")

        if tuple(run["train_episodes"]) != BUDGETS:
            raise ValueError(
                f"Unexpected checkpoint budgets: {model}, seed {seed}."
            )

        for column in (
            "train_policy_calls",
            "train_action_evals",
            "train_seconds",
        ):
            if not run[column].is_monotonic_increasing:
                raise ValueError(
                    f"Nonmonotonic {column}: {model}, seed {seed}."
                )

    frame["win_rate_pct"] = 100 * frame["win_rate"]
    frame["bank_rate_pct"] = 100 * frame["bank_rate"]
    frame["mean_p_bank_pct"] = 100 * frame["mean_p_bank"]

    denominator = frame["winning_opportunities"].where(
        frame["winning_opportunities"] > 0
    )
    frame["missed_win_pct"] = (
        100 * frame["missed_wins"] / denominator
    )

    # Reconstruct the integer count from its reported rate.
    frame["bank_decisions_reconstructed"] = (
        frame["bank_rate"] * frame["nonwinning_decisions"]
    ).round().astype(int)

    return frame.sort_values(keys).reset_index(drop=True)


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "win_rate_pct",
        "train_policy_calls",
        "train_action_evals",
        "train_seconds",
        "bank_rate_pct",
        "missed_win_pct",
    ]

    grouped = frame.groupby(
        ["model", "train_episodes"],
        sort=False,
    )

    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    summary = summary.join(
        grouped.size().rename("training_seeds")
    )

    return summary.reset_index()


def plot_mean_curve(
    summary: pd.DataFrame,
    models: list[str],
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))

    for model in models:
        rows = summary.loc[
            summary["model"] == model
        ].sort_values("train_episodes")

        x = rows["train_episodes"].to_numpy()
        mean = rows["win_rate_pct_mean"].to_numpy()
        sd = rows["win_rate_pct_std"].fillna(0).to_numpy()

        axis.plot(
            x,
            mean,
            marker="o",
            color=COLORS[model],
            label=LABELS[model],
        )
        axis.fill_between(
            x,
            (mean - sd).clip(0, 100),
            (mean + sd).clip(0, 100),
            color=COLORS[model],
            alpha=0.15,
        )

    axis.set(
        xlabel="Training matches",
        ylabel="Greedy win rate against v2 (%)",
        title="Checkpoint means; shading = +/- 1 SD across training seeds",
        ylim=(0, 100),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_seed_curves(
    frame: pd.DataFrame,
    models: list[str],
    output: Path,
) -> None:
    resources = (
        ("train_episodes", 1000, "Training matches (thousands)"),
        (
            "train_policy_calls",
            1000,
            "Candidate decisions (thousands)",
        ),
        (
            "train_action_evals",
            1000,
            "State-action evaluations (thousands)",
        ),
        ("train_seconds", 60, "Elapsed training time (minutes)"),
    )

    fig, axes = plt.subplots(
        len(models),
        len(resources),
        figsize=(16, 3.1 * len(models)),
        squeeze=False,
        sharey=True,
    )

    for row_index, model in enumerate(models):
        model_rows = frame.loc[frame["model"] == model]

        for column_index, (resource, scale, label) in enumerate(
            resources
        ):
            axis = axes[row_index, column_index]

            for seed, run in model_rows.groupby("seed"):
                run = run.sort_values("train_episodes")

                axis.plot(
                    run[resource] / scale,
                    run["win_rate_pct"],
                    marker="o",
                    markersize=4,
                    label=f"Seed {seed}",
                )

            axis.set_xlabel(label)
            axis.set_ylim(0, 100)
            axis.grid(alpha=0.25)

            if column_index == 0:
                axis.set_ylabel(
                    f"{LABELS[model]}\nWin rate against v2 (%)"
                )
                axis.legend(fontsize=8)

    fig.suptitle(
        "Individual training runs; lines connect saved checkpoints only"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_runtime(
    frame: pd.DataFrame,
    models: list[str],
    output: Path,
) -> None:
    final = frame.loc[frame["train_episodes"] == max(BUDGETS)]

    runtime = (
        final.groupby("model")["train_seconds"]
        .agg(["mean", "std"])
        .reindex(models)
    )

    fig, axis = plt.subplots(figsize=(7, 5))

    axis.bar(
        [LABELS[model] for model in models],
        runtime["mean"],
        yerr=runtime["std"].fillna(0),
        color=[COLORS[model] for model in models],
        capsize=5,
    )
    axis.set(
        ylabel="Elapsed training time (seconds)",
        title="10,000 training matches: mean +/- SD across seeds",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze saved sample-efficiency results."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV file does not exist: {args.csv}")

    output = args.output_dir or args.csv.parent / "analysis"

    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            parser.error("Choose an empty output directory.")

    frame = load_data(args.csv)
    models = [
        model for model in MODELS if model in set(frame["model"])
    ]
    summary = make_summary(frame)

    output.mkdir(parents=True, exist_ok=True)

    save_table(summary, output / "summary_by_budget.csv")

    parameters = frame.groupby("model", as_index=False)[
        ["total_parameters", "trainable_parameters"]
    ].first()

    final = summary.loc[
        summary["train_episodes"] == max(BUDGETS)
    ].merge(parameters, on="model", validate="one_to_one")

    save_table(final, output / "final_summary.csv")

    diagnostic_columns = [
        "model",
        "seed",
        "train_episodes",
        "win_rate_pct",
        "nonwinning_decisions",
        "bank_decisions_reconstructed",
        "bank_rate_pct",
        "mean_p_bank_pct",
        "winning_opportunities",
        "missed_wins",
        "missed_win_pct",
    ]
    diagnostics = frame.loc[
        frame["seed"] == 43,
        diagnostic_columns,
    ]
    save_table(diagnostics, output / "seed_43_diagnostics.csv")

    plot_mean_curve(
        summary,
        models,
        output / "mean_learning_curve.png",
    )
    plot_seed_curves(
        frame,
        models,
        output / "individual_learning_curves.png",
    )
    plot_runtime(
        frame,
        models,
        output / "final_training_time.png",
    )

    print("\nWin rate against v2: mean +/- SD, percentage points")

    for budget in BUDGETS:
        print(f"\nTraining matches: {budget}")

        for model in models:
            row = summary.loc[
                (summary["model"] == model)
                & (summary["train_episodes"] == budget)
            ].iloc[0]

            print(
                f"  {LABELS[model]:<18} "
                f"{row['win_rate_pct_mean']:.2f} "
                f"+/- {row['win_rate_pct_std']:.2f}"
            )

    print(f"\nSaved analysis: {output.resolve()}")


if __name__ == "__main__":
    main()
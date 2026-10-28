from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

import compare_policy_sizes as experiment
import dice_rl_critic as critic_experiment
import dice_rl_v2 as game


def read_previous_results(
    path: Path,
    seeds: list[int],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Chýbajú staršie výsledky: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return [
        row
        for row in rows
        if row["architecture"] in {"large", "medium"}
        and int(row["seed"]) in seeds
    ]


def summarize(rows: list[dict[str, str | int | float]]) -> None:
    print("\nPorovnanie proti v2:")

    names = ["large", "medium", "bottleneck4", "bottleneck6"]

    for name in names:
        selected = [
            row
            for row in rows
            if row["architecture"] == name
        ]

        if not selected:
            continue

        values = [float(row["vs_v2"]) for row in selected]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0

        print(
            f"  {name:<12} "
            f"{int(selected[0]['parameters']):>5} parametrov | "
            f"{mean:.1%} ± {sd:.1%} medzi seedmi"
        )

    print("\nJednotlivé tréningy nového modelu:")

    for row in rows:
        if not str(row["architecture"]).startswith("bottleneck"):
            continue

        print(
            f"  {row['architecture']}, seed {row['seed']}: "
            f"v2 {float(row['vs_v2']):.1%}; "
            f"250 {float(row['vs_250']):.1%}; "
            f"500 {float(row['vs_500']):.1%}; "
            f"650 {float(row['vs_650']):.1%}; "
            f"800 {float(row['vs_800']):.1%}"
        )


def plot_results(
    rows: list[dict[str, str | int | float]],
    output_dir: Path,
) -> None:
    colors = {
        "large": "#31688e",
        "medium": "#35b779",
        "bottleneck4": "#e78a31",
        "bottleneck6": "#756bb1",
    }

    available = [
        name
        for name in colors
        if any(row["architecture"] == name for row in rows)
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    figure.suptitle(
        "Klasická politika a kvantovo pripravené hrdlo",
        fontsize=14,
        weight="bold",
    )

    for position, name in enumerate(available):
        selected = [
            row
            for row in rows
            if row["architecture"] == name
        ]
        values = [100 * float(row["vs_v2"]) for row in selected]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0

        axes[0].errorbar(
            position,
            mean,
            yerr=sd,
            fmt="o",
            capsize=5,
            markersize=9,
            color=colors[name],
        )

        for index, value in enumerate(values):
            offset = (index - (len(values) - 1) / 2) * 0.08
            axes[0].scatter(
                position + offset,
                value,
                facecolors="white",
                edgecolors=colors[name],
                s=38,
                zorder=4,
            )

        thresholds = (250, 500, 650, 800)
        means = [
            100
            * statistics.mean(
                float(row[f"vs_{threshold}"])
                for row in selected
            )
            for threshold in thresholds
        ]

        axes[1].plot(
            thresholds,
            means,
            marker="o",
            linewidth=2,
            color=colors[name],
            label=name,
        )

    axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(len(available)), available)
    axes[0].set_ylabel("Výhry proti v2 (%)")
    axes[0].set_title("Bod: priemer ± SD; krúžky: tréningové seedy")

    axes[1].set_xticks((250, 500, 650, 800))
    axes[1].set_xlabel("Bankovací prah testovacieho súpera")
    axes[1].set_ylabel("Podiel výhier (%)")
    axes[1].set_title("Výkon proti heuristikám")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(
        output_dir / "bottleneck_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "bottleneck_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=(4, 6), default=4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-matches", type=int, default=400)
    parser.add_argument(
        "--previous-results",
        type=Path,
        default=Path("results/equal_depth_10k/results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bottleneck4_10k"),
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes musí byť kladné")

    if args.eval_matches < 4 or args.eval_matches % 4:
        parser.error("--eval-matches musí byť násobok štyroch")

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds obsahuje duplicity")

    name = f"bottleneck{args.width}"
    experiment.ARCHITECTURES[name] = (
        32,
        args.width,
        args.width,
    )

    planned = [
        args.output_dir / f"{name}_seed_{seed}.pt"
        for seed in args.seeds
    ]

    for path in planned:
        if path.exists():
            parser.error(
                f"Checkpoint už existuje: {path}. "
                "Použi nový --output-dir."
            )

    game.test_rules()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = critic_experiment.load_policy(
        Path("dice_policy_v2.pt")
    )
    reference.eval()
    reference.requires_grad_(False)

    rows: list[dict[str, str | int | float]] = []

    # Staršie čísla pridávame iba pri rovnakom tréningovom
    # a evaluačnom rozpočte.
    if args.episodes == 10_000 and args.eval_matches == 400:
        rows.extend(
            read_previous_results(
                args.previous_results,
                args.seeds,
            )
        )

    for seed in args.seeds:
        policy = experiment.train_one(
            name=name,
            seed=seed,
            episodes=args.episodes,
            reference=reference,
        )

        checkpoint_path = (
            args.output_dir / f"{name}_seed_{seed}.pt"
        )
        torch.save(
            {
                "architecture": name,
                "seed": seed,
                "state_dict": policy.state_dict(),
            },
            checkpoint_path,
        )

        row = experiment.evaluate_one(
            name=name,
            seed=seed,
            policy=policy,
            reference=reference,
            matches=args.eval_matches,
        )
        rows.append(row)

        print(
            f"  {name}, seed {seed}: "
            f"výhry proti v2 {float(row['vs_v2']):.1%}",
            flush=True,
        )

    csv_path = args.output_dir / "results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    plot_results(rows, args.output_dir)

    print(f"\nCSV: {csv_path}")
    print(
        "Grafy: "
        f"{args.output_dir / 'bottleneck_comparison.png'} "
        f"a {args.output_dir / 'bottleneck_comparison.pdf'}"
    )


if __name__ == "__main__":
    main()
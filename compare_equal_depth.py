from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

import compare_policy_sizes as previous
import dice_rl_critic as critic_experiment
import dice_rl_v2 as game


NEW_MODELS = {
    "medium": (32, 32),
    "compact": (16, 16),
}

COLORS = {
    "large": "#31688e",
    "medium": "#35b779",
    "compact": "#e78a31",
}

METRICS = ("vs_v2", "vs_250", "vs_500", "vs_650", "vs_800")


def load_existing_large(
    folder: Path,
    seed: int,
) -> previous.ActionPolicy | None:
    path = folder / f"large_seed_{seed}.pt"

    if not path.is_file():
        return None

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["architecture"] != "large":
        raise ValueError(f"Nesprávna architektúra v {path}")

    policy = previous.ActionPolicy((64, 64))
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    policy.requires_grad_(False)
    return policy


def save_results(
    rows: list[dict[str, str | int | float]],
    path: Path,
) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def format_metric(
    rows: list[dict[str, str | int | float]],
    name: str,
    metric: str,
) -> str:
    values = [
        float(row[metric])
        for row in rows
        if row["architecture"] == name
    ]

    if not values:
        return "—"

    average = statistics.mean(values)

    if len(values) == 1:
        return f"{average:.1%}"

    deviation = statistics.stdev(values)
    return f"{average:.1%} ± {deviation:.1%}"


def print_summary(
    rows: list[dict[str, str | int | float]],
) -> None:
    print("\nVýsledky jednotlivých tréningov:")

    for row in rows:
        print(
            f"  {row['architecture']:>7}, seed {row['seed']}: "
            f"v2 {float(row['vs_v2']):.1%}, "
            f"250 {float(row['vs_250']):.1%}, "
            f"500 {float(row['vs_500']):.1%}, "
            f"650 {float(row['vs_650']):.1%}, "
            f"800 {float(row['vs_800']):.1%}"
        )

    print("\nPriemer ± SD medzi tréningovými seedmi:")

    for name in COLORS:
        selected = [
            row
            for row in rows
            if row["architecture"] == name
        ]

        if not selected:
            continue

        print(
            f"  {name:>7} "
            f"({selected[0]['parameters']} parametrov): "
            f"v2 {format_metric(rows, name, 'vs_v2')}; "
            f"250 {format_metric(rows, name, 'vs_250')}; "
            f"500 {format_metric(rows, name, 'vs_500')}; "
            f"650 {format_metric(rows, name, 'vs_650')}; "
            f"800 {format_metric(rows, name, 'vs_800')}"
        )


def save_plot(
    figure: plt.Figure,
    folder: Path,
    name: str,
) -> None:
    figure.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_results(
    rows: list[dict[str, str | int | float]],
    folder: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = {"large": 0, "medium": 1, "compact": 2}

    for name, position in positions.items():
        selected = [
            row
            for row in rows
            if row["architecture"] == name
        ]

        if not selected:
            continue

        values = [100 * float(row["vs_v2"]) for row in selected]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0

        axes[0].errorbar(
            position,
            mean,
            yerr=sd,
            fmt="o",
            markersize=9,
            capsize=5,
            linewidth=2,
            color=COLORS[name],
        )

        for index, value in enumerate(values):
            offset = (index - (len(values) - 1) / 2) * 0.08
            axes[0].scatter(
                position + offset,
                value,
                s=35,
                facecolors="white",
                edgecolors=COLORS[name],
                linewidths=1.5,
                zorder=4,
            )

        thresholds = (250, 500, 650, 800)
        averages = []
        deviations = []

        for threshold in thresholds:
            scores = [
                100 * float(row[f"vs_{threshold}"])
                for row in selected
            ]
            averages.append(statistics.mean(scores))
            deviations.append(
                statistics.stdev(scores)
                if len(scores) > 1
                else 0.0
            )

        axes[1].errorbar(
            thresholds,
            averages,
            yerr=deviations,
            fmt="-o",
            capsize=3,
            linewidth=2,
            color=COLORS[name],
            label=name,
        )

    axes[0].axhline(
        50,
        color="#777777",
        linestyle="--",
        linewidth=1,
    )
    axes[0].set_xticks(
        list(positions.values()),
        list(positions.keys()),
    )
    axes[0].set_ylabel("Podiel výhier proti v2 (%)")
    axes[0].set_title("Priame zápasy")
    axes[0].text(
        0.02,
        0.03,
        "Plný bod: priemer ± SD\n"
        "Prázdny bod: tréningový seed",
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
    )

    axes[1].set_xticks((250, 500, 650, 800))
    axes[1].set_xlabel("Bankovací prah testovacieho súpera")
    axes[1].set_ylabel("Podiel výhier (%)")
    axes[1].set_title("Výkon proti pevným súperom")
    axes[1].legend(frameon=False)

    figure.suptitle(
        "Vplyv šírky politiky pri rovnakom počte vrstiev",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout()

    save_plot(figure, folder, "equal_depth_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(NEW_MODELS),
        default=list(NEW_MODELS),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-matches", type=int, default=400)

    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("dice_policy_v2.pt"),
    )
    parser.add_argument(
        "--large-dir",
        type=Path,
        default=Path("results/policy_sizes_10k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/equal_depth_10k"),
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes musí byť aspoň 1")

    if args.eval_matches < 4 or args.eval_matches % 4:
        parser.error("--eval-matches musí byť násobok štyroch")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds obsahuje duplicity")

    if len(set(args.models)) != len(args.models):
        parser.error("--models obsahuje duplicity")

    if not args.reference.is_file():
        parser.error(f"Chýba {args.reference}")

    planned = [
        args.output_dir / f"{name}_seed_{seed}.pt"
        for seed in args.seeds
        for name in args.models
    ]

    existing = [path for path in planned if path.exists()]

    if existing:
        parser.error(
            f"Checkpoint už existuje: {existing[0]}. "
            "Zvoľ nový --output-dir."
        )

    game.test_rules()
    torch.set_num_threads(1)

    # Tréningové a evaluačné funkcie používajú tento slovník.
    previous.ARCHITECTURES.update(NEW_MODELS)

    reference = critic_experiment.load_policy(args.reference)
    reference.eval()
    reference.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []

    for seed in args.seeds:
        # Pôvodný large znovu netrénujeme. Načítame ho len vtedy,
        # keď má experiment rovnaký rozpočet ako predchádzajúci.
        if args.episodes == 10_000 and args.eval_matches == 400:
            large = load_existing_large(args.large_dir, seed)

            if large is not None:
                rows.append(
                    previous.evaluate_one(
                        "large",
                        seed,
                        large,
                        reference,
                        args.eval_matches,
                    )
                )
                print(
                    f"\nlarge, seed {seed}: "
                    "načítaný pôvodný checkpoint",
                    flush=True,
                )

        for name in args.models:
            policy = previous.train_one(
                name=name,
                seed=seed,
                episodes=args.episodes,
                reference=reference,
            )

            path = (
                args.output_dir
                / f"{name}_seed_{seed}.pt"
            )

            torch.save(
                {
                    "architecture": name,
                    "seed": seed,
                    "state_dict": policy.state_dict(),
                },
                path,
            )

            row = previous.evaluate_one(
                name=name,
                seed=seed,
                policy=policy,
                reference=reference,
                matches=args.eval_matches,
            )
            rows.append(row)

            print(
                f"  uložené: {path}; "
                f"proti v2: {float(row['vs_v2']):.1%}",
                flush=True,
            )

    csv_path = args.output_dir / "results.csv"
    save_results(rows, csv_path)
    print_summary(rows)
    plot_results(rows, args.output_dir)

    print(f"\nCSV: {csv_path}")
    print(
        f"Grafy: {args.output_dir / 'equal_depth_comparison.png'} "
        "a rovnomenné PDF"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

import compare_policy_sizes as previous
import dice_rl_critic as critic_experiment
import dice_rl_v2 as game


MODELS: dict[str, tuple[tuple[int, ...], Path]] = {
    "bottleneck4": (
        (32, 4, 4),
        Path("results/bottleneck4_10k"),
    ),
    "medium": (
        (32, 32),
        Path("results/equal_depth_10k"),
    ),
    "large": (
        (64, 64),
        Path("results/policy_sizes_10k"),
    ),
}

TAPERED = (32, 16, 8)
METRICS = ("vs_v2", "vs_250", "vs_500", "vs_650", "vs_800")


def load_existing(
    name: str,
    seed: int,
) -> previous.ActionPolicy:
    hidden_sizes, folder = MODELS[name]
    path = folder / f"{name}_seed_{seed}.pt"

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["architecture"] != name:
        raise ValueError(f"Nesprávna architektúra v {path}")

    policy = previous.ActionPolicy(hidden_sizes)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    policy.requires_grad_(False)
    return policy


def print_results(
    rows: list[dict[str, str | int | float]],
    seeds: list[int],
    matches: int,
) -> None:
    names = ("bottleneck4", "tapered", "medium", "large")

    print(
        "\nVYHODNOTENIE BEZ STRATEGICKÉHO FILTRA"
        f"\n{matches} zápasov na model, seed a súpera;"
        " greedy výber"
    )
    print("v2 = zmrazená politika; čísla = bankovacie prahy")

    header = (
        f"{'Model':<13} {'Seed':>5} {'Param.':>7} "
        f"{'v2':>8} {'250':>8} {'500':>8} "
        f"{'650':>8} {'800':>8}"
    )
    print("\nJednotlivé tréningy")
    print(header)
    print("-" * len(header))

    for name in names:
        for seed in seeds:
            row = next(
                item
                for item in rows
                if item["architecture"] == name
                and item["seed"] == seed
            )
            scores = " ".join(
                f"{float(row[metric]):>8.1%}"
                for metric in METRICS
            )
            print(
                f"{name:<13} {seed:>5} "
                f"{int(row['parameters']):>7} {scores}"
            )

    print("\nPriemer ± SD medzi tréningovými seedmi")
    print(f"{'Model':<13} {'Param.':>7} " + "  ".join(METRICS))
    print("-" * 84)

    for name in names:
        selected = [
            row
            for row in rows
            if row["architecture"] == name
        ]
        cells = []

        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            mean = statistics.mean(values)
            sd = (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            )
            cells.append(f"{mean:>5.1%} ± {sd:>4.1%}")

        parameters = int(selected[0]["parameters"])
        print(
            f"{name:<13} {parameters:>7} "
            + "  ".join(cells)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument(
        "--eval-matches",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("dice_policy_v2.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tapered_10k"),
    )
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Zadaj neprázdny zoznam rôznych seedov")

    if args.eval_matches < 4 or args.eval_matches % 4:
        parser.error("--eval-matches musí byť násobok štyroch")

    if not args.reference.is_file():
        parser.error(f"Chýba {args.reference}")

    for seed in args.seeds:
        for name, (_, folder) in MODELS.items():
            path = folder / f"{name}_seed_{seed}.pt"
            if not path.is_file():
                parser.error(f"Chýba porovnávací checkpoint: {path}")

        output = args.output_dir / f"tapered_seed_{seed}.pt"
        if output.exists():
            parser.error(
                f"Checkpoint už existuje: {output}. "
                "Zvoľ nový --output-dir."
            )

    csv_path = args.output_dir / "results.csv"
    if csv_path.exists():
        parser.error(
            f"CSV už existuje: {csv_path}. "
            "Zvoľ nový --output-dir."
        )

    game.test_rules()
    torch.set_num_threads(1)
    previous.ARCHITECTURES["tapered"] = TAPERED

    reference = critic_experiment.load_policy(args.reference)
    reference.eval()
    reference.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []

    for seed in args.seeds:
        tapered = previous.train_one(
            name="tapered",
            seed=seed,
            episodes=10_000,
            reference=reference,
        )

        output = args.output_dir / f"tapered_seed_{seed}.pt"
        torch.save(
            {
                "architecture": "tapered",
                "seed": seed,
                "state_dict": tapered.state_dict(),
            },
            output,
        )
        print(f"  uložené: {output}", flush=True)

        for name in (
            "bottleneck4",
            "tapered",
            "medium",
            "large",
        ):
            policy = (
                tapered
                if name == "tapered"
                else load_existing(name, seed)
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
                f"  {name:<13} seed {seed}: "
                f"v2 {float(row['vs_v2']):.1%}",
                flush=True,
            )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print_results(rows, args.seeds, args.eval_matches)
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
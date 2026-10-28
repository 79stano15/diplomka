from __future__ import annotations

import argparse
import csv
import random
import statistics
from contextlib import redirect_stdout
from pathlib import Path

import torch

import dice_rl_critic as critic_experiment
import dice_rl_scalar_control as scalar_experiment
import dice_rl_v2 as game


THRESHOLDS = (250, 500, 800)


def load_policy(path: Path) -> game.PolicyNetwork:
    policy = critic_experiment.load_policy(path)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def evaluate_against_heuristic(
    policy: game.PolicyNetwork,
    threshold: int,
    matches: int,
    seed: int,
) -> float:
    wins = 0

    with torch.no_grad():
        for index in range(matches):
            winner, _, _, _ = game.play_match(
                policy=policy,
                rng=random.Random(seed + index),
                opponent_threshold=threshold,
                training=False,
                first_player=index % 2,
            )
            wins += int(winner == 0)

    return wins / matches


def evaluate_head_to_head(
    first: game.PolicyNetwork,
    second: game.PolicyNetwork,
    matches_per_seat: int,
    seed: int,
) -> float:
    """Podiel výhier `first`, aj keď hrá ako hráč 0 aj ako hráč 1."""

    first_wins = 0

    with torch.no_grad():
        for first_seat in (0, 1):
            player_zero = first if first_seat == 0 else second
            player_one = second if first_seat == 0 else first

            for index in range(matches_per_seat):
                episode = critic_experiment.play_match(
                    policy=player_zero,
                    opponent=player_one,
                    rng=random.Random(seed + index),
                    training=False,
                    first_player=index % 2,
                )
                first_wins += int(episode.winner == first_seat)

    return first_wins / (2 * matches_per_seat)


def train_pair(
    seed: int,
    episodes: int,
    initial_path: Path,
    opponent_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    critic_path = output_dir / f"critic_seed_{seed}.pt"
    value_path = output_dir / f"critic_value_seed_{seed}.pt"
    scalar_path = output_dir / f"scalar_seed_{seed}.pt"

    existing = [
        path
        for path in (critic_path, value_path, scalar_path)
        if path.exists()
    ]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Niektoré výsledky už existujú: {names}. "
            "Použi nový --output-dir; checkpointy neprepisujeme."
        )

    critic_log = output_dir / f"critic_seed_{seed}.log"
    print(f"\nSeed {seed}: trénujem model s kritikom...", flush=True)

    with critic_log.open("w", encoding="utf-8") as log:
        with redirect_stdout(log):
            critic_policy, value_network = critic_experiment.train(
                initial_path=initial_path,
                opponent_path=opponent_path,
                episodes=episodes,
                seed=seed,
                entropy_coefficient=0.01,
                actor_lr=0.0003,
                critic_lr=0.0005,
            )

    torch.save(critic_policy.state_dict(), critic_path)
    torch.save(value_network.state_dict(), value_path)
    del critic_policy, value_network

    scalar_log = output_dir / f"scalar_seed_{seed}.log"
    print(f"Seed {seed}: trénujem skalárnu baseline...", flush=True)

    with scalar_log.open("w", encoding="utf-8") as log:
        with redirect_stdout(log):
            scalar_policy = scalar_experiment.train(
                initial_path=initial_path,
                opponent_path=opponent_path,
                episodes=episodes,
                seed=seed,
                entropy_coefficient=0.01,
                actor_lr=0.0003,
                baseline_rate=0.02,
            )

    torch.save(scalar_policy.state_dict(), scalar_path)
    del scalar_policy

    print(f"  Checkpointy: {critic_path}, {scalar_path}", flush=True)
    print(f"  Priebehy: {critic_log}, {scalar_log}", flush=True)

    return critic_path, scalar_path


def evaluate_pair(
    seed: int,
    critic_path: Path,
    scalar_path: Path,
    reference: game.PolicyNetwork,
    matches: int,
) -> dict[str, float | int]:
    critic = load_policy(critic_path)
    scalar = load_policy(scalar_path)

    # Rovnaké evaluačné seedy pre všetky tréningové behy.
    # Sú nezávislé od tréningového seedu.
    row: dict[str, float | int] = {"seed": seed}

    row["critic_vs_v2"] = evaluate_head_to_head(
        critic,
        reference,
        matches_per_seat=matches,
        seed=100_000,
    )
    row["scalar_vs_v2"] = evaluate_head_to_head(
        scalar,
        reference,
        matches_per_seat=matches,
        seed=200_000,
    )
    row["critic_vs_scalar"] = evaluate_head_to_head(
        critic,
        scalar,
        matches_per_seat=matches,
        seed=300_000,
    )

    for threshold in THRESHOLDS:
        evaluation_seed = 400_000 + threshold * 1_000

        row[f"critic_vs_{threshold}"] = evaluate_against_heuristic(
            critic,
            threshold=threshold,
            matches=matches,
            seed=evaluation_seed,
        )
        row[f"scalar_vs_{threshold}"] = evaluate_against_heuristic(
            scalar,
            threshold=threshold,
            matches=matches,
            seed=evaluation_seed,
        )

    return row


def print_summary(rows: list[dict[str, float | int]]) -> None:
    print("\nVýsledky jednotlivých tréningov:")
    print(
        "seed | kritik–v2 | skalár–v2 | kritik–skalár | "
        "kritik 250/500/800 | skalár 250/500/800"
    )

    for row in rows:
        print(
            f"{row['seed']:>4} | "
            f"{row['critic_vs_v2']:>9.1%} | "
            f"{row['scalar_vs_v2']:>9.1%} | "
            f"{row['critic_vs_scalar']:>13.1%} | "
            f"{row['critic_vs_250']:.1%}/"
            f"{row['critic_vs_500']:.1%}/"
            f"{row['critic_vs_800']:.1%} | "
            f"{row['scalar_vs_250']:.1%}/"
            f"{row['scalar_vs_500']:.1%}/"
            f"{row['scalar_vs_800']:.1%}"
        )

    print("\nRozdiely kritik mínus skalárna baseline:")
    comparisons = [
        ("proti v2", "critic_vs_v2", "scalar_vs_v2"),
        ("proti prahu 250", "critic_vs_250", "scalar_vs_250"),
        ("proti prahu 500", "critic_vs_500", "scalar_vs_500"),
        ("proti prahu 800", "critic_vs_800", "scalar_vs_800"),
    ]

    for label, critic_key, scalar_key in comparisons:
        differences = [
            float(row[critic_key]) - float(row[scalar_key])
            for row in rows
        ]
        average = statistics.mean(differences) * 100
        deviation = (
            statistics.stdev(differences) * 100
            if len(differences) > 1
            else float("nan")
        )

        print(
            f"  {label:<18}: priemer {average:+.2f} "
            f"percentuálneho bodu, SD medzi seedmi {deviation:.2f}"
        )

    direct = [
        float(row["critic_vs_scalar"])
        for row in rows
    ]
    print(
        "\nPriamy súboj kritik–skalár: "
        f"priemer výhier kritika {statistics.mean(direct):.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--output-dir",
        type=Path,
        default=Path("results/critic_vs_scalar"),
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes musí byť aspoň 1")

    if args.eval_matches < 2 or args.eval_matches % 2:
        parser.error("--eval-matches musí byť kladné párne číslo")

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds obsahuje duplicitné seedy")

    if not args.reference.is_file():
        parser.error(f"Referenčný checkpoint neexistuje: {args.reference}")

    game.test_rules()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_policy(args.reference)
    rows: list[dict[str, float | int]] = []

    for seed in args.seeds:
        critic_path, scalar_path = train_pair(
            seed=seed,
            episodes=args.episodes,
            initial_path=args.reference,
            opponent_path=args.reference,
            output_dir=args.output_dir,
        )

        row = evaluate_pair(
            seed=seed,
            critic_path=critic_path,
            scalar_path=scalar_path,
            reference=reference,
            matches=args.eval_matches,
        )
        rows.append(row)

        print(
            f"  Vyhodnotené: kritik vs. skalár "
            f"{row['critic_vs_scalar']:.1%}",
            flush=True,
        )

    csv_path = args.output_dir / "results.csv"
    fieldnames = list(rows[0])

    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"\nTabuľka uložená do {csv_path}")


if __name__ == "__main__":
    main()
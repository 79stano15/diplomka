from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

import compare_sample_efficiency as runner


experiment = runner.experiment

TARGETS = (
    ("compact_quantum", 1000),
    ("compact_quantum", 2000),
    ("compact_frozen", 1000),
)

TRAINING_FIELDS = (
    "train_policy_calls",
    "train_action_evals",
    "train_rolls",
    "train_seconds",
    "total_parameters",
    "trainable_parameters",
)

REPLAY_FIELDS = (
    "matches",
    "wins",
    "rolls",
    "policy_calls",
    "action_evals",
    "nonwinning_decisions",
    "bank_rate",
    "dominated_banks",
    "winning_opportunities",
    "missed_wins",
)


def numbers_match(left: object, right: object) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-10,
        abs_tol=1e-8,
    )


def load_original_row(
    rows: list[dict[str, str]],
    model: str,
    seed: int,
    budget: int,
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and int(row["seed"]) == seed
        and int(row["train_episodes"]) == budget
    ]

    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one CSV row for {model}, "
            f"seed {seed}, budget {budget}; got {len(selected)}."
        )

    return selected[0]


def validate_checkpoint(
    checkpoint: dict,
    original: dict[str, str],
    model: str,
    seed: int,
    budget: int,
) -> torch.nn.Module:
    expected_metadata = {
        "model": model,
        "seed": seed,
        "episodes": budget,
    }

    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint metadata mismatch: {key}; "
                f"expected {expected}, got {checkpoint.get(key)}."
            )

    for key in TRAINING_FIELDS:
        if not numbers_match(checkpoint[key], original[key]):
            raise RuntimeError(
                f"Checkpoint/CSV mismatch: {key}; "
                f"{checkpoint[key]} versus {original[key]}."
            )

    policy = runner.make_policy(model, seed)

    initial_quantum_weights = (
        policy.block.weights.detach().clone()
        if model == "compact_frozen"
        else None
    )

    policy.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )
    policy.eval()

    actual_total = runner.total_parameters(policy)
    actual_trainable = experiment.parameter_count(policy)

    if actual_total != int(checkpoint["total_parameters"]):
        raise RuntimeError("Total parameter count mismatch.")

    if actual_trainable != int(checkpoint["trainable_parameters"]):
        raise RuntimeError("Trainable parameter count mismatch.")

    for key, value in policy.state_dict().items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite model values: {key}")

    if initial_quantum_weights is not None:
        if not torch.equal(
            initial_quantum_weights,
            policy.block.weights.detach(),
        ):
            raise RuntimeError(
                "Frozen weights differ from reconstructed initialization. "
                "Check initialization code and checkpoint provenance."
            )

    return policy


@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    matches: int,
    sampled: bool,
) -> experiment.DecisionStats:
    stats = experiment.DecisionStats()

    for match_index in range(matches):
        match_seed = 100_000_000 + match_index // 2

        experiment.play_match(
            policy=policy,
            reference=reference,
            opponent_kind="v2",
            dice_rng=experiment.random.Random(match_seed),
            heuristic_rng=experiment.random.Random(
                match_seed + 10_000_000
            ),
            action_generator=torch.Generator().manual_seed(
                match_seed + 20_000_000
            ),
            first_player=match_index % 2,
            training=False,
            sampled=sampled,
            stats=stats,
            max_rolls=300,
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect saved checkpoints without training."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )
    parser.add_argument(
        "--eval-matches",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("dice_policy_v2.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    args = parser.parse_args()

    if args.eval_matches < 4 or args.eval_matches % 2:
        parser.error("--eval-matches must be even and >= 4.")

    if args.seed < 0:
        parser.error("--seed must be nonnegative.")

    csv_path = args.results_dir / "efficiency.csv"

    if not csv_path.is_file():
        parser.error(f"Missing CSV: {csv_path}")

    if not args.reference.is_file():
        parser.error(f"Missing reference: {args.reference}")

    checkpoint_paths = {
        (model, budget): (
            args.results_dir
            / f"{model}_seed_{args.seed}"
            / f"episode_{budget}.pt"
        )
        for model, budget in TARGETS
    }

    missing = [
        str(path)
        for path in checkpoint_paths.values()
        if not path.is_file()
    ]

    if missing:
        parser.error(
            "Missing checkpoints:\n" + "\n".join(missing)
        )

    output = args.output_dir or (
        args.results_dir / "checkpoint_diagnostics"
    )

    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            parser.error("Choose an empty output directory.")

    config_path = args.results_dir / "config.json"

    if config_path.is_file():
        config = json.loads(
            config_path.read_text(encoding="utf-8-sig")
        )
        expected_hash = config.get("reference_sha256")

        if expected_hash is not None:
            actual_hash = experiment.file_hash(args.reference)

            if actual_hash != expected_hash:
                parser.error(
                    "Reference checkpoint hash differs from config.json."
                )

    with csv_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as source:
        original_rows = list(csv.DictReader(source))

    torch.set_num_threads(1)
    experiment.game.test_rules()

    reference = experiment.critic_experiment.load_policy(
        args.reference
    )
    reference.eval()
    reference.requires_grad_(False)

    output.mkdir(parents=True, exist_ok=True)

    for model, budget in TARGETS:
        original = load_original_row(
            original_rows,
            model,
            args.seed,
            budget,
        )
        checkpoint = torch.load(
            checkpoint_paths[(model, budget)],
            map_location="cpu",
            weights_only=True,
        )
        policy = validate_checkpoint(
            checkpoint,
            original,
            model,
            args.seed,
            budget,
        )

        print(
            f"\n{model}, seed {args.seed}, episode {budget}: "
            "metadata OK",
            flush=True,
        )

        for mode in ("greedy", "sampled"):
            started = time.perf_counter()
            stats = evaluate(
                policy,
                reference,
                args.eval_matches,
                sampled=mode == "sampled",
            )
            elapsed = time.perf_counter() - started
            metrics = stats.as_dict()

            replay_status = "not_applicable"

            if mode == "greedy":
                if args.eval_matches == int(original["matches"]):
                    differences = [
                        key
                        for key in REPLAY_FIELDS
                        if not numbers_match(
                            metrics[key],
                            original[key],
                        )
                    ]

                    replay_status = (
                        "match"
                        if not differences
                        else "mismatch:" + ",".join(differences)
                    )
                else:
                    replay_status = "different_eval_match_count"

            row = {
                "model": model,
                "seed": args.seed,
                "train_episodes": budget,
                "mode": mode,
                "metadata_check": "ok",
                "original_greedy_replay": replay_status,
                "eval_seconds": elapsed,
                **metrics,
            }
            experiment.append_csv(
                output / "diagnostic_evaluation.csv",
                row,
            )

            print(
                f"  {mode:<7} "
                f"wins={stats.wins}/{stats.matches} "
                f"({metrics['win_rate']:.1%}), "
                f"bank={metrics['bank_rate']:.2%}, "
                f"P(BANK)={metrics['mean_p_bank']:.2%}, "
                f"missed={stats.missed_wins}/"
                f"{stats.winning_opportunities}, "
                f"time={elapsed:.1f}s",
                flush=True,
            )

            if mode == "greedy":
                print(
                    f"  Original greedy replay: {replay_status}",
                    flush=True,
                )

    print(f"\nSaved diagnostics: {output.resolve()}")


if __name__ == "__main__":
    main()
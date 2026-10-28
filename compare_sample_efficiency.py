from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

import compare_compact_quantum as compact
import compare_quantum_classical as experiment


MODEL_NAMES = (
    "medium",
    "compact_quantum",
    "compact_frozen",
)

BUDGETS = (500, 1000, 2000, 5000, 10_000)


def make_policy(name: str, seed: int) -> nn.Module:
    if name == "medium":
        return experiment.make_policy("medium", seed)

    policy = compact.make_policy("compact_quantum", seed)

    if name == "compact_frozen":
        policy.block.weights.requires_grad_(False)

    return policy


def total_parameters(policy: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in policy.parameters()
    )


def check_trainable_gradients(policy: nn.Module) -> None:
    generator = torch.Generator().manual_seed(654321)

    inputs = torch.randn(
        (7, 22),
        generator=generator,
    )
    logits = policy(inputs)

    if logits.shape != (7,) or not torch.isfinite(logits).all():
        raise RuntimeError("Invalid policy outputs.")

    named_parameters = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]

    loss = -torch.log_softmax(logits, dim=0)[0]

    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named_parameters],
    )

    gradient_map = {
        name: gradient
        for (name, _), gradient in zip(
            named_parameters,
            gradients,
        )
    }

    for name, gradient in gradient_map.items():
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"Invalid gradient: {name}")

    # Frozen circuit weights must not block encoder gradients.
    for name in (
        "encoder.0.weight",
        "projection.0.weight",
    ):
        if name in gradient_map:
            if gradient_map[name].norm().item() <= 1e-12:
                raise RuntimeError(
                    f"Missing effective gradient: {name}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
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
        default=Path("results/sample_efficiency"),
    )

    args = parser.parse_args()

    if args.eval_matches < 4 or args.eval_matches % 2:
        parser.error(
            "--eval-matches must be an even number >= 4."
        )

    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative.")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must not repeat.")

    if len(set(args.models)) != len(args.models):
        parser.error("Models must not repeat.")

    if not args.reference.is_file():
        parser.error("Reference checkpoint does not exist.")

    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            parser.error("Output path is not a directory.")

        if any(args.output_dir.iterdir()):
            parser.error(
                "Choose an empty output directory."
            )

    torch.set_num_threads(1)
    experiment.game.test_rules()
    compact.verify_quantum_correlations()

    reference = experiment.critic_experiment.load_policy(
        args.reference
    )
    reference.eval()
    reference.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "runner": Path(__file__).resolve(),
        "shared_training": Path(
            experiment.__file__
        ).resolve(),
        "compact_models": Path(compact.__file__).resolve(),
        "fast_simulator": Path(
            compact.fast_quantum.__file__
        ).resolve(),
        "game": Path(
            experiment.game.__file__
        ).resolve(),
        "critic": Path(
            experiment.critic_experiment.__file__
        ).resolve(),
        "policy_sizes": (
            Path(experiment.__file__).resolve().parent
            / "compare_policy_sizes.py"
        ),
    }

    configuration = {
        "models": args.models,
        "seeds": args.seeds,
        "training_budgets": list(BUDGETS),
        "eval_matches": args.eval_matches,
        "evaluation_opponent": "v2",
        "evaluation_mode": "greedy",
        "strategic_filter": False,
        "train_opponents": list(
            experiment.TRAIN_OPPONENTS
        ),
        "actor_lr": experiment.ACTOR_LR,
        "critic_lr": experiment.CRITIC_LR,
        "entropy_coefficient": (
            experiment.ENTROPY_COEFFICIENT
        ),
        "max_grad_norm": experiment.MAX_GRAD_NORM,
        "goal": experiment.game.GOAL,
        "qubits": compact.QUBITS,
        "block_depth": compact.DEPTH,
        "quantum_backend": "torch_statevector",
        "torch_version": str(torch.__version__),
        "pennylane_version": str(compact.qml.__version__),
        "reference_sha256": experiment.file_hash(
            args.reference
        ),
        "source_sha256": {
            name: experiment.file_hash(path)
            for name, path in sources.items()
        },
    }

    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    for seed in args.seeds:
        for name in args.models:
            policy = make_policy(name, seed)
            check_trainable_gradients(policy)

            run_directory = (
                args.output_dir / f"{name}_seed_{seed}"
            )
            run_directory.mkdir()

            run_args = argparse.Namespace(
                episodes=max(BUDGETS),
                log_every=2000,
                max_rolls=300,
                output_dir=run_directory,
            )

            frozen_initial = None

            if name == "compact_frozen":
                frozen_initial = (
                    policy.block.weights.detach().clone()
                )

            print(
                f"\n{name}, seed {seed}: "
                f"total={total_parameters(policy)}, "
                f"trainable="
                f"{experiment.parameter_count(policy)}",
                flush=True,
            )

            def save_checkpoint(
                episode_number: int,
                stats: experiment.DecisionStats,
                elapsed: float,
            ) -> None:
                if episode_number not in BUDGETS:
                    return

                torch.save(
                    {
                        "model": name,
                        "seed": seed,
                        "episodes": episode_number,
                        "state_dict": policy.state_dict(),
                        "train_seconds": elapsed,
                        "train_policy_calls": (
                            stats.policy_calls
                        ),
                        "train_action_evals": (
                            stats.action_evals
                        ),
                        "train_rolls": stats.rolls,
                        "trainable_parameters": (
                            experiment.parameter_count(policy)
                        ),
                        "total_parameters": (
                            total_parameters(policy)
                        ),
                    },
                    run_directory
                    / f"episode_{episode_number}.pt",
                )
                print(
                    f"  Checkpoint saved: {name}, "
                    f"seed {seed}, episode {episode_number}",
                    flush=True,
                )

            critic, _, _ = experiment.train_policy(
                policy=policy,
                reference=reference,
                name=name,
                seed=seed,
                args=run_args,
                checkpoint_callback=save_checkpoint,
            )

            torch.save(
                critic.state_dict(),
                run_directory / "final_critic.pt",
            )

            if frozen_initial is not None:
                if not torch.equal(
                    frozen_initial,
                    policy.block.weights.detach(),
                ):
                    raise RuntimeError(
                        "Frozen quantum weights changed."
                    )

                print("Frozen weights verified unchanged.")

            # Evaluate only after training is finished.
            # Evaluation does not affect the training RNG stream.
            for budget in BUDGETS:
                checkpoint = torch.load(
                    run_directory
                    / f"episode_{budget}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                policy.load_state_dict(
                    checkpoint["state_dict"]
                )
                policy.eval()

                stats = experiment.DecisionStats()

                for match_index in range(
                    args.eval_matches
                ):
                    match_seed = (
                        100_000_000 + match_index // 2
                    )

                    experiment.play_match(
                        policy=policy,
                        reference=reference,
                        opponent_kind="v2",
                        dice_rng=experiment.random.Random(
                            match_seed
                        ),
                        heuristic_rng=(
                            experiment.random.Random(
                                match_seed + 10_000_000
                            )
                        ),
                        action_generator=(
                            torch.Generator().manual_seed(
                                match_seed + 20_000_000
                            )
                        ),
                        first_player=match_index % 2,
                        training=False,
                        sampled=False,
                        stats=stats,
                        max_rolls=300,
                    )

                row = {
                    "model": name,
                    "seed": seed,
                    "train_episodes": budget,
                    "train_policy_calls": checkpoint[
                        "train_policy_calls"
                    ],
                    "train_action_evals": checkpoint[
                        "train_action_evals"
                    ],
                    "train_rolls": checkpoint[
                        "train_rolls"
                    ],
                    "train_seconds": checkpoint[
                        "train_seconds"
                    ],
                    "total_parameters": checkpoint[
                        "total_parameters"
                    ],
                    "trainable_parameters": checkpoint[
                        "trainable_parameters"
                    ],
                    **stats.as_dict(),
                }

                experiment.append_csv(
                    args.output_dir / "efficiency.csv",
                    row,
                )

                print(
                    f"  {name}, seed {seed}, "
                    f"budget={budget}: "
                    f"{stats.wins}/{stats.matches} "
                    f"= {stats.wins / stats.matches:.1%}",
                    flush=True,
                )

    print(f"\nResults: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
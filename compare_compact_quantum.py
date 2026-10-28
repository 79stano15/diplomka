from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pennylane as qml
import torch
from torch import nn

import compare_quantum_classical as experiment
import quantum_block_fast as fast_quantum


QUBITS = 4
DEPTH = 2

MODEL_NAMES = (
    "compact_quantum",
    "compact_no_ent",
    "compact_classical",
)

EXPECTED_PARAMETERS = {
    "compact_quantum": 389,
    "compact_no_ent": 389,
    "compact_classical": 405,
}

game = experiment.game


class QuantumCorrelationBlock(fast_quantum.FastQuantumBlock):
    """The existing circuit with four Z and four adjacent ZZ outputs."""

    def __init__(self, entangle: bool) -> None:
        super().__init__(
            entangle=entangle,
            qubits=QUBITS,
            depth=DEPTH,
        )

        # The inherited forward computes probabilities @ z_signs.
        # Extend the observable matrix from 4 to 8 columns.
        local_z = self.z_signs
        adjacent_zz = local_z * torch.roll(
            local_z,
            shifts=-1,
            dims=1,
        )

        self.register_buffer(
            "z_signs",
            torch.cat((local_z, adjacent_zz), dim=1),
            persistent=False,
        )


class ClassicalCorrelationBlock(nn.Module):
    """Four learned outputs and their adjacent products."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(QUBITS, QUBITS),
            nn.Tanh(),
            nn.Linear(QUBITS, QUBITS),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = self.network(inputs)

        adjacent_products = local * torch.roll(
            local,
            shifts=-1,
            dims=1,
        )

        return torch.cat(
            (local, adjacent_products),
            dim=-1,
        )


class CompactPolicy(nn.Module):
    """Small action scorer with a shared classical bypass."""

    def __init__(self, name: str) -> None:
        super().__init__()

        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")

        self.encoder = nn.Sequential(
            nn.Linear(22, 8),
            nn.Tanh(),
        )

        self.projection = nn.Sequential(
            nn.Linear(8, QUBITS),
            nn.Tanh(),
        )

        # Initialize all shared components before the differing block.
        # For a given seed, these start identically across variants.
        self.readout = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

        if name == "compact_classical":
            self.block = ClassicalCorrelationBlock()
        else:
            self.block = QuantumCorrelationBlock(
                entangle=name == "compact_quantum",
            )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        classical_features = self.encoder(actions)
        block_inputs = self.projection(classical_features)
        block_outputs = self.block(block_inputs)

        combined = torch.cat(
            (classical_features, block_outputs),
            dim=-1,
        )

        return self.readout(combined).squeeze(-1)


def make_policy(name: str, seed: int) -> CompactPolicy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = CompactPolicy(name)

    actual = experiment.parameter_count(policy)
    expected = EXPECTED_PARAMETERS[name]

    if actual != expected:
        raise RuntimeError(
            f"{name}: expected {expected} parameters, got {actual}."
        )

    return policy


def verify_quantum_correlations() -> None:
    """Compare all eight outputs and gradients against PennyLane."""

    generator = torch.Generator().manual_seed(202610)

    for entangle in (False, True):
        device = qml.device(
            "default.qubit",
            wires=QUBITS,
            shots=None,
        )

        @qml.qnode(
            device,
            interface="torch",
            diff_method="backprop",
        )
        def reference_circuit(
            inputs: torch.Tensor,
            weights: torch.Tensor,
        ):
            for layer in range(DEPTH):
                for wire in range(QUBITS):
                    qml.RY(
                        math.pi * inputs[..., wire],
                        wires=wire,
                    )
                    qml.RY(
                        weights[layer, wire, 0],
                        wires=wire,
                    )
                    qml.RZ(
                        weights[layer, wire, 1],
                        wires=wire,
                    )

                if entangle:
                    for wire in range(QUBITS):
                        qml.CNOT(
                            wires=[wire, (wire + 1) % QUBITS]
                        )

                for wire in range(QUBITS):
                    qml.RX(
                        weights[layer, wire, 2],
                        wires=wire,
                    )

            local_measurements = [
                qml.expval(qml.PauliZ(wire))
                for wire in range(QUBITS)
            ]

            correlation_measurements = [
                qml.expval(
                    qml.PauliZ(wire)
                    @ qml.PauliZ((wire + 1) % QUBITS)
                )
                for wire in range(QUBITS)
            ]

            return tuple(
                local_measurements + correlation_measurements
            )

        # Local RNG scope avoids changing policy initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            block = QuantumCorrelationBlock(entangle)

        test_weights = (
            2.0
            * math.pi
            * torch.rand(
                block.weights.shape,
                generator=generator,
            )
            - math.pi
        )

        with torch.no_grad():
            block.weights.copy_(test_weights)

        reference_weights = (
            test_weights.clone().requires_grad_(True)
        )

        for batch_size in (1, 17):
            inputs = (
                2.0
                * torch.rand(
                    (batch_size, QUBITS),
                    generator=generator,
                )
                - 1.0
            )

            fast_inputs = inputs.clone().requires_grad_(True)
            reference_inputs = (
                inputs.clone().requires_grad_(True)
            )

            fast_outputs = block(fast_inputs)

            reference_outputs = torch.stack(
                reference_circuit(
                    reference_inputs,
                    reference_weights,
                ),
                dim=-1,
            ).to(dtype=inputs.dtype)

            if fast_outputs.shape != (batch_size, 8):
                raise RuntimeError(
                    f"Unexpected block shape: {fast_outputs.shape}"
                )

            torch.testing.assert_close(
                fast_outputs,
                reference_outputs,
                rtol=1e-5,
                atol=3e-6,
            )

            coefficients = torch.randn(
                (8,),
                generator=generator,
            )

            def objective(outputs: torch.Tensor) -> torch.Tensor:
                logits = outputs @ coefficients

                return (
                    -torch.log_softmax(logits, dim=0)[0]
                    + 0.1 * outputs.square().mean()
                )

            fast_gradients = torch.autograd.grad(
                objective(fast_outputs),
                (fast_inputs, block.weights),
            )

            reference_gradients = torch.autograd.grad(
                objective(reference_outputs),
                (reference_inputs, reference_weights),
            )

            for actual, expected in zip(
                fast_gradients,
                reference_gradients,
            ):
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-3,
                    atol=3e-5,
                )

            # A product state must satisfy <Zi Zj> = <Zi><Zj>.
            if not entangle:
                local = fast_outputs[:, :QUBITS]
                correlations = fast_outputs[:, QUBITS:]

                expected_correlations = local * torch.roll(
                    local,
                    shifts=-1,
                    dims=1,
                )

                torch.testing.assert_close(
                    correlations,
                    expected_correlations,
                    rtol=1e-5,
                    atol=3e-6,
                )

            output_error = (
                fast_outputs - reference_outputs
            ).abs().max().item()

            gradient_error = max(
                (actual - expected).abs().max().item()
                for actual, expected in zip(
                    fast_gradients,
                    reference_gradients,
                )
            )

            print(
                f"Z/ZZ test OK: entangle={entangle}, "
                f"batch={batch_size}, "
                f"output error={output_error:.3e}, "
                f"gradient error={gradient_error:.3e}",
                flush=True,
            )


def save_configuration(args: argparse.Namespace) -> None:
    sources = {
        "compact_runner": Path(__file__).resolve(),
        "shared_runner": Path(experiment.__file__).resolve(),
        "fast_simulator": Path(fast_quantum.__file__).resolve(),
        "game": Path(game.__file__).resolve(),
        "critic": Path(
            experiment.critic_experiment.__file__
        ).resolve(),
        "policy_sizes": (
            Path(experiment.__file__).resolve().parent
            / "compare_policy_sizes.py"
        ),
    }

    configuration = {
        **vars(args),
        "reference": str(args.reference.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "goal": game.GOAL,
        "strategic_filter": False,
        "reference_mode": "greedy",
        "primary_evaluation": "greedy",
        "train_opponents": list(experiment.TRAIN_OPPONENTS),
        "eval_opponents": list(experiment.EVAL_OPPONENTS),
        "encoder": [22, 8],
        "projection": [8, QUBITS],
        "readout": [16, 8, 1],
        "classical_bypass": True,
        "activation": "Tanh",
        "qubits": QUBITS,
        "block_depth": DEPTH,
        "observables": [
            "Z0",
            "Z1",
            "Z2",
            "Z3",
            "Z0Z1",
            "Z1Z2",
            "Z2Z3",
            "Z3Z0",
        ],
        "quantum_device": "torch_statevector",
        "quantum_diff_method": "torch_autograd",
        "quantum_state_dtype": "complex128",
        "shots": None,
        "actor_lr": experiment.ACTOR_LR,
        "critic_lr": experiment.CRITIC_LR,
        "entropy_coefficient": (
            experiment.ENTROPY_COEFFICIENT
        ),
        "max_grad_norm": experiment.MAX_GRAD_NORM,
        "policy_parameters": EXPECTED_PARAMETERS,
        "torch_version": str(torch.__version__),
        "pennylane_version": str(qml.__version__),
        "reference_sha256": experiment.file_hash(
            args.reference
        ),
        "source_sha256": {
            name: experiment.file_hash(path)
            for name, path in sources.items()
        },
    }

    path = args.output_dir / "config.json"
    path.write_text(
        json.dumps(
            configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compact hybrid policies with a classical bypass "
            "and Z/ZZ quantum measurements."
        )
    )

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
        "--episodes",
        type=int,
        default=10_000,
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
        default=Path("results/compact_quantum_10k"),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--max-rolls",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Also run secondary sampled evaluation.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Verify implementation without training.",
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive.")

    if args.eval_matches < 4 or args.eval_matches % 2:
        parser.error(
            "--eval-matches must be an even number >= 4."
        )

    if args.log_every < 1 or args.max_rolls < 1:
        parser.error(
            "--log-every and --max-rolls must be positive."
        )

    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative.")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must not repeat.")

    if len(set(args.models)) != len(args.models):
        parser.error("Models must not repeat.")

    if not args.test_only:
        if not args.reference.is_file():
            parser.error(
                f"Missing checkpoint: {args.reference}"
            )

        if args.output_dir.exists():
            if not args.output_dir.is_dir():
                parser.error(
                    "--output-dir is not a directory."
                )

            if any(args.output_dir.iterdir()):
                parser.error(
                    "Output directory is not empty. "
                    "Choose a new --output-dir."
                )

    torch.set_num_threads(1)
    game.test_rules()
    verify_quantum_correlations()

    for name in MODEL_NAMES:
        policy = make_policy(name, seed=42)
        experiment.check_policy_gradients(policy)

        print(
            f"Policy test OK: {name}, "
            f"{experiment.parameter_count(policy)} parameters",
            flush=True,
        )

    if args.test_only:
        print("\nAll checks passed. No training performed.")
        return

    reference = experiment.critic_experiment.load_policy(
        args.reference
    )
    reference.eval()
    reference.requires_grad_(False)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    save_configuration(args)

    print("\nRules OK. Strategic filter: OFF.")
    print("Frozen v2 opponent: GREEDY.")
    print("CPU threads: 1.")
    print("Classical bypass: ON.")
    print("Quantum outputs: 4 local Z + 4 adjacent ZZ.")

    all_rows: list[dict[str, str | int | float]] = []

    for seed in args.seeds:
        for name in args.models:
            policy = make_policy(name, seed)
            experiment.check_policy_gradients(policy)

            print(
                f"\n{name}, seed {seed}: "
                f"{experiment.parameter_count(policy)} "
                "policy parameters; gradient check OK.",
                flush=True,
            )

            critic, stats, elapsed = experiment.train_policy(
                policy=policy,
                reference=reference,
                name=name,
                seed=seed,
                args=args,
            )

            checkpoint_path = (
                args.output_dir / f"{name}_seed_{seed}.pt"
            )

            torch.save(
                {
                    "model": name,
                    "seed": seed,
                    "episodes": args.episodes,
                    "state_dict": policy.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "train_seconds": elapsed,
                    "qubits": QUBITS,
                    "block_depth": DEPTH,
                    "classical_bypass": True,
                    "block_outputs": 8,
                    "strategic_filter": False,
                },
                checkpoint_path,
            )

            experiment.append_csv(
                args.output_dir / "training.csv",
                {
                    "model": name,
                    "seed": seed,
                    "parameters": experiment.parameter_count(
                        policy
                    ),
                    "train_seconds": elapsed,
                    **stats.as_dict(),
                },
            )

            all_rows.extend(
                experiment.evaluate_policy(
                    policy=policy,
                    reference=reference,
                    name=name,
                    seed=seed,
                    args=args,
                )
            )

    experiment.print_summary(
        all_rows,
        args.models,
    )

    print(
        f"\nResults: {args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import pennylane as qml
import torch
from torch import nn
from torch.distributions import Categorical
import quantum_block_fast as fast_quantum
from collections.abc import Callable

import dice_rl_critic as critic_experiment
import dice_rl_v2 as game
from compare_policy_sizes import ActionPolicy


QUBITS = 4
BLOCK_DEPTH = 2

TRAIN_OPPONENTS: tuple[str | int, ...] = (
    "v2",
    250,
    500,
    800,
)
EVAL_OPPONENTS: tuple[str | int, ...] = (
    "v2",
    250,
    500,
    650,
    800,
)

MODEL_NAMES = (
    "medium",
    "classical",
    "hybrid",
    "hybrid_no_ent",
)

ACTOR_LR = 0.0003
CRITIC_LR = 0.0005
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 1.0


class QuantumBlock(nn.Module):
    """Four bounded inputs -> four Pauli-Z expectation values."""

    def __init__(self, entangle: bool) -> None:
        super().__init__()

        self.weights = nn.Parameter(
            torch.empty(BLOCK_DEPTH, QUBITS, 3)
        )
        nn.init.uniform_(self.weights, -0.1, 0.1)

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
        def circuit(
            inputs: torch.Tensor,
            weights: torch.Tensor,
        ):
            for layer in range(BLOCK_DEPTH):
                # Re-upload the same encoded features in each layer.
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

            return tuple(
                qml.expval(qml.PauliZ(wire))
                for wire in range(QUBITS)
            )

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # PennyLane broadcasts over the batch of candidate actions.
        measurements = self.circuit(inputs, self.weights)
        outputs = torch.stack(measurements, dim=-1)

        # Keep the surrounding PyTorch network in its original dtype.
        return outputs.to(dtype=inputs.dtype)


class BlockPolicy(nn.Module):
    """Shared encoder/readout with a classical or quantum block."""

    def __init__(self, kind: str) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(22, 32),
            nn.Tanh(),
            nn.Linear(32, QUBITS),
            nn.Tanh(),
        )

        # Initialize shared components before the differing block.
        # With the same initialization seed, encoder/readout weights
        # are identical across the block-policy variants.
        self.readout = nn.Linear(QUBITS, 1)

        if kind == "classical":
            layers: list[nn.Module] = []

            for _ in range(BLOCK_DEPTH):
                layers.extend(
                    [
                        nn.Linear(QUBITS, QUBITS),
                        nn.Tanh(),
                    ]
                )

            self.block = nn.Sequential(*layers)

        elif kind in ("hybrid", "hybrid_no_ent"):
            self.block = fast_quantum.FastQuantumBlock(
                entangle=kind == "hybrid",
                qubits=QUBITS,
                depth=BLOCK_DEPTH,
            )

        else:
            raise ValueError(f"Unknown block-policy kind: {kind}")

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(actions)
        transformed = self.block(encoded)
        return self.readout(transformed).squeeze(-1)


def make_policy(name: str, seed: int) -> nn.Module:
    # Local initialization seed, independent of action sampling.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        if name == "medium":
            return ActionPolicy((32, 32))

        if name in MODEL_NAMES:
            return BlockPolicy(name)

    raise ValueError(f"Unknown model: {name}")


def make_critic(seed: int) -> critic_experiment.ValueNetwork:
    # Identical initial critic for all models sharing a training seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1_000_000)
        return critic_experiment.ValueNetwork()


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def check_policy_gradients(policy: nn.Module) -> None:
    """Check batched output, singleton output and policy gradients."""

    generator = torch.Generator().manual_seed(987_654)
    inputs = torch.randn(
        (5, 22),
        generator=generator,
        dtype=torch.float32,
    )

    logits = policy(inputs)

    if logits.shape != (5,):
        raise RuntimeError(
            f"Expected logits shape (5,), got {logits.shape}."
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite policy outputs.")

    # A genuine softmax-policy objective, not a raw-output loss.
    loss = -torch.log_softmax(logits, dim=0)[0]
    named_parameters = list(policy.named_parameters())

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
            raise RuntimeError(
                f"Non-finite gradient in {name}."
            )

    for name in ("encoder.0.weight", "block.weights"):
        if name in gradient_map:
            norm = gradient_map[name].norm().item()

            if norm <= 1e-12:
                raise RuntimeError(
                    f"Gradient check failed for {name}."
                )

    with torch.no_grad():
        singleton = policy(inputs[:1])

    if singleton.shape != (1,):
        raise RuntimeError(
            "The policy does not preserve a singleton batch."
        )


@dataclass
class DecisionStats:
    matches: int = 0
    wins: int = 0
    rolls: int = 0
    policy_calls: int = 0
    action_evals: int = 0

    nonwinning_decisions: int = 0
    bank_decisions: int = 0
    bank_probability_sum: float = 0.0
    dominated_banks: int = 0

    winning_opportunities: int = 0
    missed_wins: int = 0

    def record(
        self,
        state: game.MatchState,
        actions: list[game.Action],
        chosen_index: int,
        probabilities: torch.Tensor,
    ) -> None:
        self.policy_calls += 1
        self.action_evals += len(actions)

        points_before_selection = (
            state.scores[state.player] + state.turn_points
        )

        winning = [
            points_before_selection + action.points >= game.GOAL
            for action in actions
        ]

        if any(winning):
            self.winning_opportunities += 1
            self.missed_wins += int(not winning[chosen_index])
            return

        self.nonwinning_decisions += 1

        probabilities_list = probabilities.detach().tolist()
        self.bank_probability_sum += sum(
            probability
            for action, probability in zip(
                actions,
                probabilities_list,
            )
            if action.bank
        )

        chosen = actions[chosen_index]
        self.bank_decisions += int(chosen.bank)

        max_points = max(action.points for action in actions)

        if chosen.bank and chosen.points < max_points:
            self.dominated_banks += 1

    def as_dict(self) -> dict[str, int | float]:
        denominator = self.nonwinning_decisions

        def rate(value: int | float) -> float:
            return (
                value / denominator
                if denominator
                else float("nan")
            )

        return {
            "matches": self.matches,
            "wins": self.wins,
            "win_rate": (
                self.wins / self.matches
                if self.matches
                else float("nan")
            ),
            "rolls": self.rolls,
            "policy_calls": self.policy_calls,
            "action_evals": self.action_evals,
            "nonwinning_decisions": denominator,
            "bank_rate": rate(self.bank_decisions),
            "mean_p_bank": rate(self.bank_probability_sum),
            "dominated_banks": self.dominated_banks,
            "dominated_bank_rate": rate(self.dominated_banks),
            "winning_opportunities": self.winning_opportunities,
            "missed_wins": self.missed_wins,
        }


def play_match(
    policy: nn.Module,
    reference: nn.Module,
    opponent_kind: str | int,
    dice_rng: random.Random,
    heuristic_rng: random.Random,
    action_generator: torch.Generator,
    first_player: int,
    training: bool,
    sampled: bool,
    stats: DecisionStats,
    max_rolls: int,
) -> critic_experiment.Episode:
    """Play an unfiltered match; the candidate is always player 0."""

    state = game.MatchState(
        scores=[0, 0],
        player=first_player,
    )

    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    critic_inputs: list[torch.Tensor] = []

    for roll_number in range(1, max_rolls + 1):
        stats.rolls += 1

        values = [
            dice_rng.randint(1, 6)
            for _ in range(state.dice_available)
        ]
        state.roll = game.count_dice(values)

        # Deliberately no strategic action filter.
        actions = game.legal_actions(state)

        if not actions:
            state.turn_points = 0
            state.dice_available = 6
            state.player = 1 - state.player
            state.roll = None
            continue

        if state.player == 0:
            with torch.set_grad_enabled(training):
                batch = torch.stack(
                    [
                        game.action_features(state, action)
                        for action in actions
                    ]
                )
                logits = policy(batch)
                distribution = Categorical(logits=logits)

                if training or sampled:
                    index = torch.multinomial(
                        distribution.probs.detach(),
                        num_samples=1,
                        generator=action_generator,
                    ).squeeze(0)
                else:
                    index = logits.argmax()

                chosen_index = int(index.item())
                action = actions[chosen_index]

                stats.record(
                    state,
                    actions,
                    chosen_index,
                    distribution.probs,
                )

                if training:
                    critic_inputs.append(
                        critic_experiment.critic_features(state)
                    )
                    log_probs.append(
                        distribution.log_prob(index)
                    )
                    entropies.append(distribution.entropy())

        elif opponent_kind == "v2":
            batch = torch.stack(
                [
                    game.action_features(state, action)
                    for action in actions
                ]
            )

            # The frozen v2 opponent always uses greedy selection.
            with torch.no_grad():
                chosen_index = int(
                    reference(batch).argmax().item()
                )

            action = actions[chosen_index]

        else:
            action = game.choose_heuristic_action(
                state,
                heuristic_rng,
                bank_threshold=int(opponent_kind),
            )

        outcome = game.apply_action(state, action)

        if outcome == "win":
            if state.winner is None:
                raise RuntimeError("Match ended without a winner.")

            stats.matches += 1
            stats.wins += int(state.winner == 0)

            return critic_experiment.Episode(
                winner=state.winner,
                rolls=roll_number,
                log_probs=log_probs,
                entropies=entropies,
                critic_inputs=critic_inputs,
            )

    # Never silently discard or relabel an unfinished match.
    raise RuntimeError(
        f"Match exceeded {max_rolls} rolls. "
        "Investigate before continuing the experiment."
    )


def append_csv(
    path: Path,
    row: dict[str, str | int | float],
) -> None:
    exists = path.exists()

    # Write the UTF-8 BOM only once, for Windows/Excel.
    encoding = "utf-8" if exists else "utf-8-sig"

    with path.open(
        "a",
        newline="",
        encoding=encoding,
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(row),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def train_policy(
    policy: nn.Module,
    reference: nn.Module,
    name: str,
    seed: int,
    args: argparse.Namespace,
    checkpoint_callback: Callable[
        [int, DecisionStats, float], None
    ]
    | None = None,
) -> tuple[
    critic_experiment.ValueNetwork,
    DecisionStats,
    float,
]:
    critic = make_critic(seed)

    policy.train()
    critic.train()

    actor_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=ACTOR_LR,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=CRITIC_LR,
    )

    # Opponent selection and starting player are independent of
    # how many random numbers a particular game consumes.
    schedule_rng = random.Random(seed + 101)

    # Action sampling is separate from all dice randomness.
    action_generator = torch.Generator().manual_seed(seed + 303)

    stats = DecisionStats()
    recent_wins: list[int] = []
    started = time.perf_counter()

    for episode_number in range(1, args.episodes + 1):
        opponent_kind = schedule_rng.choice(TRAIN_OPPONENTS)
        first_player = schedule_rng.randrange(2)

        episode = play_match(
            policy=policy,
            reference=reference,
            opponent_kind=opponent_kind,
            dice_rng=random.Random(
                1_000_000 * seed + episode_number
            ),
            heuristic_rng=random.Random(
                1_000_000 * seed + 100_000 + episode_number
            ),
            action_generator=action_generator,
            first_player=first_player,
            training=True,
            sampled=True,
            stats=stats,
            max_rolls=args.max_rolls,
        )

        reward = 1.0 if episode.winner == 0 else -1.0
        recent_wins.append(int(episode.winner == 0))

        if episode.log_probs:
            states = torch.stack(episode.critic_inputs)
            values = critic(states)
            targets = torch.full_like(values, reward)

            # Monte Carlo REINFORCE with a detached state baseline.
            advantages = (targets - values).detach()

            log_probs = torch.stack(episode.log_probs)
            entropies = torch.stack(episode.entropies)

            actor_loss = (
                -(advantages * log_probs).sum()
                - ENTROPY_COEFFICIENT * entropies.sum()
            )
            critic_loss = ((values - targets) ** 2).mean()

            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                MAX_GRAD_NORM,
            )
            actor_optimizer.step()

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                MAX_GRAD_NORM,
            )
            critic_optimizer.step()

        if checkpoint_callback is not None:
            checkpoint_callback(
                episode_number,
                stats,
                time.perf_counter() - started,
            )

        if (
            episode_number % args.log_every == 0
            or episode_number == args.episodes
        ):
            window = recent_wins[-200:]
            recent_win_rate = sum(window) / len(window)
            elapsed = time.perf_counter() - started

            append_csv(
                args.output_dir / "learning_curves.csv",
                {
                    "model": name,
                    "seed": seed,
                    "episode": episode_number,
                    "recent_win_rate": recent_win_rate,
                    "elapsed_seconds": elapsed,
                    "policy_calls": stats.policy_calls,
                    "action_evals": stats.action_evals,
                },
            )

            print(
                f"  {name}, seed {seed}, "
                f"episode {episode_number}/{args.episodes}: "
                f"recent wins {recent_win_rate:.1%}, "
                f"time {elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    policy.eval()
    critic.eval()

    return critic, stats, elapsed


def evaluate_policy(
    policy: nn.Module,
    reference: nn.Module,
    name: str,
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, str | int | float]]:
    policy.eval()
    rows: list[dict[str, str | int | float]] = []

    modes = ["greedy"]

    if args.sampled:
        modes.append("sampled")

    for opponent_index, opponent_kind in enumerate(
        EVAL_OPPONENTS
    ):
        for mode in modes:
            stats = DecisionStats()
            started = time.perf_counter()

            for match_index in range(args.eval_matches):
                pair_index = match_index // 2

                # Each evaluation seed is used once with the
                # candidate starting and once playing second.
                match_seed = (
                    100_000_000
                    + opponent_index * 1_000_000
                    + pair_index
                )

                play_match(
                    policy=policy,
                    reference=reference,
                    opponent_kind=opponent_kind,
                    dice_rng=random.Random(match_seed),
                    heuristic_rng=random.Random(
                        match_seed + 10_000_000
                    ),
                    action_generator=(
                        torch.Generator().manual_seed(
                            match_seed + 20_000_000
                        )
                    ),
                    first_player=match_index % 2,
                    training=False,
                    sampled=mode == "sampled",
                    stats=stats,
                    max_rolls=args.max_rolls,
                )

            elapsed = time.perf_counter() - started

            row: dict[str, str | int | float] = {
                "model": name,
                "seed": seed,
                "parameters": parameter_count(policy),
                "opponent": str(opponent_kind),
                "mode": mode,
                "eval_seconds": elapsed,
                **stats.as_dict(),
            }

            append_csv(
                args.output_dir / "evaluation.csv",
                row,
            )
            rows.append(row)

            print(
                f"  eval {name}, {mode}, "
                f"opponent {opponent_kind}: "
                f"{stats.wins}/{stats.matches} "
                f"= {stats.wins / stats.matches:.1%}, "
                f"missed wins "
                f"{stats.missed_wins}/"
                f"{stats.winning_opportunities}",
                flush=True,
            )

    return rows


def print_summary(
    rows: list[dict[str, str | int | float]],
    models: list[str],
) -> None:
    print("\nPRIMARY RESULTS: GREEDY")
    print("Mean win rate and SD across training seeds.")
    print("SD is reported in percentage points.")

    for name in models:
        print(f"\n{name}")

        for opponent in EVAL_OPPONENTS:
            selected = [
                row
                for row in rows
                if row["model"] == name
                and row["mode"] == "greedy"
                and row["opponent"] == str(opponent)
            ]
            values = [
                float(row["win_rate"])
                for row in selected
            ]

            mean = statistics.mean(values)
            sd = (
                statistics.stdev(values)
                if len(values) > 1
                else None
            )

            spread = (
                f" +/- {100 * sd:.2f} pp"
                if sd is not None
                else " (one training seed)"
            )

            print(
                f"  vs {str(opponent):>3}: "
                f"{mean:.1%}{spread}"
            )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unfiltered classical/quantum RL comparison."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["medium", "classical", "hybrid"],
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
        default=Path("results/quantum_classical"),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--max-rolls",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Also evaluate sampled action selection.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive.")

    if (
        args.eval_matches < 4
        or args.eval_matches % 2 != 0
    ):
        parser.error(
            "--eval-matches must be an even number >= 4."
        )

    if args.log_every < 1 or args.max_rolls < 1:
        parser.error(
            "--log-every and --max-rolls must be positive."
        )

    if any(seed < 0 for seed in args.seeds):
        parser.error("Training seeds must be nonnegative.")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must not repeat.")

    if len(set(args.models)) != len(args.models):
        parser.error("Models must not repeat.")

    if not args.reference.is_file():
        parser.error(
            f"Missing reference checkpoint: {args.reference}"
        )

    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            parser.error("--output-dir is not a directory.")

        if any(args.output_dir.iterdir()):
            parser.error(
                "Output directory is not empty. "
                "Choose a new --output-dir."
            )

    torch.set_num_threads(1)
    game.test_rules()

    # Immediate victory is independent of the bank flag.
    test_state = game.MatchState(
        scores=[3_900, 3_000],
        dice_available=1,
        roll=game.count_dice([1]),
    )
    test_action = game.Action(
        keep=game.count_dice([1]),
        points=100,
        bank=False,
    )

    if game.apply_action(test_state, test_action) != "win":
        raise RuntimeError(
            "Immediate-win semantics do not match the protocol."
        )

    reference = critic_experiment.load_policy(args.reference)
    reference.eval()
    reference.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "game": Path(game.__file__).resolve(),
        "critic": Path(critic_experiment.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "fast_quantum": Path(fast_quantum.__file__).resolve(),
        "policy_sizes": (
            Path(__file__).resolve().parent
            / "compare_policy_sizes.py"
        ),
    }

    config = {
        **vars(args),
        "reference": str(args.reference.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "goal": game.GOAL,
        "strategic_filter": False,
        "reference_mode": "greedy",
        "train_opponents": list(TRAIN_OPPONENTS),
        "eval_opponents": list(EVAL_OPPONENTS),
        "qubits": QUBITS,
        "block_depth": BLOCK_DEPTH,
        "shots": None,
        "quantum_device": "torch_statevector",
        "quantum_diff_method": "torch_autograd",
        "quantum_state_dtype": "complex128",
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "max_grad_norm": MAX_GRAD_NORM,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "pennylane_version": str(qml.__version__),
        "reference_sha256": file_hash(args.reference),
        "source_sha256": {
            name: file_hash(path)
            for name, path in source_paths.items()
        },
    }

    config_path = args.output_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_rows: list[dict[str, str | int | float]] = []

    print("Rules OK. Strategic filter: OFF.")
    print("Frozen v2 opponent: GREEDY.")
    print("CPU threads: 1.")

    for seed in args.seeds:
        for name in args.models:
            policy = make_policy(name, seed)
            check_policy_gradients(policy)

            parameters = parameter_count(policy)
            print(
                f"\n{name}, seed {seed}: "
                f"{parameters} policy parameters; "
                "gradient check OK.",
                flush=True,
            )

            critic, stats, elapsed = train_policy(
                policy,
                reference,
                name,
                seed,
                args,
            )

            checkpoint_path = (
                args.output_dir / f"{name}_seed_{seed}.pt"
            )

            # Save state dictionaries, never a pickled QNode.
            torch.save(
                {
                    "model": name,
                    "seed": seed,
                    "episodes": args.episodes,
                    "qubits": QUBITS,
                    "block_depth": BLOCK_DEPTH,
                    "state_dict": policy.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "train_seconds": elapsed,
                    "strategic_filter": False,
                },
                checkpoint_path,
            )

            append_csv(
                args.output_dir / "training.csv",
                {
                    "model": name,
                    "seed": seed,
                    "parameters": parameters,
                    "train_seconds": elapsed,
                    **stats.as_dict(),
                },
            )

            all_rows.extend(
                evaluate_policy(
                    policy,
                    reference,
                    name,
                    seed,
                    args,
                )
            )

    print_summary(all_rows, args.models)
    print(f"\nResults: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
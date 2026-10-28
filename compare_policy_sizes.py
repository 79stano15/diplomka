from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

import dice_rl_critic as critic_experiment
import dice_rl_v2 as game


ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "large": (64, 64),
    "small": (32, 16),
    "narrow": (16, 4, 4),
}

TRAIN_OPPONENTS: tuple[str | int, ...] = ("v2", 250, 500, 800)
EVAL_THRESHOLDS = (250, 500, 650, 800)


class ActionPolicy(nn.Module):
    """Vráti jedno skóre pre každú dvojicu stav–legálna akcia."""

    def __init__(self, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_size = 22

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.network(actions).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def play_training_match(
    policy: ActionPolicy,
    reference: game.PolicyNetwork,
    opponent_kind: str | int,
    rng: random.Random,
    max_rolls: int = 300,
) -> tuple[
    int,
    int,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Odohrá zápas a uloží rozhodnutia trénovaného hráča 0."""

    state = game.MatchState(
        scores=[0, 0],
        player=rng.randint(0, 1),
    )

    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    critic_inputs: list[torch.Tensor] = []

    for roll_number in range(1, max_rolls + 1):
        values = [
            rng.randint(1, 6)
            for _ in range(state.dice_available)
        ]
        state.roll = game.count_dice(values)
        actions = game.legal_actions(state)

        if not actions:
            state.turn_points = 0
            state.dice_available = 6
            state.player = 1 - state.player
            state.roll = None
            continue

        if state.player == 0:
            batch = torch.stack(
                [
                    game.action_features(state, action)
                    for action in actions
                ]
            )

            distribution = Categorical(logits=policy(batch))
            index = distribution.sample()
            action = actions[index.item()]

            critic_inputs.append(
                critic_experiment.critic_features(state)
            )
            log_probs.append(distribution.log_prob(index))
            entropies.append(distribution.entropy())

        elif opponent_kind == "v2":
            batch = torch.stack(
                [
                    game.action_features(state, action)
                    for action in actions
                ]
            )

            with torch.no_grad():
                index = reference(batch).argmax().item()

            action = actions[index]

        else:
            action = game.choose_heuristic_action(
                state,
                rng,
                bank_threshold=opponent_kind,
            )

        outcome = game.apply_action(state, action)

        if outcome == "win":
            if state.winner is None:
                raise RuntimeError("Zápas skončil bez víťaza.")

            return (
                state.winner,
                roll_number,
                log_probs,
                entropies,
                critic_inputs,
            )

    raise RuntimeError(
        "Zápas prekročil limit hodov. "
        "Skontroluj priebeh alebo zvýš max_rolls."
    )


def train_one(
    name: str,
    seed: int,
    episodes: int,
    reference: game.PolicyNetwork,
) -> ActionPolicy:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy = ActionPolicy(ARCHITECTURES[name])
    critic = critic_experiment.ValueNetwork()

    policy.train()
    critic.train()

    actor_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=0.0003,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=0.0005,
    )

    recent_wins: list[int] = []

    print(
        f"\n{name}, seed {seed}: "
        f"{parameter_count(policy)} parametrov politiky",
        flush=True,
    )

    for episode_number in range(1, episodes + 1):
        opponent_kind = rng.choice(TRAIN_OPPONENTS)

        (
            winner,
            _rolls,
            log_probs,
            entropies,
            critic_inputs,
        ) = play_training_match(
            policy,
            reference,
            opponent_kind,
            rng,
        )

        reward = 1.0 if winner == 0 else -1.0
        recent_wins.append(int(winner == 0))

        if log_probs:
            values = critic(torch.stack(critic_inputs))
            targets = torch.full_like(values, reward)
            advantages = (targets - values).detach()

            actor_loss = (
                -(advantages * torch.stack(log_probs)).sum()
                - 0.01 * torch.stack(entropies).sum()
            )
            critic_loss = ((values - targets) ** 2).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=1.0,
            )
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=1.0,
            )
            critic_optimizer.step()

        if episode_number % 2_000 == 0 or episode_number == episodes:
            window = min(200, len(recent_wins))
            win_rate = sum(recent_wins[-window:]) / window

            print(
                f"  epizóda {episode_number:5d}/{episodes}: "
                f"tréningové výhry {win_rate:.1%}",
                flush=True,
            )

    policy.eval()
    return policy


def evaluate_vs_heuristic(
    policy: ActionPolicy,
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


def evaluate_vs_v2(
    policy: ActionPolicy,
    reference: game.PolicyNetwork,
    matches: int,
    seed: int,
) -> float:
    """Politika hrá polovicu zápasov v každej hráčskej pozícii."""

    wins = 0
    matches_per_seat = matches // 2

    with torch.no_grad():
        for candidate_seat in (0, 1):
            player_zero = (
                policy if candidate_seat == 0 else reference
            )
            player_one = (
                reference if candidate_seat == 0 else policy
            )

            for index in range(matches_per_seat):
                episode = critic_experiment.play_match(
                    policy=player_zero,
                    opponent=player_one,
                    rng=random.Random(seed + index),
                    training=False,
                    first_player=index % 2,
                )
                wins += int(episode.winner == candidate_seat)

    return wins / matches


def evaluate_one(
    name: str,
    seed: int,
    policy: ActionPolicy,
    reference: game.PolicyNetwork,
    matches: int,
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "architecture": name,
        "seed": seed,
        "parameters": parameter_count(policy),
        "vs_v2": evaluate_vs_v2(
            policy,
            reference,
            matches,
            seed=100_000,
        ),
    }

    for threshold in EVAL_THRESHOLDS:
        row[f"vs_{threshold}"] = evaluate_vs_heuristic(
            policy,
            threshold,
            matches,
            seed=200_000 + threshold * 1_000,
        )

    return row


def print_summary(
    rows: list[dict[str, str | int | float]],
) -> None:
    metrics = [
        "vs_v2",
        "vs_250",
        "vs_500",
        "vs_650",
        "vs_800",
    ]

    print("\nPriemer ± SD medzi tréningovými seedmi:")
    print(
        "model  | parametre | v2 | 250 | 500 | 650 | 800"
    )

    for name in ARCHITECTURES:
        selected = [
            row for row in rows
            if row["architecture"] == name
        ]

        results: list[str] = []

        for metric in metrics:
            values = [
                float(row[metric])
                for row in selected
            ]
            mean = statistics.mean(values)

            if len(values) > 1:
                sd = statistics.stdev(values)
                results.append(f"{mean:.1%}±{sd:.1%}")
            else:
                results.append(f"{mean:.1%}")

        print(
            f"{name:<6} | "
            f"{selected[0]['parameters']:>9} | "
            + " | ".join(results)
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
        default=Path("results/policy_sizes_10k"),
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes musí byť aspoň 1")

    if args.eval_matches < 4 or args.eval_matches % 4 != 0:
        parser.error(
            "--eval-matches musí byť kladný násobok štyroch"
        )

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Tréningové seedy sa nesmú opakovať")

    if not args.reference.is_file():
        parser.error(
            f"Chýba referenčný model: {args.reference}"
        )

    planned_paths = [
        args.output_dir / f"{name}_seed_{seed}.pt"
        for name in ARCHITECTURES
        for seed in args.seeds
    ]
    existing = [
        path for path in planned_paths if path.exists()
    ]

    if existing:
        parser.error(
            f"Checkpoint už existuje: {existing[0]}. "
            "Použi iný --output-dir."
        )

    game.test_rules()
    torch.set_num_threads(1)

    reference = critic_experiment.load_policy(
        args.reference
    )
    reference.eval()
    reference.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []

    for seed in args.seeds:
        for name in ARCHITECTURES:
            policy = train_one(
                name,
                seed,
                args.episodes,
                reference,
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

            row = evaluate_one(
                name,
                seed,
                policy,
                reference,
                args.eval_matches,
            )
            rows.append(row)

            print(
                f"  uložené: {path} | "
                f"proti v2: {row['vs_v2']:.1%}",
                flush=True,
            )

    csv_path = args.output_dir / "results.csv"

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

    print_summary(rows)
    print(f"\nVýsledky uložené do {csv_path}")


if __name__ == "__main__":
    main()
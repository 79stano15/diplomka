from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

import dice_rl_v2 as game


@dataclass
class Episode:
    winner: int
    rolls: int
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    critic_inputs: list[torch.Tensor]


class ValueNetwork(nn.Module):
    """Odhad výsledku zápasu z pohľadu hráča 0 v aktuálnom stave."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


def critic_features(state: game.MatchState) -> torch.Tensor:
    """Príznaky stavu pred rozhodnutím hráča 0."""

    if state.roll is None or state.player != 0:
        raise ValueError("Kritik očakáva hod hráča 0.")

    own = state.scores[0]
    opponent = state.scores[1]
    goal = game.GOAL

    values = [
        own / goal,
        opponent / goal,
        (own - opponent) / goal,
        state.turn_points / goal,
        state.dice_available / 6,
        (goal - own - state.turn_points) / goal,
    ]

    values.extend(amount / 6 for amount in state.roll)

    return torch.tensor(values, dtype=torch.float32)


def choose_policy_action(
    policy: game.PolicyNetwork,
    state: game.MatchState,
    actions: list[game.Action],
    training: bool,
) -> tuple[
    game.Action,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    batch = torch.stack(
        [game.action_features(state, action) for action in actions]
    )
    logits = policy(batch)

    if training:
        distribution = Categorical(logits=logits)
        index = distribution.sample()

        return (
            actions[index.item()],
            distribution.log_prob(index),
            distribution.entropy(),
        )

    index = torch.argmax(logits).item()
    return actions[index], None, None


def play_match(
    policy: game.PolicyNetwork,
    opponent: game.PolicyNetwork,
    rng: random.Random,
    training: bool,
    first_player: int | None = None,
    verbose: bool = False,
    max_rolls: int = 300,
) -> Episode:
    if first_player is None:
        first_player = rng.randint(0, 1)

    state = game.MatchState(scores=[0, 0], player=first_player)

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

        if verbose:
            print(
                f"\nHod {roll_number}: hráč {state.player}, "
                f"skóre {state.scores}, v ťahu {state.turn_points}, "
                f"kocky {game.format_dice(state.roll)}"
            )

        if not actions:
            if verbose:
                print("  FARKLE: priebežné body prepadli.")

            state.turn_points = 0
            state.dice_available = 6
            state.player = 1 - state.player
            state.roll = None
            continue

        if state.player == 0:
            if training:
                critic_inputs.append(critic_features(state))

            action, log_prob, entropy = choose_policy_action(
                policy,
                state,
                actions,
                training=training,
            )

            if log_prob is not None and entropy is not None:
                log_probs.append(log_prob)
                entropies.append(entropy)
        else:
            with torch.no_grad():
                action, _, _ = choose_policy_action(
                    opponent,
                    state,
                    actions,
                    training=False,
                )

        decision = game.action_label(state, action)

        if verbose:
            print(
                f"  Odloží {game.format_dice(action.keep)} "
                f"za {action.points} → {decision}"
            )

        outcome = game.apply_action(state, action)

        if verbose:
            print(
                f"  Skóre {state.scores}, "
                f"priebežne v ťahu {state.turn_points}"
            )

        if outcome == "win":
            if state.winner is None:
                raise RuntimeError("Hra skončila bez určenia víťaza.")

            if verbose:
                print(f"  VYHRAL HRÁČ {state.winner}")

            return Episode(
                winner=state.winner,
                rolls=roll_number,
                log_probs=log_probs,
                entropies=entropies,
                critic_inputs=critic_inputs,
            )

    raise RuntimeError(
        "Zápas prekročil limit hodov; skontroluj priebeh hry."
    )


def load_policy(path: Path) -> game.PolicyNetwork:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint neexistuje: {path}")

    policy = game.PolicyNetwork()
    policy.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    return policy


def train(
    initial_path: Path,
    opponent_path: Path,
    episodes: int,
    seed: int,
    entropy_coefficient: float,
    actor_lr: float,
    critic_lr: float,
) -> tuple[game.PolicyNetwork, ValueNetwork]:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy = load_policy(initial_path)
    policy.train()

    opponent = load_policy(opponent_path)
    opponent.eval()
    opponent.requires_grad_(False)

    critic = ValueNetwork()
    critic.train()

    actor_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=actor_lr,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=critic_lr,
    )

    recent_wins: list[int] = []
    recent_rolls: list[int] = []
    recent_entropies: list[float] = []
    recent_critic_errors: list[float] = []

    for episode_number in range(1, episodes + 1):
        episode = play_match(
            policy=policy,
            opponent=opponent,
            rng=rng,
            training=True,
        )

        reward = 1.0 if episode.winner == 0 else -1.0

        if episode.log_probs:
            states = torch.stack(episode.critic_inputs)
            values = critic(states)

            targets = torch.full_like(values, reward)
            advantages = (targets - values).detach()

            log_probs = torch.stack(episode.log_probs)
            entropies = torch.stack(episode.entropies)

            actor_loss = (
                -(advantages * log_probs).sum()
                - entropy_coefficient * entropies.sum()
            )
            critic_loss = torch.mean((values - targets) ** 2)

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

            recent_entropies.append(entropies.mean().item())
            recent_critic_errors.append(critic_loss.item())

        recent_wins.append(int(episode.winner == 0))
        recent_rolls.append(episode.rolls)

        if episode_number % 200 == 0 or episode_number == episodes:
            window = min(200, episode_number)

            wins = sum(recent_wins[-window:]) / window
            rolls = sum(recent_rolls[-window:]) / window

            entropy_values = recent_entropies[-window:]
            critic_values = recent_critic_errors[-window:]

            entropy = (
                sum(entropy_values) / len(entropy_values)
                if entropy_values
                else float("nan")
            )
            critic_error = (
                sum(critic_values) / len(critic_values)
                if critic_values
                else float("nan")
            )

            print(
                f"Epizóda {episode_number:5d} | "
                f"výhry {wins:6.1%} | "
                f"hody {rolls:5.1f} | "
                f"entropia {entropy:6.3f} | "
                f"chyba kritika {critic_error:6.3f}"
            )

    return policy, critic


def evaluate_direct(
    policy: game.PolicyNetwork,
    opponent: game.PolicyNetwork,
    label: str,
    matches: int,
    seed: int,
) -> None:
    wins_first = 0
    wins_second = 0

    policy.eval()
    opponent.eval()

    with torch.no_grad():
        for index in range(matches):
            first_player = index % 2

            episode = play_match(
                policy=policy,
                opponent=opponent,
                rng=random.Random(seed + index),
                training=False,
                first_player=first_player,
            )

            if episode.winner == 0 and first_player == 0:
                wins_first += 1
            elif episode.winner == 0:
                wins_second += 1

    wins = wins_first + wins_second
    half = matches // 2

    print(
        f"{label}: {wins}/{matches} = {wins / matches:.1%}; "
        f"začína {wins_first}/{half}, "
        f"hrá druhý {wins_second}/{half}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-matches", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--initial",
        type=Path,
        default=Path("dice_policy_v2.pt"),
    )
    parser.add_argument(
        "--opponent",
        type=Path,
        default=Path("dice_policy_v2.pt"),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("dice_policy_critic.pt"),
    )

    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=0.0003,
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=0.0005,
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes musí byť kladné")

    if args.eval_matches < 2 or args.eval_matches % 2:
        parser.error("--eval-matches musí byť kladné párne číslo")

    game.test_rules()
    print("Testy pravidiel prešli.")
    print(f"Počiatočný model: {args.initial}")
    print(f"Zmrazený súper: {args.opponent}\n")

    policy, critic = train(
        initial_path=args.initial,
        opponent_path=args.opponent,
        episodes=args.episodes,
        seed=args.seed,
        entropy_coefficient=args.entropy_coefficient,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )

    torch.save(policy.state_dict(), args.save)

    critic_path = args.save.with_name(
        f"{args.save.stem}_value.pt"
    )
    torch.save(critic.state_dict(), critic_path)

    print(f"\nPolitika uložená do {args.save}")
    print(f"Kritik uložený do {critic_path}")

    reference = load_policy(args.opponent)

    print("\nPriame vyhodnotenie proti zmrazenému v2:")
    evaluate_direct(
        policy,
        reference,
        label="Nový model proti v2",
        matches=args.eval_matches,
        seed=50_000,
    )
    evaluate_direct(
        reference,
        policy,
        label="V2 proti novému modelu",
        matches=args.eval_matches,
        seed=60_000,
    )

    print("\nNezávislé testy proti pôvodným heuristikám:")
    game.evaluate(
        policy,
        label="model s kritikom",
        matches=args.eval_matches,
    )
    game.evaluate(
        reference,
        label="zmrazený v2",
        matches=args.eval_matches,
    )

    example_state = game.MatchState(
        scores=[2_800, 3_200],
        player=0,
        turn_points=350,
        dice_available=6,
        roll=game.count_dice([2, 2, 2, 1, 5, 3]),
    )

    print("\nDiagnostika rozhodnutia:")
    game.inspect_decision(policy, example_state)

    # print("\nUkážkový zápas proti v2:")
    # with torch.no_grad():
    #     play_match(
    #         policy=policy,
    #         opponent=reference,
    #         rng=random.Random(2026),
    #         training=False,
    #         verbose=True,
    #     )


if __name__ == "__main__":
    main()
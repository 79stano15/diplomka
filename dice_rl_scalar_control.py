from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

import dice_rl_critic as experiment
import dice_rl_v2 as game


def train(
    initial_path: Path,
    opponent_path: Path,
    episodes: int,
    seed: int,
    entropy_coefficient: float,
    actor_lr: float,
    baseline_rate: float,
) -> game.PolicyNetwork:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy = experiment.load_policy(initial_path)
    policy.train()

    opponent = experiment.load_policy(opponent_path)
    opponent.eval()
    opponent.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=actor_lr)

    # Odhad priemerného výsledku z predchádzajúcich zápasov.
    # Na rozdiel od kritika nezávisí od aktuálneho stavu hry.
    baseline = 0.0

    recent_wins: list[int] = []
    recent_rolls: list[int] = []
    recent_entropies: list[float] = []

    for episode_number in range(1, episodes + 1):
        episode = experiment.play_match(
            policy=policy,
            opponent=opponent,
            rng=rng,
            training=True,
        )

        reward = 1.0 if episode.winner == 0 else -1.0

        # Použijeme baseline z PRECHÁDZAJÚCICH epizód.
        advantage = reward - baseline

        if episode.log_probs:
            log_probs = torch.stack(episode.log_probs)
            entropies = torch.stack(episode.entropies)

            actor_loss = (
                -advantage * log_probs.sum()
                - entropy_coefficient * entropies.sum()
            )

            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            recent_entropies.append(entropies.mean().item())

        # Aktualizácia až PO výpočte gradientu tejto epizódy.
        baseline = (
            (1.0 - baseline_rate) * baseline
            + baseline_rate * reward
        )

        recent_wins.append(int(episode.winner == 0))
        recent_rolls.append(episode.rolls)

        if episode_number % 200 == 0 or episode_number == episodes:
            window = min(200, episode_number)
            entropies_window = recent_entropies[-window:]

            mean_entropy = (
                sum(entropies_window) / len(entropies_window)
                if entropies_window
                else float("nan")
            )

            print(
                f"Epizóda {episode_number:5d} | "
                f"výhry {sum(recent_wins[-window:]) / window:6.1%} | "
                f"hody {sum(recent_rolls[-window:]) / window:5.1f} | "
                f"entropia {mean_entropy:6.3f} | "
                f"baseline {baseline:+.3f}"
            )

    return policy


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
        "--critic-checkpoint",
        type=Path,
        default=Path("dice_policy_critic.pt"),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("dice_policy_scalar_control.pt"),
    )

    # Rovnaké predvolené hodnoty ako pri experimente s kritikom.
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
        "--baseline-rate",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes musí byť aspoň 1")

    if args.eval_matches < 2 or args.eval_matches % 2 != 0:
        parser.error("--eval-matches musí byť kladné párne číslo")

    if not 0.0 < args.baseline_rate <= 1.0:
        parser.error("--baseline-rate musí byť v intervale (0, 1]")

    game.test_rules()
    print("Testy pravidiel prešli.")
    print(f"Počiatočný model: {args.initial}")
    print(f"Zmrazený súper: {args.opponent}\n")

    policy = train(
        initial_path=args.initial,
        opponent_path=args.opponent,
        episodes=args.episodes,
        seed=args.seed,
        entropy_coefficient=args.entropy_coefficient,
        actor_lr=args.actor_lr,
        baseline_rate=args.baseline_rate,
    )

    torch.save(policy.state_dict(), args.save)
    print(f"\nKontrolný model uložený do {args.save}")

    reference = experiment.load_policy(args.opponent)

    print("\nPriame zápasy proti v2:")
    experiment.evaluate_direct(
        policy,
        reference,
        label="Skalárna baseline proti v2",
        matches=args.eval_matches,
        seed=50_000,
    )
    experiment.evaluate_direct(
        reference,
        policy,
        label="V2 proti skalárnej baseline",
        matches=args.eval_matches,
        seed=60_000,
    )

    print("\nTesty proti heuristikám:")
    game.evaluate(
        policy,
        label="skalárna baseline",
        matches=args.eval_matches,
    )
    game.evaluate(
        reference,
        label="zmrazený v2",
        matches=args.eval_matches,
    )

    if args.critic_checkpoint.exists():
        critic_policy = experiment.load_policy(
            args.critic_checkpoint
        )

        print("\nTest kritika na rovnakej sade heuristík:")
        game.evaluate(
            critic_policy,
            label="model s kritikom",
            matches=args.eval_matches,
        )

        print("\nPriamy zápas kontrolného modelu a kritika:")
        experiment.evaluate_direct(
            policy,
            critic_policy,
            label="Skalárna baseline proti kritikovi",
            matches=args.eval_matches,
            seed=70_000,
        )
        experiment.evaluate_direct(
            critic_policy,
            policy,
            label="Kritik proti skalárnej baseline",
            matches=args.eval_matches,
            seed=80_000,
        )


if __name__ == "__main__":
    main()
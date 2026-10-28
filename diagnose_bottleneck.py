from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn

import compare_policy_sizes as experiment
import dice_rl_critic as critic_experiment
import dice_rl_v2 as game


CHECKPOINT_DIR = Path("results/bottleneck4_10k")
ARCHITECTURE = (32, 4, 4)
SEEDS = (42, 43, 44)
THRESHOLDS = (250, 500, 650, 800)


class AlwaysBankMax(nn.Module):
    """Vždy bankuje najvyšší dostupný bodový zisk."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        points = features[:, 5] * game.GOAL
        bank = features[:, 9]
        return 10_000 * bank + points


def load_bottleneck(seed: int) -> experiment.ActionPolicy:
    path = CHECKPOINT_DIR / f"bottleneck4_seed_{seed}.pt"
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    policy = experiment.ActionPolicy(ARCHITECTURE)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def chosen_action(
    policy: nn.Module,
    state: game.MatchState,
) -> game.Action:
    actions = game.legal_actions(state)
    features = torch.stack(
        [game.action_features(state, action) for action in actions]
    )

    with torch.no_grad():
        index = policy(features).argmax().item()

    return actions[index]


def make_test_states(amount: int = 500) -> list[game.MatchState]:
    """Reprodukovateľné stavy bez okamžitej výhernej akcie."""

    rng = random.Random(12_345)
    states: list[game.MatchState] = []

    while len(states) < amount:
        dice_available = rng.randint(1, 6)
        roll = game.count_dice(
            [rng.randint(1, 6) for _ in range(dice_available)]
        )

        state = game.MatchState(
            scores=[
                50 * rng.randint(0, 64),
                50 * rng.randint(0, 78),
            ],
            player=0,
            turn_points=50 * rng.randint(0, 14),
            dice_available=dice_available,
            roll=roll,
        )

        actions = game.legal_actions(state)

        if len(actions) < 2:
            continue

        if any(
            state.scores[0] + state.turn_points + action.points
            >= game.GOAL
            for action in actions
        ):
            continue

        states.append(state)

    return states


def inspect_agreement(
    policy: experiment.ActionPolicy,
    greedy: AlwaysBankMax,
    states: list[game.MatchState],
    seed: int,
) -> None:
    agreements = 0
    banked = 0
    differences = []

    for state in states:
        model_action = chosen_action(policy, state)
        greedy_action = chosen_action(greedy, state)

        agreements += model_action == greedy_action
        banked += model_action.bank

        if model_action != greedy_action and len(differences) < 3:
            assert state.roll is not None

            differences.append(
                (
                    game.format_dice(state.roll),
                    model_action,
                    greedy_action,
                )
            )

    print(
        f"Seed {seed}: zhoda s 'vždy bankuj maximum' "
        f"{agreements}/{len(states)} = {agreements / len(states):.1%}; "
        f"bankovanie {banked / len(states):.1%}"
    )

    for roll, model_action, greedy_action in differences:
        print(
            f"  Hod {roll}: model "
            f"{game.format_dice(model_action.keep)} "
            f"za {model_action.points}, "
            f"{'BANK' if model_action.bank else 'POKRAČUJ'}; "
            f"jednoduchá stratégia "
            f"{game.format_dice(greedy_action.keep)} "
            f"za {greedy_action.points}, BANK"
        )


def main() -> None:
    game.test_rules()
    torch.set_num_threads(1)

    reference = critic_experiment.load_policy(
        Path("dice_policy_v2.pt")
    )
    reference.eval()

    greedy = AlwaysBankMax()
    states = make_test_states()

    print("Porovnanie rozhodnutí na rovnakých 500 stavoch:")

    for seed in SEEDS:
        inspect_agreement(
            load_bottleneck(seed),
            greedy,
            states,
            seed,
        )

    print("\nVýsledky jednoduchej stratégie na pôvodných testoch:")

    vs_v2 = experiment.evaluate_vs_v2(
        greedy,
        reference,
        matches=400,
        seed=100_000,
    )
    print(f"  proti v2: {vs_v2:.1%}")

    for threshold in THRESHOLDS:
        result = experiment.evaluate_vs_heuristic(
            greedy,
            threshold=threshold,
            matches=400,
            seed=200_000 + threshold * 1_000,
        )
        print(f"  proti prahu {threshold}: {result:.1%}")


if __name__ == "__main__":
    main()
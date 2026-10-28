from pathlib import Path

import torch

import dice_rl_v2 as game
from compare_policy_sizes import ARCHITECTURES, ActionPolicy


FOLDER = Path("results/policy_sizes_10k")

STATES = [
    game.MatchState(
        scores=[800, 800],
        turn_points=100,
        dice_available=6,
        roll=game.count_dice([1, 2, 3, 4, 5, 6]),
    ),
    game.MatchState(
        scores=[2_800, 3_200],
        turn_points=350,
        dice_available=6,
        roll=game.count_dice([2, 2, 2, 1, 5, 3]),
    ),
    game.MatchState(
        scores=[1_000, 3_600],
        turn_points=150,
        dice_available=6,
        roll=game.count_dice([1, 1, 2, 3, 4, 6]),
    ),
]


def load_narrow(seed: int) -> ActionPolicy:
    path = FOLDER / f"narrow_seed_{seed}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    policy = ActionPolicy(ARCHITECTURES["narrow"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def main() -> None:
    policies = {seed: load_narrow(seed) for seed in (42, 43, 44)}
    reference = policies[42].state_dict()

    for seed in (43, 44):
        candidate = policies[seed].state_dict()
        largest_difference = max(
            (candidate[key] - reference[key]).abs().max().item()
            for key in reference
        )
        print(
            f"Najväčší rozdiel váh: seed 42 vs. {seed}: "
            f"{largest_difference:.6f}"
        )

    for state in STATES:
        assert state.roll is not None

        actions = game.legal_actions(state)
        features = torch.stack(
            [game.action_features(state, action) for action in actions]
        )

        print(
            f"\nJa={state.scores[0]}, súper={state.scores[1]}, "
            f"v ťahu={state.turn_points}, "
            f"hod={game.format_dice(state.roll)}"
        )

        with torch.no_grad():
            for seed, policy in policies.items():
                logits = policy(features)
                probabilities = torch.softmax(logits, dim=0)
                index = logits.argmax().item()
                action = actions[index]

                print(
                    f"  seed {seed}: "
                    f"odlož {game.format_dice(action.keep)} "
                    f"za {action.points}, "
                    f"{game.action_label(state, action)}, "
                    f"p={probabilities[index].item():.1%}"
                )


if __name__ == "__main__":
    main()
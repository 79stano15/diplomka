from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

import compare_policy_sizes as sizes
import dice_rl_critic as critic_experiment
import dice_rl_v2 as game


CHECKPOINTS = {
    "tapered": ((32, 16, 8), Path("results/tapered_10k")),
    "medium": ((32, 32), Path("results/equal_depth_10k")),
}

FIELDNAMES = [
    "seed",
    "match",
    "decision",
    "model",
    "own_score",
    "opponent_score",
    "turn_points",
    "dice_available",
    "roll",
    "legal_actions",
    "filtered_actions",
    "greedy_bank",
    "p_bank",
    "p_best_continue",
    "best_bank_minus_best_continue_logit",
    "greedy_action",
    "best_continue_action",
]


def load_policy(name: str, seed: int) -> sizes.ActionPolicy:
    hidden_sizes, folder = CHECKPOINTS[name]
    path = folder / f"{name}_seed_{seed}.pt"

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["architecture"] != name:
        raise ValueError(f"Nesprávna architektúra: {path}")

    policy = sizes.ActionPolicy(hidden_sizes)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    policy.requires_grad_(False)
    return policy


def filtered_nonwinning_actions(
    state: game.MatchState,
    actions: list[game.Action],
) -> list[game.Action]:
    """
    Volá sa iba v stave bez okamžite výhernej akcie.
    Bankovanie za menej bodov odstráni; pokračovania ponechá.
    """
    max_bank_points = max(
        action.points for action in actions if action.bank
    )

    return [
        action
        for action in actions
        if not action.bank or action.points == max_bank_points
    ]


def evaluate_decision(
    policy: sizes.ActionPolicy,
    state: game.MatchState,
    actions: list[game.Action],
) -> dict[str, object]:
    features = torch.stack(
        [game.action_features(state, action) for action in actions]
    )

    with torch.no_grad():
        logits = policy(features).reshape(-1)
        probabilities = torch.softmax(logits, dim=0)

    greedy_index = int(torch.argmax(logits).item())
    bank_indices = [
        index for index, action in enumerate(actions) if action.bank
    ]
    continue_indices = [
        index for index, action in enumerate(actions) if not action.bank
    ]

    best_bank_index = max(
        bank_indices,
        key=lambda index: float(logits[index]),
    )
    best_continue_index = (
        max(
            continue_indices,
            key=lambda index: float(logits[index]),
        )
        if continue_indices
        else None
    )

    p_bank = sum(
        float(probabilities[index]) for index in bank_indices
    )
    p_best_continue = (
        float(probabilities[best_continue_index])
        if best_continue_index is not None
        else 0.0
    )
    logit_gap = (
        float(logits[best_bank_index] - logits[best_continue_index])
        if best_continue_index is not None
        else None
    )

    return {
        "greedy_bank": actions[greedy_index].bank,
        "p_bank": p_bank,
        "p_best_continue": p_best_continue,
        "best_bank_minus_best_continue_logit": logit_gap,
        "greedy_action": repr(actions[greedy_index]),
        "best_continue_action": (
            repr(actions[best_continue_index])
            if best_continue_index is not None
            else ""
        ),
        "_chosen_action": actions[greedy_index],
    }


def turn_bucket(points: int) -> str:
    if points < 250:
        return "0000-0249"
    if points < 500:
        return "0250-0499"
    if points < 1000:
        return "0500-0999"
    return "1000+"


def print_summary(
    rows: list[dict[str, object]],
    seed: int,
) -> None:
    print(f"\nSEED {seed}: tie isté stavy pre obe politiky")
    print(
        f"{'Model':<8} {'Kocky':>5} {'Body':>10} {'Stavy':>7} "
        f"{'Greedy BANK':>12} {'P(BANK)':>9} "
        f"{'P(best CONT)':>13}"
    )
    print("-" * 75)

    groups: dict[
        tuple[str, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            str(row["model"]),
            str(row["dice_available"]),
            turn_bucket(int(row["turn_points"])),
        )
        groups[key].append(row)

    for (model, dice, bucket), selected in sorted(groups.items()):
        bank_rate = statistics.mean(
            int(row["greedy_bank"]) for row in selected
        )
        p_bank = statistics.mean(
            float(row["p_bank"]) for row in selected
        )
        p_continue = statistics.mean(
            float(row["p_best_continue"]) for row in selected
        )

        print(
            f"{model:<8} {dice:>5} {bucket:>10} "
            f"{len(selected):>7} {bank_rate:>11.1%} "
            f"{p_bank:>9.1%} {p_continue:>13.1%}"
        )

    print("\nCelkom na navštívených nevýherných stavoch:")

    for model in CHECKPOINTS:
        selected = [row for row in rows if row["model"] == model]
        bank_rate = statistics.mean(
            int(row["greedy_bank"]) for row in selected
        )
        p_bank = statistics.mean(
            float(row["p_bank"]) for row in selected
        )

        print(
            f"  {model:<8}: {len(selected)} stavov, "
            f"greedy BANK {bank_rate:.1%}, "
            f"priemerné P(BANK) {p_bank:.1%}"
        )


def collect_seed(
    seed: int,
    matches: int,
    policies: dict[str, sizes.ActionPolicy],
    reference: game.PolicyNetwork,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for match_index in range(matches):
        dice_rng = random.Random(800_000 + match_index)
        state = game.MatchState(
            scores=[0, 0],
            player=match_index % 2,
        )
        decision_number = 0

        for _ in range(300):
            values = [
                dice_rng.randint(1, 6)
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
                decision_number += 1
                current_total = state.scores[0] + state.turn_points

                winning_actions = [
                    action
                    for action in actions
                    if current_total + action.points >= game.GOAL
                ]

                if winning_actions:
                    # Zdrojová medium politika berie výhru.
                    filtered = winning_actions
                else:
                    filtered = filtered_nonwinning_actions(
                        state,
                        actions,
                    )

                decisions = {
                    name: evaluate_decision(policy, state, filtered)
                    for name, policy in policies.items()
                }

                if not winning_actions:
                    for name, result in decisions.items():
                        rows.append(
                            {
                                "seed": seed,
                                "match": match_index,
                                "decision": decision_number,
                                "model": name,
                                "own_score": state.scores[0],
                                "opponent_score": state.scores[1],
                                "turn_points": state.turn_points,
                                "dice_available": state.dice_available,
                                "roll": repr(state.roll),
                                "legal_actions": len(actions),
                                "filtered_actions": len(filtered),
                                "greedy_bank": int(
                                    result["greedy_bank"]
                                ),
                                "p_bank": result["p_bank"],
                                "p_best_continue": result[
                                    "p_best_continue"
                                ],
                                "best_bank_minus_best_continue_logit":
                                    result[
                                        "best_bank_minus_best_continue_logit"
                                    ],
                                "greedy_action": result[
                                    "greedy_action"
                                ],
                                "best_continue_action": result[
                                    "best_continue_action"
                                ],
                            }
                        )

                # Stavy zbiera výlučne medium. Tapered ich hodnotí,
                # ale jeho rozhodnutie nemení priebeh zápasu.
                action = decisions["medium"]["_chosen_action"]
            else:
                features = torch.stack(
                    [
                        game.action_features(state, action)
                        for action in actions
                    ]
                )

                with torch.no_grad():
                    index = int(
                        torch.argmax(reference(features)).item()
                    )

                # Referenčný súper zostáva bez filtra.
                action = actions[index]

            outcome = game.apply_action(state, action)

            if outcome == "win":
                break
        else:
            raise RuntimeError(
                f"Zápas {match_index} prekročil 300 hodov."
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--matches", type=int, default=400)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tapered_decisions"),
    )
    args = parser.parse_args()

    if args.matches < 2 or args.matches % 2:
        parser.error("--matches musí byť kladné párne číslo")

    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds musia byť rôzne")

    output = args.output_dir / "shared_states.csv"

    if output.exists():
        parser.error(
            f"Súbor už existuje: {output}. "
            "Použi nový --output-dir."
        )

    game.test_rules()
    torch.set_num_threads(1)

    reference = critic_experiment.load_policy(
        Path("dice_policy_v2.pt")
    )
    reference.eval()
    reference.requires_grad_(False)

    all_rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        policies = {
            name: load_policy(name, seed)
            for name in CHECKPOINTS
        }
        rows = collect_seed(
            seed,
            args.matches,
            policies,
            reference,
        )
        print_summary(rows, seed)
        all_rows.extend(rows)

    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nPodrobné rozhodnutia: {output}")
    print(
        "Poznámka: riadky sú rozhodnutia v zápasoch, "
        "nie nezávislé tréningové seedy."
    )


if __name__ == "__main__":
    main()
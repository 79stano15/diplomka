from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import dice_rl_critic as critic_experiment
import dice_rl_v2 as game
from compare_policy_sizes import ActionPolicy


MODELS: dict[str, tuple[tuple[int, ...], Path]] = {
    "bottleneck4": (
        (32, 4, 4),
        Path("results/bottleneck4_10k"),
    ),
    "tapered": (
    (32, 16, 8),
    Path("results/tapered_10k"),
    ),
    "medium": (
        (32, 32),
        Path("results/equal_depth_10k"),
    ),
    "large": (
        (64, 64),
        Path("results/policy_sizes_10k"),
    ),
}


@dataclass
class MatchResult:
    won: bool
    decisions: int
    nonwinning_decisions: int
    banked_without_win: int
    bank_probability_sum: float
    suboptimal_bank_points: int
    winning_opportunities: int
    missed_immediate_wins: int


def load_candidate(name: str, seed: int) -> ActionPolicy:
    hidden_sizes, folder = MODELS[name]
    path = folder / f"{name}_seed_{seed}.pt"

    if not path.is_file():
        raise FileNotFoundError(f"Chýba checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["architecture"] != name:
        raise ValueError(f"Nesprávna architektúra v {path}")

    policy = ActionPolicy(hidden_sizes)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    policy.requires_grad_(False)

    return policy

def filter_strategic_actions(state: game.MatchState, actions: list[game.Action]) -> list[game.Action]:
    """
    Odfiltruje zjavne nevýhodné akcie:
    1. Ak existuje výherná akcia, ponechá len výherné.
    2. Ak sa bankuje, ponechá len bank s maximálnym počtom bodov.
    """
    if not actions:
        return []

    # 1. Okamžitá výhra
    current_total = state.scores[state.player] + state.turn_points
    winning_actions = [
        a for a in actions 
        if a.bank and (current_total + a.points >= game.GOAL)
    ]
    if winning_actions:
        return winning_actions

    # 2. Dominované bankovanie
    # Nájdeme max body medzi všetkými akciami, ktoré majú .bank == True
    max_bank_points = max((a.points for a in actions if a.bank), default=-1)
    
    filtered = []
    for a in actions:
        if a.bank:
            if a.points == max_bank_points:
                filtered.append(a)
        else:
            # Akcie 'continue' (pokračovať v hode) nechávame všetky
            filtered.append(a)
            
    return filtered


def choose_action(
    policy: nn.Module,
    state: game.MatchState,
    actions: list[game.Action],
    mode: str,
    rng: random.Random,
    use_filter: bool = True,  # Pridaný prepínač
) -> tuple[game.Action, float]:
    
    # --- NOVÁ LOGIKA FILTRA ---
    if use_filter:
        actions = filter_strategic_actions(state, actions)
    # ---------------------------

    features = torch.stack(
        [game.action_features(state, action) for action in actions]
    )

    with torch.no_grad():
        logits = policy(features)
        probabilities = torch.softmax(logits, dim=0)

    if mode == "sampled":
        index = rng.choices(
            range(len(actions)),
            weights=probabilities.tolist(),
            k=1,
        )[0]
    else:
        index = int(torch.argmax(logits).item())

    bank_probability = sum(
        float(probabilities[i])
        for i, action in enumerate(actions)
        if action.bank
    )

    return actions[index], bank_probability


def play_one(
    candidate: ActionPolicy,
    reference: game.PolicyNetwork,
    dice_seed: int,
    first_player: int,
    mode: str,
    use_filter: bool = True,
    max_rolls: int = 300,
) -> MatchResult:
    # Náhodnosť kociek a výberu akcií používajú oddelené generátory.
    dice_rng = random.Random(dice_seed)
    action_rng = random.Random(dice_seed + 1_000_003)

    state = game.MatchState(scores=[0, 0], player=first_player)

    decisions = 0
    nonwinning_decisions = 0
    banked_without_win = 0
    bank_probability_sum = 0.0
    suboptimal_bank_points = 0
    winning_opportunities = 0
    missed_immediate_wins = 0

    for _ in range(max_rolls):
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
            action, bank_probability = choose_action(
                candidate,
                state,
                actions,
                mode,
                action_rng,
                use_filter=use_filter,
            )
            decisions += 1

            current_total = state.scores[0] + state.turn_points
            can_win = any(
                current_total + option.points >= game.GOAL
                for option in actions
            )
            chosen_wins = (
                current_total + action.points >= game.GOAL
            )

            if can_win:
                winning_opportunities += 1
                missed_immediate_wins += int(not chosen_wins)
            else:
                nonwinning_decisions += 1
                bank_probability_sum += bank_probability
                banked_without_win += int(action.bank)

                max_points = max(
                    option.points
                    for option in actions
                    if option.bank
                )
                suboptimal_bank_points += int(
                    action.bank and action.points < max_points
                )
        else:
            action, _ = choose_action(
                reference,
                state,
                actions,
                "greedy",
                action_rng,
                use_filter=use_filter,
            )

        outcome = game.apply_action(state, action)

        if outcome == "win":
            return MatchResult(
                won=state.winner == 0,
                decisions=decisions,
                nonwinning_decisions=nonwinning_decisions,
                banked_without_win=banked_without_win,
                bank_probability_sum=bank_probability_sum,
                suboptimal_bank_points=suboptimal_bank_points,
                winning_opportunities=winning_opportunities,
                missed_immediate_wins=missed_immediate_wins,
            )

    raise RuntimeError("Zápas prekročil limit hodov.")


def evaluate(
    candidate: ActionPolicy,
    reference: game.PolicyNetwork,
    matches: int,
    mode: str,
    use_filter: bool = True,
) -> str:
    results = [
        play_one(
            candidate=candidate,
            reference=reference,
            dice_seed=800_000 + index,
            first_player=index % 2,
            mode=mode,
            use_filter=use_filter,
        )
        for index in range(matches)
    ]

    wins = sum(result.won for result in results)
    nonwinning = sum(
        result.nonwinning_decisions for result in results
    )
    banked = sum(
        result.banked_without_win for result in results
    )
    bank_mass = sum(
        result.bank_probability_sum for result in results
    )
    worse_bank = sum(
        result.suboptimal_bank_points for result in results
    )
    opportunities = sum(
        result.winning_opportunities for result in results
    )
    missed = sum(
        result.missed_immediate_wins for result in results
    )

    if nonwinning == 0:
        raise RuntimeError("Nenašli sa žiadne nevýherné rozhodnutia.")

    line = (
        f"  {mode:<7} | výhry {wins / matches:6.1%} | "
        f"BANK {banked / nonwinning:6.1%} | "
        f"P(BANK) {bank_mass / nonwinning:6.1%} | "
        f"bankovanie za menej bodov "
        f"{worse_bank / nonwinning:6.1%} | "
        f"prehliadnutá výhra {missed}/{opportunities}"
    )
    return line

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=["bottleneck4", "medium"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--matches", type=int, default=400)
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Vypne redukciu nevýhodných akcí pred výberom politiky.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sampling_vs_greedy.txt"),
        help="Cesta k súboru, kam sa uloží textový výstup.",
    )
    args = parser.parse_args()

    if args.matches < 2 or args.matches % 2:
        parser.error("--matches musí byť kladné párne číslo")

    game.test_rules()
    torch.set_num_threads(1)

    reference = critic_experiment.load_policy(
        Path("dice_policy_v2.pt")
    )
    reference.eval()
    reference.requires_grad_(False)

    use_filter = not args.no_filter
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        "BANK: skutočný podiel bankovania bez okamžitej výhry\n"
        "P(BANK): priemerná pravdepodobnosť bankovania "
        "priradená politikou v navštívených stavoch\n"
        "Súper v2 vždy používa deterministický výber.\n"
        f"Strategický filter: {'ON' if use_filter else 'OFF'}\n"
    )

    with args.output.open("w", encoding="utf-8") as fh:
        for name in args.models:
            for seed in args.seeds:
                candidate = load_candidate(name, seed)
                print(f"{name}, seed {seed}:")
                fh.write(f"{name}, seed {seed}:\n")

                for mode in ("greedy", "sampled"):
                    result_line = evaluate(
                        candidate,
                        reference,
                        args.matches,
                        mode,
                        use_filter=use_filter,
                    )
                    print(result_line)
                    fh.write(f"{result_line}\n")

                    fh.flush()


if __name__ == "__main__":
    main()
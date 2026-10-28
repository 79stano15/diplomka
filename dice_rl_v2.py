from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from torch import nn
from torch.distributions import Categorical


Counts: TypeAlias = tuple[int, int, int, int, int, int]

GOAL = 4_000
FULL_STRAIGHT: Counts = (1, 1, 1, 1, 1, 1)
SMALL_STRAIGHT: Counts = (0, 1, 1, 1, 1, 1)
TRAIN_THRESHOLDS = (250, 500, 800)


@dataclass(frozen=True)
class Action:
    keep: Counts
    points: int
    bank: bool


@dataclass
class MatchState:
    scores: list[int]
    player: int = 0
    turn_points: int = 0
    dice_available: int = 6
    roll: Counts | None = None
    winner: int | None = None


def count_dice(values: list[int]) -> Counts:
    counts = [0] * 6

    for value in values:
        if value < 1 or value > 6:
            raise ValueError(f"Neplatná hodnota kocky: {value}")
        counts[value - 1] += 1

    return (
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        counts[5],
    )


def format_dice(counts: Counts) -> str:
    return "".join(
        str(face) * amount
        for face, amount in enumerate(counts, start=1)
    )


def score_selection(keep: Counts) -> int:
    """Body za skutočne odložené kocky; nula znamená nelegálny výber."""

    if keep == FULL_STRAIGHT:
        return 1_500

    if keep == SMALL_STRAIGHT:
        return 750

    points = 0

    for face, amount in enumerate(keep, start=1):
        if amount == 0:
            continue

        if amount >= 3:
            base = 1_000 if face == 1 else 100 * face
            points += base * (2 ** (amount - 3))
        elif face == 1:
            points += 100 * amount
        elif face == 5:
            points += 50 * amount
        else:
            return 0

    return points


def all_submultisets(roll: Counts) -> list[Counts]:
    result: list[Counts] = []

    def generate(index: int, current: list[int]) -> None:
        if index == 6:
            if any(current):
                result.append(
                    (
                        current[0],
                        current[1],
                        current[2],
                        current[3],
                        current[4],
                        current[5],
                    )
                )
            return

        for amount in range(roll[index] + 1):
            current.append(amount)
            generate(index + 1, current)
            current.pop()

    generate(0, [])
    return result


def legal_actions(state: MatchState, goal: int = GOAL) -> list[Action]:
    if state.roll is None:
        return []

    actions: list[Action] = []

    for keep in all_submultisets(state.roll):
        points = score_selection(keep)

        if points == 0:
            continue

        wins = (
            state.scores[state.player] + state.turn_points + points >= goal
        )

        actions.append(Action(keep=keep, points=points, bank=True))

        if not wins:
            actions.append(Action(keep=keep, points=points, bank=False))

    return actions


def action_features(
    state: MatchState,
    action: Action,
    goal: int = GOAL,
) -> torch.Tensor:
    assert state.roll is not None

    own = state.scores[state.player]
    opponent = state.scores[1 - state.player]
    kept = sum(action.keep)
    remaining = state.dice_available - kept
    next_dice = 6 if remaining == 0 else remaining

    features = [
        own / goal,
        opponent / goal,
        (own - opponent) / goal,
        state.turn_points / goal,
        state.dice_available / 6,
        action.points / goal,
        (state.turn_points + action.points) / goal,
        kept / 6,
        next_dice / 6,
        float(action.bank),
    ]

    features.extend(amount / 6 for amount in state.roll)
    features.extend(amount / 6 for amount in action.keep)

    return torch.tensor(features, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    """Ohodnotí jednu dvojicu stav–akcia jedným logitom."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(22, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, action_batch: torch.Tensor) -> torch.Tensor:
        return self.network(action_batch).squeeze(-1)


def choose_neural_action(
    policy: PolicyNetwork,
    state: MatchState,
    training: bool,
) -> tuple[Action, torch.Tensor | None, torch.Tensor | None]:
    actions = legal_actions(state)
    batch = torch.stack(
        [action_features(state, action) for action in actions]
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


def choose_heuristic_action(
    state: MatchState,
    rng: random.Random,
    bank_threshold: int,
) -> Action:
    actions = legal_actions(state)
    assert actions

    best = max(
        actions,
        key=lambda action: (
            action.points,
            6
            if sum(action.keep) == state.dice_available
            else state.dice_available - sum(action.keep),
        ),
    )

    should_bank = (
        state.turn_points + best.points >= bank_threshold
        or state.scores[state.player]
        + state.turn_points
        + best.points
        >= GOAL
    )

    candidates = [
        action
        for action in actions
        if action.keep == best.keep and action.bank == should_bank
    ]

    return rng.choice(candidates)


def apply_action(
    state: MatchState,
    action: Action,
    goal: int = GOAL,
) -> str:
    """Zmení stav a vráti 'win', 'bank' alebo 'continue'."""

    state.turn_points += action.points
    own_score = state.scores[state.player]

    if own_score + state.turn_points >= goal:
        state.scores[state.player] += state.turn_points
        state.winner = state.player
        state.roll = None
        return "win"

    if action.bank:
        state.scores[state.player] += state.turn_points
        state.turn_points = 0
        state.dice_available = 6
        state.player = 1 - state.player
        state.roll = None
        return "bank"

    remaining = state.dice_available - sum(action.keep)
    state.dice_available = 6 if remaining == 0 else remaining
    state.roll = None
    return "continue"


def describe_action(action: Action, wins: bool) -> str:
    decision = "VÝHRA" if wins else ("BANK" if action.bank else "ZNOVA")

    return (
        f"odloží {format_dice(action.keep)} "
        f"za {action.points} → {decision}"
    )

def choose_checkpoint_action(
    opponent: PolicyNetwork,
    state: MatchState,
    sample: bool = True,
) -> Action:
    """Vyberie akciu súpera bez vytvárania tréningového grafu."""
    actions = legal_actions(state)
    batch = torch.stack(
        [action_features(state, action) for action in actions]
    )

    with torch.no_grad():
        logits = opponent(batch)

        if sample:
            index = Categorical(logits=logits).sample().item()
        else:
            index = torch.argmax(logits).item()

    return actions[index]


def play_match(
    policy: PolicyNetwork,
    rng: random.Random,
    opponent_threshold: int | None,
    training: bool,
    verbose: bool = False,
    first_player: int | None = None,
    max_rolls: int = 300,
    opponent_policy: PolicyNetwork | None = None,
    opponent_sample: bool = True,
) -> tuple[
    int,
    list[torch.Tensor],
    list[torch.Tensor],
    int,
]:
    if first_player is None:
        first_player = rng.randint(0, 1)

    state = MatchState(scores=[0, 0], player=first_player)
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for roll_number in range(1, max_rolls + 1):
        values = [
            rng.randint(1, 6)
            for _ in range(state.dice_available)
        ]
        state.roll = count_dice(values)

        if verbose:
            print(
                f"\nHod {roll_number}: hráč {state.player}, "
                f"skóre {state.scores}, priebežne "
                f"{state.turn_points}, kocky "
                f"{format_dice(state.roll)}"
            )

        actions = legal_actions(state)

        if not actions:
            if verbose:
                print("  FARKLE: body v tomto ťahu prepadli.")

            state.turn_points = 0
            state.dice_available = 6
            state.player = 1 - state.player
            state.roll = None
            continue

        if state.player == 0:
            action, log_prob, entropy = choose_neural_action(
                policy,
                state,
                training=training,
            )

            if log_prob is not None and entropy is not None:
                log_probs.append(log_prob)
                entropies.append(entropy)
        else:
            if opponent_policy is not None:
                action = choose_checkpoint_action(
                    opponent_policy,
                    state,
                    sample=opponent_sample,
                )
            elif opponent_threshold is not None:
                action = choose_heuristic_action(
                    state,
                    rng,
                    bank_threshold=opponent_threshold,
                )
            else:
                raise ValueError("Chýba model alebo prah súpera.")

        wins = (
            state.scores[state.player]
            + state.turn_points
            + action.points
            >= GOAL
        )

        if verbose:
            print(f"  {describe_action(action, wins)}")

        outcome = apply_action(state, action)

        if verbose:
            print(f"  Skóre: {state.scores}")
            print(f"  Body v aktuálnom ťahu: {state.turn_points}")

        if outcome == "win":
            if verbose:
                print(f"  KONIEC: vyhral hráč {state.winner}.")

            return state.winner, log_probs, entropies, roll_number

    raise RuntimeError(
        "Zápas prekročil limit hodov. "
        "Skontroluj pravidlá alebo zvýš max_rolls."
    )


def train(
    episodes: int = 10_000,
    seed: int = 42,
    entropy_coefficient: float = 0.01,
    baseline_rate: float = 0.02,
) -> PolicyNetwork:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy = PolicyNetwork()
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    anchor_path = Path("dice_policy_v2.pt")

    if not anchor_path.exists():
        raise FileNotFoundError(
            "Chýba dice_policy_v2.pt. "
            "Skopíruj ho do priečinka so skriptom."
        )

    anchor = PolicyNetwork()
    anchor.load_state_dict(
        torch.load(
            anchor_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    anchor.eval()
    anchor.requires_grad_(False)

    opponent_pool: list[PolicyNetwork] = [anchor]
    snapshot_interval = 1_000
    max_opponents = 5
    baseline = 0.0

    recent_wins: list[int] = []
    recent_rolls: list[int] = []
    recent_entropies: list[float] = []

    for episode in range(1, episodes + 1):
        opponent = rng.choice(opponent_pool)

        winner, log_probs, entropies, rolls = play_match(
            policy,
            rng,
            opponent_threshold=None,
            training=True,
            opponent_policy=opponent,
        )

        reward = 1.0 if winner == 0 else -1.0

        # Baseline sa aktualizuje AŽ po gradiente tejto epizódy.
        # Nezávisí teda od aktuálne vybratých akcií.
        advantage = reward - baseline

        if log_probs:
            log_probability_sum = torch.stack(log_probs).sum()
            entropy_sum = torch.stack(entropies).sum()

            loss = (
                -advantage * log_probability_sum
                - entropy_coefficient * entropy_sum
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            mean_entropy = torch.stack(entropies).mean().item()
            recent_entropies.append(mean_entropy)

        baseline = (
            (1.0 - baseline_rate) * baseline
            + baseline_rate * reward
        )

        if episode % snapshot_interval == 0:
            snapshot = copy.deepcopy(policy)
            snapshot.eval()
            snapshot.requires_grad_(False)
            opponent_pool.append(snapshot)

            # Prvého, pôvodného súpera ponecháme ako stabilný referenčný bod.
            # Zo starších nových snapshotov odstránime najstarší.
            if len(opponent_pool) > max_opponents:
                opponent_pool.pop(1)

        recent_wins.append(int(winner == 0))
        recent_rolls.append(rolls)

        if episode % 200 == 0 or episode == episodes:
            window = min(200, len(recent_wins))
            win_rate = sum(recent_wins[-window:]) / window
            avg_rolls = sum(recent_rolls[-window:]) / window

            entropy_window = recent_entropies[-window:]
            avg_entropy = (
                sum(entropy_window) / len(entropy_window)
                if entropy_window
                else 0.0
            )

            print(
                f"Epizóda {episode:5d} | "
                f"výhry: {win_rate:6.1%} | "
                f"hody: {avg_rolls:5.1f} | "
                f"entropia: {avg_entropy:6.3f}"
            )

    return policy


def wilson_interval(wins: int, matches: int) -> tuple[float, float]:
    """Orientačný 95 % Wilsonov interval pre podiel výhier."""

    z = 1.96
    proportion = wins / matches
    denominator = 1.0 + z * z / matches

    center = (
        proportion + z * z / (2.0 * matches)
    ) / denominator

    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / matches
            + z * z / (4.0 * matches * matches)
        )
        / denominator
    )

    return center - radius, center + radius


def evaluate(
    policy: PolicyNetwork,
    label: str,
    matches: int = 400,
    seed: int = 12_345,
) -> None:
    if matches < 2 or matches % 2 != 0:
        raise ValueError("Počet zápasov musí byť kladný párny počet.")

    policy.eval()
    print(f"\nVyhodnotenie: {label}")

    with torch.no_grad():
        for threshold_index, threshold in enumerate(TRAIN_THRESHOLDS):
            wins = 0
            wins_starting = 0
            wins_second = 0

            for match_index in range(matches):
                first_player = match_index % 2
                match_seed = (
                    seed
                    + 100_000 * threshold_index
                    + match_index
                )

                winner, _, _, _ = play_match(
                    policy,
                    random.Random(match_seed),
                    opponent_threshold=threshold,
                    training=False,
                    first_player=first_player,
                )

                won = int(winner == 0)
                wins += won

                if first_player == 0:
                    wins_starting += won
                else:
                    wins_second += won

            lower, upper = wilson_interval(wins, matches)
            half = matches // 2

            print(
                f"  Prah {threshold:3d}: "
                f"{wins}/{matches} = {wins / matches:.1%} "
                f"(95 % interval {lower:.1%}–{upper:.1%}); "
                f"začína {wins_starting}/{half}, "
                f"hrá druhý {wins_second}/{half}"
            )


def action_label(
    state: MatchState,
    action: Action,
    goal: int = GOAL,
) -> str:
    own = state.scores[state.player]

    if own + state.turn_points + action.points >= goal:
        return "VÝHRA"

    return "BANK" if action.bank else "POKRAČUJ"


def inspect_decision(
    policy: PolicyNetwork,
    state: MatchState,
    goal: int = GOAL,
) -> None:
    actions = legal_actions(state, goal=goal)

    if not actions:
        print("FARKLE: neexistuje bodujúca voľba.")
        return

    batch = torch.stack(
        [action_features(state, action, goal) for action in actions]
    )

    policy.eval()

    with torch.no_grad():
        logits = policy(batch)
        probabilities = torch.softmax(logits, dim=0)

    assert state.roll is not None

    print(
        f"\nSkóre: ja={state.scores[state.player]}, "
        f"súper={state.scores[1 - state.player]}, "
        f"v ťahu={state.turn_points}"
    )
    print(
        f"Hod: {format_dice(state.roll)} "
        f"({state.dice_available} kociek)"
    )

    rows = sorted(
        zip(actions, logits.tolist(), probabilities.tolist()),
        key=lambda row: row[2],
        reverse=True,
    )

    for action, logit, probability in rows:
        decision = action_label(state, action, goal)

        if decision == "POKRAČUJ":
            remaining = state.dice_available - sum(action.keep)
            next_dice = 6 if remaining == 0 else remaining
            dice_info = str(next_dice)
        else:
            dice_info = "–"

        print(
            f"{format_dice(action.keep):>6} | "
            f"{action.points:>4} bodov | "
            f"{decision:<8} | "
            f"ďalej {dice_info:>2} | "
            f"logit {logit:>8.3f} | "
            f"p {probability:.3e}"
        )

    print(
        "p znamená pravdepodobnosť VÝBERU akcie politikou, "
        "nie pravdepodobnosť výhry."
    )


def inspect_score_context(policy: PolicyNetwork) -> None:
    scenarios = [
        ("vyrovnaný zápas", 2_800, 2_800, 350),
        ("súper vedie", 2_800, 3_800, 350),
        ("my vedieme", 3_800, 2_800, 350),
        ("blízko výhry", 3_800, 3_800, 100),
    ]

    roll = count_dice([2, 2, 2, 1, 5, 3])
    policy.eval()

    print("\nRovnaký hod v rôznych situáciách:")

    with torch.no_grad():
        for name, own, opponent, turn_points in scenarios:
            state = MatchState(
                scores=[own, opponent],
                player=0,
                turn_points=turn_points,
                dice_available=6,
                roll=roll,
            )

            actions = legal_actions(state)
            batch = torch.stack(
                [action_features(state, action) for action in actions]
            )

            logits = policy(batch)
            probabilities = torch.softmax(logits, dim=0)
            index = torch.argmax(logits).item()
            best = actions[index]

            print(
                f"  {name:>18}: "
                f"ja={own}, súper={opponent}, ťah={turn_points} "
                f"→ {format_dice(best.keep)} za {best.points}, "
                f"{action_label(state, best)}, "
                f"p={probabilities[index].item():.3e}"
            )


def test_rules() -> None:
    assert score_selection(count_dice([1])) == 100
    assert score_selection(count_dice([1, 1])) == 200
    assert score_selection(count_dice([1, 1, 1, 1])) == 2_000
    assert score_selection(count_dice([1] * 5)) == 4_000
    assert score_selection(count_dice([5] * 5)) == 2_000
    assert score_selection(count_dice([1, 2, 3, 4, 5, 6])) == 1_500
    assert score_selection(count_dice([2, 3, 4, 5, 6])) == 750
    assert score_selection(count_dice([2, 2, 2, 1, 5, 5])) == 400
    assert score_selection(count_dice([2, 3, 3, 4, 4, 6])) == 0

    state = MatchState(
        scores=[0, 0],
        dice_available=4,
        roll=count_dice([1, 1, 1, 5]),
    )
    keeps = {action.keep for action in legal_actions(state)}

    assert count_dice([1]) in keeps
    assert count_dice([1, 1]) in keeps
    assert count_dice([1, 1, 1]) in keeps
    assert count_dice([5]) in keeps
    assert count_dice([1, 1, 1, 5]) in keeps

    straight_state = MatchState(
        scores=[0, 0],
        dice_available=6,
        roll=count_dice([1, 2, 3, 4, 5, 6]),
    )
    straight_keeps = {
        action.keep for action in legal_actions(straight_state)
    }

    assert FULL_STRAIGHT in straight_keeps
    assert SMALL_STRAIGHT in straight_keeps

    hot_state = MatchState(
        scores=[0, 0],
        dice_available=5,
        roll=count_dice([2, 2, 2, 1, 5]),
    )
    hot_action = Action(
        keep=count_dice([2, 2, 2, 1, 5]),
        points=350,
        bank=False,
    )

    assert hot_action in legal_actions(hot_state)
    assert apply_action(hot_state, hot_action) == "continue"
    assert hot_state.dice_available == 6
    assert hot_state.turn_points == 350

    win_state = MatchState(
        scores=[3_900, 3_000],
        dice_available=6,
        roll=count_dice([1, 2, 3, 4, 5, 6]),
    )
    winning_actions = [
        action
        for action in legal_actions(win_state)
        if action.keep == count_dice([1])
    ]

    assert len(winning_actions) == 1
    assert apply_action(win_state, winning_actions[0]) == "win"
    assert win_state.winner == 0
    assert win_state.scores[0] == 4_000

    bank_state = MatchState(
        scores=[500, 600],
        turn_points=250,
        dice_available=4,
        roll=count_dice([1, 2, 3, 4]),
    )

    assert (
        apply_action(
            bank_state,
            Action(count_dice([1]), 100, bank=True),
        )
        == "bank"
    )
    assert bank_state.scores == [850, 600]
    assert bank_state.player == 1
    assert bank_state.dice_available == 6

def get_filtered_actions(state, legal_actions):
    """
    Aplikuje strategické maskovanie na legálne akcie:
    1. Ak existuje okamžitá výhra, vráť iba výherné akcie.
    2. Ak sa bankuje, vráť iba tú akciu, ktorá bankuje maximum bodov.
    """
    if not legal_actions:
        return []

    # 1. Kontrola okamžitej výhry
    winning_actions = [
        a for a in legal_actions 
        if a.is_bank and (state.scores[state.current_player] + state.turn_points + a.points >= 4000)
    ]
    if winning_actions:
        return winning_actions

    # 2. Odstránenie dominovaných bankovacích akcií
    # Nájdeme maximálne body, ktoré sa dajú v tomto hode bankovať
    max_bank_points = max((a.points for a in legal_actions if a.is_bank), default=-1)
    
    filtered = []
    for a in legal_actions:
        if a.is_bank:
            # Ponecháme iba ten bank, ktorý prináša maximum
            if a.points == max_bank_points:
                filtered.append(a)
        else:
            # Všetky akcie "pokračovať" (continue) ponechávame na rozhodnutie siete
            filtered.append(a)
            
    return filtered

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-matches", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes musí byť aspoň 1")

    if args.eval_matches < 2 or args.eval_matches % 2 != 0:
        parser.error("--eval-matches musí byť kladné párne číslo")

    test_rules()
    print("Testy pravidiel prešli.")

    policy = train(episodes=args.episodes, seed=args.seed)

    new_path = Path("dice_policy_selfplay.pt")
    torch.save(policy.state_dict(), new_path)
    print(f"\nNový model uložený do {new_path}")

    evaluate(
        policy,
        label="nový model v2",
        matches=args.eval_matches,
    )

    old_path = Path("dice_policy_v2.pt")

    if old_path.exists():
        old_policy = PolicyNetwork()
        old_policy.load_state_dict(
            torch.load(
                old_path,
                map_location="cpu",
                weights_only=True,
            )
        )

        evaluate(
            old_policy,
            label="pôvodný checkpoint",
            matches=args.eval_matches,
        )
    else:
        print("\nPôvodný dice_policy_v2.pt sa nenašiel; porovnanie preskočené.")

    example_state = MatchState(
        scores=[2_800, 3_200],
        player=0,
        turn_points=350,
        dice_available=6,
        roll=count_dice([2, 2, 2, 1, 5, 3]),
    )

    # print("\nRozhodnutia nového modelu:")
    # inspect_decision(policy, example_state)
    # inspect_score_context(policy)

    
    # print("\nUkážkový zápas nového modelu:")
    # with torch.no_grad():
    #     play_match(
    #         policy,
    #         random.Random(2026),
    #         opponent_threshold=500,
    #         training=False,
    #         verbose=True,
    #     ) 


if __name__ == "__main__":
    main()
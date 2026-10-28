from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, TypeAlias

import torch
from torch import nn
from torch.distributions import Categorical


Counts: TypeAlias = tuple[int, int, int, int, int, int]

GOAL = 4_000
FULL_STRAIGHT: Counts = (1, 1, 1, 1, 1, 1)
SMALL_STRAIGHT: Counts = (0, 1, 1, 1, 1, 1)


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
    return tuple(values.count(face) for face in range(1, 7))  # type: ignore[return-value]


def format_dice(counts: Counts) -> str:
    return "".join(
        str(face) * amount
        for face, amount in enumerate(counts, start=1)
    )


def score_selection(keep: Counts) -> int:
    """Vráti body, alebo 0, ak výber nie je bodujúci."""

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

    def generate(face_index: int, current: list[int]) -> None:
        if face_index == 6:
            if any(current):
                result.append(tuple(current))  # type: ignore[arg-type]
            return

        for amount in range(roll[face_index] + 1):
            current.append(amount)
            generate(face_index + 1, current)
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

        wins_immediately = (
            state.scores[state.player] + state.turn_points + points >= goal
        )

        if wins_immediately:
            actions.append(Action(keep, points, bank=True))
        else:
            actions.append(Action(keep, points, bank=True))
            actions.append(Action(keep, points, bank=False))

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

    features.extend(count / 6 for count in state.roll)
    features.extend(count / 6 for count in action.keep)

    return torch.tensor(features, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    """Rovnaká sieť ohodnotí každú legálnu akciu v danom stave."""

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
    rng: random.Random,
    training: bool,
) -> tuple[Action, torch.Tensor | None]:
    del rng

    actions = legal_actions(state)
    batch = torch.stack([action_features(state, action) for action in actions])
    logits = policy(batch)

    if training:
        distribution = Categorical(logits=logits)
        index = distribution.sample()
        return actions[index.item()], distribution.log_prob(index)

    index = torch.argmax(logits).item()
    return actions[index], None


def choose_heuristic_action(
    state: MatchState,
    rng: random.Random,
    bank_threshold: int,
) -> Action:
    actions = legal_actions(state)
    assert actions

    best_keep = max(
        actions,
        key=lambda action: (
            action.points,
            6 if state.dice_available == sum(action.keep) else
            state.dice_available - sum(action.keep),
        ),
    )

    matching = [
        action
        for action in actions
        if action.keep == best_keep.keep
    ]

    should_bank = (
        state.turn_points + best_keep.points >= bank_threshold
        or state.scores[state.player]
        + state.turn_points
        + best_keep.points >= GOAL
    )

    candidates = [
        action
        for action in matching
        if action.bank == should_bank
    ]

    if not candidates:
        candidates = matching

    return rng.choice(candidates)


def describe_action(action: Action, wins: bool) -> str:
    decision = "VÝHRA" if wins else ("BANK" if action.bank else "ZNOVA")
    return (
        f"odloží {format_dice(action.keep)} "
        f"za {action.points} → {decision}"
    )


def play_match(
    policy: PolicyNetwork,
    rng: random.Random,
    opponent_threshold: int,
    training: bool,
    verbose: bool = False,
    max_rolls: int = 300,
) -> tuple[int, list[torch.Tensor], int]:
    state = MatchState(scores=[0, 0], player=rng.randint(0, 1))
    log_probs: list[torch.Tensor] = []

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
                print("  FARKLE: body z ťahu prepadli.")

            state.turn_points = 0
            state.dice_available = 6
            state.player = 1 - state.player
            continue

        if state.player == 0:
            action, log_prob = choose_neural_action(
                policy,
                state,
                rng,
                training=training,
            )
            if log_prob is not None:
                log_probs.append(log_prob)
        else:
            action = choose_heuristic_action(
                state,
                rng,
                bank_threshold=opponent_threshold,
            )

        state.turn_points += action.points
        wins = state.scores[state.player] + state.turn_points >= GOAL

        if verbose:
            print(f"  {describe_action(action, wins)}")
            print(f"  Body v ťahu: {state.turn_points}")

        if wins:
            state.winner = state.player
            state.scores[state.player] += state.turn_points

            if verbose:
                print(f"  KONIEC: vyhral hráč {state.winner}.")

            return state.winner, log_probs, roll_number

        if action.bank:
            state.scores[state.player] += state.turn_points

            if verbose:
                print(f"  Zapísané skóre: {state.scores}")

            state.turn_points = 0
            state.dice_available = 6
            state.player = 1 - state.player
            continue

        remaining = state.dice_available - sum(action.keep)
        state.dice_available = 6 if remaining == 0 else remaining

    raise RuntimeError(
        "Zápas prekročil bezpečnostný limit hodov. "
        "Skontroluj pravidlá alebo zvýš max_rolls."
    )


def train(
    episodes: int = 10_000,
    seed: int = 42,
) -> PolicyNetwork:
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    thresholds = [250, 500, 800]
    recent_wins: list[int] = []
    recent_rolls: list[int] = []

    for episode in range(1, episodes + 1):
        threshold = rng.choice(thresholds)
        winner, log_probs, rolls = play_match(
            policy,
            rng,
            opponent_threshold=threshold,
            training=True,
        )

        reward = 1.0 if winner == 0 else -1.0
        recent_wins.append(int(winner == 0))
        recent_rolls.append(rolls)

        if log_probs:
            loss = -reward * torch.stack(log_probs).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        if episode % 200 == 0:
            wins = sum(recent_wins[-200:]) / 200
            avg_rolls = sum(recent_rolls[-200:]) / 200

            print(
                f"Epizóda {episode:4d} | "
                f"výhry za posledných 200: {wins:.1%} | "
                f"priemerný počet hodov: {avg_rolls:.1f}"
            )

    return policy


def evaluate(
    policy: PolicyNetwork,
    matches: int = 200,
    seed: int = 123,
) -> None:
    rng = random.Random(seed)

    for threshold in (250, 500, 800):
        wins = 0

        for _ in range(matches):
            winner, _, _ = play_match(
                policy,
                rng,
                opponent_threshold=threshold,
                training=False,
            )
            wins += winner == 0

        print(
            f"Súper s hranicou {threshold}: "
            f"{wins}/{matches} výhier ({wins / matches:.1%})"
        )

def inspect_decision(
    policy: PolicyNetwork,
    state: MatchState,
    goal: int = GOAL,
) -> None:
    actions = legal_actions(state, goal=goal)

    if not actions:
        print("V tomto hode neexistuje bodujúca voľba: FARKLE.")
        return

    batch = torch.stack(
        [action_features(state, action, goal=goal) for action in actions]
    )

    policy.eval()

    with torch.no_grad():
        logits = policy(batch)
        probabilities = torch.softmax(logits, dim=0)

    own = state.scores[state.player]
    opponent = state.scores[1 - state.player]

    print(
        f"\nSkóre: ja={own}, súper={opponent} | "
        f"body v ťahu={state.turn_points}"
    )
    print(
        f"Hod={format_dice(state.roll)} | "
        f"počet kociek v hode={state.dice_available}"
    )
    print(
        "\nModel hodnotí všetky legálne voľby "
        "(vyššia pravdepodobnosť = preferovanejšia voľba):"
    )

    rows = sorted(
        zip(actions, logits.tolist(), probabilities.tolist()),
        key=lambda row: row[2],
        reverse=True,
    )

    for action, logit, probability in rows:
        remaining = state.dice_available - sum(action.keep)
        next_dice = 6 if remaining == 0 else remaining
        wins = own + state.turn_points + action.points >= goal

        decision = (
            "OKAMŽITÁ VÝHRA"
            if wins
            else "BANK"
            if action.bank
            else "POKRAČUJ"
        )

        print(
            f"{format_dice(action.keep):>6} | "
            f"{action.points:>4} bodov | "
            f"ďalej {next_dice} kociek | "
            f"{decision:<16} | "
            f"logit={logit:>7.3f} | "
            f"p={probability:>7.2%}"
        )

def inspect_score_context(policy: PolicyNetwork) -> None:
    scenarios = [
        ("vyrovnaný zápas", 2_800, 2_800, 350),
        ("súper vedie", 2_800, 3_800, 350),
        ("my vedieme", 3_800, 2_800, 350),
        ("blízko výhry", 3_800, 3_800, 100),
    ]

    roll = count_dice([2, 2, 2, 1, 5, 3])

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

        policy.eval()

        with torch.no_grad():
            logits = policy(batch)
            probabilities = torch.softmax(logits, dim=0)

        best_index = torch.argmax(probabilities).item()
        best = actions[best_index]

        print(
            f"{name:>18}: ja={own}, súper={opponent}, "
            f"ťah={turn_points} -> "
            f"odlož {format_dice(best.keep)}, "
            f"{best.points} bodov, "
            f"{'BANK' if best.bank else 'POKRAČUJ'}, "
            f"p={probabilities[best_index].item():.1%}"
        )

def test_scoring() -> None:
    assert score_selection(count_dice([1, 1, 1, 1])) == 2_000
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

    selections = {
        action.keep
        for action in legal_actions(state)
    }

    assert count_dice([5]) in selections
    assert count_dice([1]) in selections
    assert count_dice([1, 1, 1, 5]) in selections


if __name__ == "__main__":
    test_scoring()
    print("Základné testy bodovania prešli.\n")

    trained_policy = train(episodes=10_000)

    torch.save(trained_policy.state_dict(), "dice_policy.pt")
    print("\nModel uložený do dice_policy.pt")

    print("\nVyhodnotenie:")
    evaluate(trained_policy)

    print("\nUkážkový zápas:")
    play_match(
        trained_policy,
        random.Random(2026),
        opponent_threshold=500,
        training=False,
        verbose=True,
    )

    example_state = MatchState(
        scores=[2_800, 3_200],
        player=0,
        turn_points=350,
        dice_available=6,
        roll=count_dice([2, 2, 2, 1, 5, 3]),
    )

    inspect_decision(trained_policy, example_state)
    inspect_score_context(trained_policy)
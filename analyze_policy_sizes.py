from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import dice_rl_critic as critic_experiment
import dice_rl_v2 as game
from compare_policy_sizes import ARCHITECTURES, ActionPolicy


COLORS = {
    "large": "#31688e",
    "small": "#35b779",
    "narrow": "#e78a31",
}

LABELS = {
    "full_bank": "123456 → bank",
    "full_continue": "123456 → continue",
    "short_bank": "23456 → bank",
    "short_continue": "23456 → continue",
}


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Výsledky neexistujú: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("CSV súbor neobsahuje výsledky.")

    return rows


def load_policy(folder: Path, architecture: str, seed: int) -> ActionPolicy:
    path = folder / f"{architecture}_seed_{seed}.pt"

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint neexistuje: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["architecture"] != architecture:
        raise ValueError(f"Nesprávna architektúra v {path}")

    policy = ActionPolicy(ARCHITECTURES[architecture])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    policy.requires_grad_(False)
    return policy


def style_plots() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    figure.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_architectures(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[row["architecture"]].append(row)

    names = list(ARCHITECTURES)
    thresholds = (250, 500, 650, 800)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    left, right = axes

    for position, name in enumerate(names):
        group = grouped[name]
        values = [100 * float(row["vs_v2"]) for row in group]
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        left.errorbar(
            position,
            mean,
            yerr=sd,
            fmt="o",
            capsize=5,
            color=COLORS[name],
            markersize=10,
            linewidth=2,
            zorder=3,
        )

        jitter = np.linspace(-0.10, 0.10, len(values))

        for offset, value in zip(jitter, values):
            left.scatter(
                position + offset,
                value,
                s=38,
                facecolors="white",
                edgecolors=COLORS[name],
                linewidths=1.7,
                zorder=4,
            )

    left.axhline(50, color="#777777", linestyle="--", linewidth=1)
    left.set_xticks(range(len(names)), names)
    left.set_ylabel("Win rate against v2 (%)")
    left.set_title("Direct matches against v2")
    left.text(
        0.02,
        0.03,
        "Filled: mean ± SD\nOpen: individual training seeds",
        transform=left.transAxes,
        fontsize=9,
        va="bottom",
    )

    for name in names:
        group = grouped[name]
        means = []
        deviations = []

        for threshold in thresholds:
            values = [
                100 * float(row[f"vs_{threshold}"])
                for row in group
            ]
            means.append(float(np.mean(values)))
            deviations.append(
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

        right.errorbar(
            thresholds,
            means,
            yerr=deviations,
            marker="o",
            capsize=3,
            linewidth=2,
            color=COLORS[name],
            label=name,
        )

    right.set_xticks(thresholds)
    right.set_xlabel("Heuristic opponent: bank threshold")
    right.set_ylabel("Win rate (%)")
    right.set_title("Performance across fixed opponents")
    right.legend(frameon=False)

    figure.suptitle(
        "Policy size experiment: 10,000 training matches per seed",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "architecture_comparison")


def example_state() -> game.MatchState:
    return game.MatchState(
        scores=[800, 800],
        player=0,
        turn_points=100,
        dice_available=6,
        roll=game.count_dice([1, 2, 3, 4, 5, 6]),
    )


def action_probabilities(
    policy: ActionPolicy,
    state: game.MatchState,
) -> dict[game.Action, float]:
    actions = game.legal_actions(state)
    features = torch.stack(
        [game.action_features(state, action) for action in actions]
    )

    with torch.no_grad():
        probabilities = torch.softmax(policy(features), dim=0)

    return {
        action: probability.item()
        for action, probability in zip(actions, probabilities)
    }


def selected_actions(
    state: game.MatchState,
) -> dict[str, game.Action]:
    full = game.count_dice([1, 2, 3, 4, 5, 6])
    short = game.count_dice([2, 3, 4, 5, 6])

    choices = {
        "full_bank": game.Action(full, 1_500, True),
        "full_continue": game.Action(full, 1_500, False),
        "short_bank": game.Action(short, 750, True),
        "short_continue": game.Action(short, 750, False),
    }

    legal = set(game.legal_actions(state))

    for label, action in choices.items():
        if action not in legal:
            raise ValueError(f"Nelegálna testovaná akcia: {label}")

    return choices


def choose_action(
    policy: torch.nn.Module,
    state: game.MatchState,
) -> game.Action:
    actions = game.legal_actions(state)
    features = torch.stack(
        [game.action_features(state, action) for action in actions]
    )

    with torch.no_grad():
        index = policy(features).argmax().item()

    return actions[index]


def simulate_after_action(
    initial: game.MatchState,
    forced_action: game.Action,
    policy: ActionPolicy,
    opponent: game.PolicyNetwork,
    seed: int,
    max_additional_rolls: int = 300,
) -> int:
    """Vráti 1, ak po nanútenej akcii nakoniec vyhrá hráč 0."""

    state = copy.deepcopy(initial)
    outcome = game.apply_action(state, forced_action)

    if outcome == "win":
        return int(state.winner == 0)

    rng = random.Random(seed)

    for _ in range(max_additional_rolls):
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

        current_policy = policy if state.player == 0 else opponent
        action = choose_action(current_policy, state)
        outcome = game.apply_action(state, action)

        if outcome == "win":
            return int(state.winner == 0)

    raise RuntimeError(
        "Simulácia prekročila limit hodov. "
        "Zvýš max_additional_rolls."
    )


def analyze_decision(
    policy: ActionPolicy,
    opponent: game.PolicyNetwork,
    rollouts: int,
) -> list[dict[str, float | int | str]]:
    state = example_state()
    choices = selected_actions(state)
    probabilities = action_probabilities(policy, state)
    results: list[dict[str, float | int | str]] = []

    for label, action in choices.items():
        wins = 0

        for index in range(rollouts):
            wins += simulate_after_action(
                initial=state,
                forced_action=action,
                policy=policy,
                opponent=opponent,
                seed=500_000 + index,
            )

        lower, upper = game.wilson_interval(wins, rollouts)

        results.append(
            {
                "action": label,
                "policy_probability": probabilities[action],
                "wins": wins,
                "rollouts": rollouts,
                "estimated_win_rate": wins / rollouts,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    return results


def plot_decision(
    results: list[dict[str, float | int | str]],
    output_dir: Path,
    seed: int,
) -> None:
    labels = [LABELS[str(row["action"])] for row in results]
    choice_probabilities = [
        100 * float(row["policy_probability"])
        for row in results
    ]
    win_rates = [
        100 * float(row["estimated_win_rate"])
        for row in results
    ]

    lower_errors = [
        100 * (
            float(row["estimated_win_rate"])
            - float(row["ci_lower"])
        )
        for row in results
    ]
    upper_errors = [
        100 * (
            float(row["ci_upper"])
            - float(row["estimated_win_rate"])
        )
        for row in results
    ]

    y = np.arange(len(results))
    bar_colors = ["#31688e", "#35b779", "#756bb1", "#e78a31"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].barh(y, choice_probabilities, color=bar_colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Probability of choosing the action (%)")
    axes[0].set_title("What the policy prefers")

    axes[1].barh(y, win_rates, color=bar_colors)
    axes[1].errorbar(
        win_rates,
        y,
        xerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.5,
    )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Estimated final win rate (%)")
    axes[1].set_title("What happens if we force that action")

    figure.suptitle(
        f"Narrow policy, seed {seed} | score 800:800, "
        "turn points 100, roll 123456",
        fontsize=12,
        weight="bold",
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "decision_123456")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/policy_sizes_10k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/policy_analysis"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("dice_policy_v2.pt"),
    )
    args = parser.parse_args()

    if args.rollouts < 0:
        parser.error("--rollouts nesmie byť záporné")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    style_plots()

    rows = load_results(args.results_dir / "results.csv")
    plot_architectures(rows, args.output_dir)

    print(
        "Uložené: architecture_comparison.png "
        "a architecture_comparison.pdf"
    )

    if args.rollouts == 0:
        return

    policy = load_policy(args.results_dir, "narrow", args.seed)
    opponent = critic_experiment.load_policy(args.reference)
    opponent.eval()
    opponent.requires_grad_(False)

    results = analyze_decision(
        policy=policy,
        opponent=opponent,
        rollouts=args.rollouts,
    )

    csv_path = args.output_dir / "decision_123456.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\nHod 123456; skóre 800:800; body v ťahu 100:")

    for row in results:
        print(
            f"  {LABELS[str(row['action'])]:<18} | "
            f"politika vyberie: "
            f"{float(row['policy_probability']):6.1%} | "
            f"po vynútení vyhrá: "
            f"{int(row['wins'])}/{int(row['rollouts'])} = "
            f"{float(row['estimated_win_rate']):6.1%} "
            f"(95 % Wilsonov interval "
            f"{float(row['ci_lower']):.1%}–"
            f"{float(row['ci_upper']):.1%})"
        )

    plot_decision(results, args.output_dir, args.seed)
    print(f"\nGrafy a tabuľka sú v {args.output_dir}")


if __name__ == "__main__":
    main()
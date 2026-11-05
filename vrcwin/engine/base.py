"""Season-independent interface between a game definition and the solver.

The solver only ever sees this protocol. A season (e.g. Override 2026-27)
implements it in vrcwin/games/<season>/. Swapping seasons means writing a
new Game, not touching the engine.

Model: a two-alliance, simultaneous-move, semi-Markov game. Each robot runs
one macro-action ("line") at a time; a decision point occurs whenever at
least one robot is idle. All idle robots on both alliances choose at the
same instant, without seeing the other alliance's choice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, Sequence

RED, BLUE = 0, 1
ALLIANCE_NAMES = ("red", "blue")


@dataclass(frozen=True)
class Outcome:
    red: int
    blue: int

    @property
    def diff(self) -> int:
        """Score margin from red's perspective."""
        return self.red - self.blue

    @property
    def red_win(self) -> float:
        """1 for a red win, 0.5 for a tie, 0 for a loss."""
        return 1.0 if self.red > self.blue else 0.5 if self.red == self.blue else 0.0


class Game(Protocol):
    def key(self, state) -> Hashable:
        """Hashable identity of a state, used for the search tree."""

    def infoset(self, state, robot: int) -> Hashable:
        """Abstraction of what this robot conditions its choice on. States with the
        same infoset (and the same legal actions) share one regret learner."""

    def clock(self, state) -> float:
        """Seconds remaining, for display."""

    def is_terminal(self, state) -> bool: ...

    def outcome(self, state) -> Outcome: ...

    def idle_robots(self, state, alliance: int) -> Sequence[int]:
        """Robot indices on this alliance that need a new macro-action now."""

    def alliance_of(self, state, robot: int) -> int: ...

    def legal_actions(self, state, robot: int) -> Sequence[Hashable]:
        """Macro-actions open to this robot. Never empty (WAIT is always legal)."""

    def step(self, state, actions: dict, rng):
        """Commit the given {robot: action} for every idle robot, then advance
        time to the next decision point (or the end of the match), sampling
        any chance outcomes with rng. Returns the new state."""

    def rollout_action(self, state, robot: int, rng) -> Hashable:
        """Cheap default policy used past the edge of the search tree."""

    def describe(self, action) -> str: ...

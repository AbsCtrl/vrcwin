"""Simultaneous-move regret matching over macro-actions, with state abstraction.

Each robot is its own regret-matching learner (decoupled; teammates share
their alliance's utility), in the style of SM-MCTS with regret matching
(Lisý, Kovařík, Lanctot, Bošanský, NeurIPS 2013). Learners are keyed by the
game's *information abstraction* (Game.infoset) instead of the exact state,
the same move poker solvers make with card bucketing: states a robot cannot
usefully tell apart share one learner, so experience generalises across the
thousands of slightly different states a continuous-time match produces.

Each iteration plays one full episode from the root. Robots whose situation
already has a learner sample from its current regret-matching strategy (plus
gamma exploration); the first time the episode reaches situations with no
learner, those are added (MCTS-style expansion) and the rest of the episode
uses the game's default policy wherever no learner exists. Every learner
touched is then updated with the episode's utility. Average strategies are
the solved policy.

Utility is red's P(win) (win 1, tie 0.5, loss 0) plus a tiny margin term so
that among lines with equal P(win) the solver prefers the larger margin.
"""
from __future__ import annotations

import random

from .base import BLUE, RED, Game, Outcome

MARGIN_WEIGHT = 1e-3     # utility per 100 points of margin; tie-break only
PRIOR_REGRET = 0.05      # warm start toward the game's heuristic action


def utility(o: Outcome) -> float:
    return o.red_win + MARGIN_WEIGHT * max(-1.0, min(1.0, o.diff / 100.0))


class Learner:
    """Regret matching+ over one robot's macro-actions in one abstract situation.

    Variance reduction: unsampled actions are credited with a running baseline
    of their own past returns, and the sampled action with baseline + importance-
    weighted correction (VR-MCCFR, Schmid et al. 2019). Averaging is linear in
    the learner's visit count (later, better strategies weigh more)."""

    __slots__ = ("actions", "regret", "strat_sum", "visits", "base", "base_n")

    def __init__(self, actions, prior=None):
        n = len(actions)
        self.actions = list(actions)
        self.regret = [0.0] * n
        self.strat_sum = [0.0] * n
        self.base = [0.5] * n
        self.base_n = [0] * n
        self.visits = 0
        if prior is not None and prior in self.actions:
            self.regret[self.actions.index(prior)] = PRIOR_REGRET

    def current(self, gamma: float):
        n = len(self.actions)
        tot = sum(self.regret)
        if tot <= 0:
            return [1.0 / n] * n
        return [(1 - gamma) * r / tot + gamma / n for r in self.regret]

    def average(self):
        tot = sum(self.strat_sum)
        n = len(self.actions)
        return [x / tot for x in self.strat_sum] if tot > 0 else [1.0 / n] * n

    def update(self, strat, chosen, val):
        self.visits += 1
        n = len(self.actions)
        if n == 1:
            self.strat_sum[0] += 1.0
            return
        b = self.base
        est = [b[k] for k in range(n)]
        est[chosen] = b[chosen] + (val - b[chosen]) / strat[chosen]
        pure = self.current(0.0)                # average the policy, not the exploration
        ev = sum(p * e for p, e in zip(pure, est))
        w = self.visits
        for k in range(n):
            self.regret[k] = max(0.0, self.regret[k] + est[k] - ev)
            self.strat_sum[k] += w * pure[k]
        self.base_n[chosen] += 1
        b[chosen] += (val - b[chosen]) / min(self.base_n[chosen], 200)


def _sample(rng, probs):
    x, acc = rng.random(), 0.0
    for k, p in enumerate(probs):
        acc += p
        if x < acc:
            return k
    return len(probs) - 1


class Solver:
    def __init__(self, game: Game, seed: int = 0, gamma: float = 0.2, min_visits: int = 30):
        self.game = game
        self.rng = random.Random(seed)
        self.gamma = gamma
        self.min_visits = min_visits
        self.learners: dict = {}

    def _learner(self, state, r, create=True):
        g = self.game
        acts = tuple(g.legal_actions(state, r))
        key = (r, g.infoset(state, r), acts)
        lr = self.learners.get(key)
        if lr is None and create:
            lr = self.learners[key] = Learner(acts, g.rollout_action(state, r, self.rng))
        return lr

    # ---- search -----------------------------------------------------------
    def _episode(self, state):
        g, rng = self.game, self.rng
        touched = []
        expanded = False
        while not g.is_terminal(state):
            acts = {}
            grow = False
            for a in (RED, BLUE):
                for r in g.idle_robots(state, a):
                    lr = self._learner(state, r, create=not expanded)
                    if lr is None:                       # past the edge: default policy
                        acts[r] = g.rollout_action(state, r, rng)
                        continue
                    grow = grow or lr.visits == 0
                    strat = lr.current(self.gamma)
                    k = _sample(rng, strat)
                    acts[r] = lr.actions[k]
                    touched.append((lr, strat, k, a))
            expanded = expanded or grow
            state = g.step(state, acts, rng)
        u = utility(g.outcome(state))
        for lr, strat, k, a in touched:
            lr.update(strat, k, u if a == RED else 1.0 - u)
        return u

    def search(self, state, iterations: int):
        for _ in range(iterations):
            self._episode(state)

    # ---- using the solution ----------------------------------------------
    def choose(self, state, r, rng, greedy=False):
        lr = self._learner(state, r, create=False)
        if lr is None or lr.visits < self.min_visits:
            return self.game.rollout_action(state, r, rng)
        probs = lr.average()
        if greedy:
            return lr.actions[max(range(len(probs)), key=probs.__getitem__)]
        return lr.actions[_sample(rng, probs)]

    def play_out(self, state, rng, force: dict = None) -> Outcome:
        g = self.game
        first = True
        while not g.is_terminal(state):
            acts = {}
            for a in (RED, BLUE):
                for r in g.idle_robots(state, a):
                    if first and force and r in force:
                        acts[r] = force[r]
                    else:
                        acts[r] = self.choose(state, r, rng)
            first = False
            state = g.step(state, acts, rng)
        return g.outcome(state)

    def evaluate(self, state, episodes: int, force: dict = None, seed: int = 1):
        """P(red win) and E[red margin] when everyone follows the solved strategy
        (optionally with some robots' first macro-action forced)."""
        rng = random.Random(seed)
        wins, diffs = 0.0, []
        for _ in range(episodes):
            o = self.play_out(state, rng, force)
            wins += o.red_win
            diffs.append(o.diff)
        n = len(diffs)
        mean = sum(diffs) / n
        sd = (sum((d - mean) ** 2 for d in diffs) / max(1, n - 1)) ** 0.5
        p = wins / n
        return {"p_red_win": p, "p_se": (p * (1 - p) / n) ** 0.5,
                "e_margin": mean, "margin_se": sd / n ** 0.5, "n": n}

    def root_strategy(self, state):
        """{robot: [(prob, action), ...]} at the root, most likely first."""
        g = self.game
        out = {}
        for a in (RED, BLUE):
            for r in g.idle_robots(state, a):
                lr = self._learner(state, r, create=False)
                out[r] = sorted(zip(lr.average(), lr.actions), key=lambda x: -x[0]) if lr else []
        return out

    def principal_variation(self, state, max_steps: int = 14, seed: int = 7):
        """Most likely decision sequence under the solved strategy (one sampled chance path)."""
        g = self.game
        rng = random.Random(seed)
        line = []
        for _ in range(max_steps):
            if g.is_terminal(state):
                break
            acts = {r: self.choose(state, r, rng, greedy=True)
                    for a in (RED, BLUE) for r in g.idle_robots(state, a)}
            line.append((g.clock(state), acts))
            state = g.step(state, acts, rng)
        return line

    def merge(self, other: "Solver"):
        """Fold another independent run's learners into this one (sums average-
        strategy weight; visit-weighted baselines). Used to pool parallel runs."""
        for key, lo in other.learners.items():
            mine = self.learners.get(key)
            if mine is None:
                self.learners[key] = lo
                continue
            for k in range(len(mine.actions)):
                n1, n2 = mine.base_n[k], lo.base_n[k]
                if n1 + n2:
                    mine.base[k] = (mine.base[k] * n1 + lo.base[k] * n2) / (n1 + n2)
                mine.base_n[k] = n1 + n2
                mine.strat_sum[k] += lo.strat_sum[k]
                mine.regret[k] += lo.regret[k]
            mine.visits += lo.visits

"""Exact stage: a small zero-sum matrix game over shortlisted joint plans.

Screening (regret matching in smmcts.py) narrows each robot to a few plans.
Here each alliance's actions are joint plans of its two robots, every
(red joint, blue joint) cell is estimated by simulation with common random
numbers, and the matrix game is solved. A double-oracle loop then scans all
joint plans for a best response the shortlist missed and adds it, so the
answer is not limited by what screening happened to keep. This is the "small matrix game at a
decision point" from the design notes; with the match's simultaneous moves
it is the right object, not a minimax tree.
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import random

from .base import BLUE, RED

_JOB = {}


def _cell(args):
    rj, bj, episodes, seed = args
    g, root = _JOB["game"], _JOB["root"]
    acts = dict(rj)
    acts.update(bj)
    wins = margin = 0.0
    for e in range(episodes):
        end = g.step(root, acts, random.Random(seed * 1_000_003 + e))   # common random numbers
        o = g.outcome(end)
        wins += o.red_win
        margin += o.diff
    return rj, bj, episodes, wins / episodes, margin / episodes


class Payoffs:
    """Cache of simulated (red joint plan, blue joint plan) cells, filled in parallel.
    A joint plan is a tuple of (robot, plan) pairs."""

    def __init__(self, game, root, workers: int = 4, seed: int = 0, margin_weight: float = 0.0):
        self.game, self.root, self.workers, self.seed = game, root, workers, seed
        self.margin_weight = margin_weight
        self.cells = {}                    # (rj, bj) -> (episodes, p_red, margin_red)

    def fill(self, pairs, episodes):
        todo = list({(rj, bj) for rj, bj in pairs
                     if (rj, bj) not in self.cells or self.cells[(rj, bj)][0] < episodes})
        if not todo:
            return
        _JOB.update(game=self.game, root=self.root)
        with mp.get_context("fork").Pool(self.workers) as pool:
            for rj, bj, n, p, m in pool.imap_unordered(
                    _cell, [(rj, bj, episodes, self.seed) for rj, bj in todo], chunksize=4):
                self.cells[(rj, bj)] = (n, p, m)

    def p(self, rj, bj):
        return self.cells[(rj, bj)][1]

    def margin(self, rj, bj):
        return self.cells[(rj, bj)][2]

    def u(self, rj, bj):
        n, p, m = self.cells[(rj, bj)]
        return p + self.margin_weight * max(-1.0, min(1.0, m / 100.0))


def all_joints(game, root, alliance, plans_of=None):
    robots = list(game.idle_robots(root, alliance))
    menus = [plans_of[r] if plans_of else game.legal_actions(root, r) for r in robots]
    return [tuple(zip(robots, combo)) for combo in itertools.product(*menus)]


def double_oracle(pay: Payoffs, red_set, blue_set, red_all, blue_all, episodes: int,
                  scan_episodes: int = 100, tol: float = 0.01, max_rounds: int = 6, log=print):
    """Solve the restricted matrix game, then look for a joint plan outside the
    sets that beats the current equilibrium; add it and repeat. Stops when no
    alliance can gain more than `tol` P(win) by switching to any joint plan.

    Returns dict with sets, mixes, value and the best-response gaps."""
    red_set, blue_set = list(red_set), list(blue_set)
    for rnd in range(max_rounds):
        pay.fill([(r, b) for r in red_set for b in blue_set], episodes)
        U = [[pay.u(r, b) for b in blue_set] for r in red_set]
        x, y, value, _ = solve_zero_sum(U)
        sup_y = [(w, b) for w, b in zip(y, blue_set) if w > 0.02]
        sup_x = [(w, r) for w, r in zip(x, red_set) if w > 0.02]
        ty, tx = sum(w for w, _ in sup_y), sum(w for w, _ in sup_x)
        # Scan every joint plan against the opponent's support, cheaply.
        scan = Payoffs(pay.game, pay.root, pay.workers, pay.seed + 17, pay.margin_weight)
        scan.fill([(r, b) for r in red_all for _, b in sup_y] + [(r, b) for _, r in sup_x for b in blue_all],
                  scan_episodes)
        red_scan = sorted(red_all, key=lambda r: -sum(w * scan.u(r, b) for w, b in sup_y) / ty)[:3]
        blue_scan = sorted(blue_all, key=lambda b: sum(w * scan.u(r, b) for w, r in sup_x) / tx)[:3]
        # Verify the best candidates at full precision against the full mixes.
        pay.fill([(r, b) for r in red_scan for b in blue_set] + [(r, b) for r in red_set for b in blue_scan], episodes)
        red_br = max(red_scan, key=lambda r: sum(w * pay.u(r, b) for w, b in zip(y, blue_set)))
        blue_br = min(blue_scan, key=lambda b: sum(w * pay.u(r, b) for w, r in zip(x, red_set)))
        red_gap = sum(w * pay.u(red_br, b) for w, b in zip(y, blue_set)) - value
        blue_gap = value - sum(w * pay.u(r, blue_br) for w, r in zip(x, red_set))
        log(f"  round {rnd}: {len(red_set)}x{len(blue_set)} value {value:.3f}, "
            f"red could gain {red_gap:+.3f}, blue could gain {blue_gap:+.3f}")
        added = False
        if red_gap > tol and red_br not in red_set:
            red_set.append(red_br)
            added = True
        if blue_gap > tol and blue_br not in blue_set:
            blue_set.append(blue_br)
            added = True
        if not added:
            break
    pay.fill([(r, b) for r in red_set for b in blue_set], episodes)
    U = [[pay.u(r, b) for b in blue_set] for r in red_set]
    x, y, value, exploit = solve_zero_sum(U)
    return {"red": red_set, "blue": blue_set, "x": x, "y": y, "value": value,
            "red_gap": red_gap, "blue_gap": blue_gap, "rounds": rnd + 1}


def solve_zero_sum(A, iterations: int = 20000):
    """Row player maximises x'Ay, column player minimises. Regret matching+ with
    linear averaging. Returns (x, y, value, exploitability)."""
    import numpy as np

    A = np.asarray(A, dtype=float)
    n, m = A.shape
    rx, ry = np.zeros(n), np.zeros(m)
    sx, sy = np.zeros(n), np.zeros(m)

    def strat(r):
        t = r.sum()
        return r / t if t > 0 else np.full(len(r), 1.0 / len(r))

    for it in range(1, iterations + 1):
        x, y = strat(rx), strat(ry)
        row, col = A @ y, x @ A
        v = x @ row
        rx = np.maximum(0.0, rx + row - v)
        ry = np.maximum(0.0, ry + v - col)
        sx += it * x
        sy += it * y
    x, y = strat(sx), strat(sy)
    row, col = A @ y, x @ A
    return x.tolist(), y.tolist(), float(x @ row), float(row.max() - col.min())

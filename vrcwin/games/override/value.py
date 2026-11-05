"""Learned value function for Override endgame snapshots (first version).

The build order is: the endgame solver gives exact-ish values for a
snapshot; a cheap model is trained on those values so it can rate
positions instantly (and, later, earlier in the match). This module has
the three pieces:

  random_scenario(rng, base)  a randomised snapshot to label
  label(scenario)             equilibrium P(red win) / margin from the solver
                              (screening + double oracle, no line table)
  features(scenario)          hand-made features, red's perspective, including
                              "who gets there first" times (the continuum part)
  ValueModel                  logistic regression on soft P(win) labels, numpy only

The model is deliberately simple: with tens to hundreds of solver labels a
linear model on good features beats anything deeper, and its weights can
be read. The baseline it must beat is score margin + time remaining only,
the same baseline the verification plan uses for footage.
"""
from __future__ import annotations

import copy
import json
import math
import random

import numpy as np

from ...engine import matrix, parallel
from ...engine.base import BLUE, RED
from ...engine.smmcts import MARGIN_WEIGHT
from . import plans, rules as R, sim

# ---- features ---------------------------------------------------------------

FEATURES = (
    "margin_now",            # red - blue if the match ended now (incl. Midfield robots), /50
    "margin_per_sqrt_t",     # margin_now / sqrt(time left), the classic win-prob scaling
    "time_left",             # seconds / 30
    "yellow_owned",          # yellow halves on Quadrant Goals red owns minus blue owns, /5
    "yellow_neutral_q",      # yellow halves on Quadrant Goals with a neutral Toggle, /5
    "mid_yellow",            # yellow halves on the Midfield Goal, /5
    "toggles",               # Toggles red minus Toggles blue, /4
    "held_pins",             # Pins held, red minus blue, /2
    "loader_pins",           # Pins in own Loader, red minus blue, /4
    "floor_yellow",          # yellow halves still on the floor, /6
    "mid_reach",             # sum over alliances of (blue - red) fastest time into the Midfield, s/3
    "toggle_reach",          # over Toggles worth holding: yellow-weighted (blue - red) first-arrival time
    "missing_caps",          # capabilities removed, blue minus red
)


def _reach(prof, pos, target):
    return prof.travel_time(R.dist(pos, target))


def features(scenario: dict, profiles=None) -> list:
    game, s = sim.build(scenario, profiles)
    out = game.outcome(s)
    t = max(s.t, 0.5)
    margin = out.diff
    toggles = dict(zip(R.QUADRANT_ORDER, s.toggles))
    goals = dict(zip(sim.GOAL_ORDER, s.goals))
    owned = neutral = 0
    yellow_q = {q: 0 for q in R.QUADRANT_ORDER}
    for g, (_, _, y) in goals.items():
        q = R.GOALS[g][3]
        if q is None:
            continue
        yellow_q[q] += y
        if toggles[q] == "R":
            owned += y
        elif toggles[q] == "B":
            owned -= y
        else:
            neutral += y
    tog = sum(1 for c in s.toggles if c == "R") - sum(1 for c in s.toggles if c == "B")
    held = sum(1 if b.alliance == RED else -1 for b in s.bots if b.holding)
    loader = sum(s.loader[RED]) - sum(s.loader[BLUE])
    fy = sum(c[R.PIN_TYPES.index("YY")] * 2 + c[R.PIN_TYPES.index("RY")] + c[R.PIN_TYPES.index("BY")]
             for c in s.floor)

    def fastest(alliance, target):
        ts = [_reach(game.profiles[i], b.pos, target) for i, b in enumerate(s.bots) if b.alliance == alliance]
        return min(ts) if ts else 30.0

    mid = fastest(BLUE, (0, 0)) - fastest(RED, (0, 0))
    treach = 0.0
    for q in R.QUADRANT_ORDER:
        p = R.QUADRANTS[q][:2]
        w = 1 + yellow_q[q]
        treach += w * max(-3.0, min(3.0, fastest(BLUE, p) - fastest(RED, p)))
    caps = 0
    for i, b in enumerate(s.bots):
        missing = 3 - len(game.profiles[i].capabilities & {"tall_goal", "toggle", "push"})
        caps += missing if b.alliance == BLUE else -missing
    return [margin / 50, margin / math.sqrt(t) / 10, t / 30, owned / 5, neutral / 5,
            goals["MID"][2] / 5, tog / 4, held / 2, loader / 4, fy / 6, mid / 3, treach / 10, caps]


# ---- model ------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ValueModel:
    """P(red win) = sigmoid(w . standardised features + b), fit to soft labels."""

    def __init__(self, names=FEATURES):
        self.names = list(names)
        self.w = np.zeros(len(self.names))
        self.b = 0.0
        self.mu = np.zeros(len(self.names))
        self.sd = np.ones(len(self.names))
        self.stats = {}

    def fit(self, X, y, l2=0.3, iters=4000, lr=0.2):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w, b = np.zeros(Z.shape[1]), 0.0
        n = len(y)
        for _ in range(iters):
            p = _sigmoid(Z @ w + b)
            g = p - y
            w -= lr * (Z.T @ g / n + l2 * w / n)
            b -= lr * g.mean()
        self.w, self.b = w, float(b)
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return _sigmoid(Z @ self.w + self.b)

    def contributions(self, x):
        """Per-feature logit contribution for one snapshot (positive helps red)."""
        z = (np.asarray(x, float) - self.mu) / self.sd
        return {n: float(c) for n, c in zip(self.names, z * self.w)}

    def to_json(self):
        return {"names": self.names, "w": self.w.tolist(), "b": self.b, "mu": self.mu.tolist(),
                "sd": self.sd.tolist(), "stats": self.stats}

    @classmethod
    def from_json(cls, d):
        m = cls(d["names"])
        m.w, m.b = np.array(d["w"]), d["b"]
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        m.stats = d.get("stats", {})
        return m


BASELINE = ("margin_now", "margin_per_sqrt_t", "time_left")


def cross_validate(X, y, names=FEATURES, folds=5, seed=0):
    """Out-of-fold Brier score and log loss for the full model and the
    margin + time baseline."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    idx = np.random.default_rng(seed).permutation(len(y))
    cols = [names.index(n) for n in BASELINE]
    pred = np.zeros(len(y))
    base = np.zeros(len(y))
    k = max(2, min(folds, len(y)))
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        pred[te] = ValueModel(names).fit(X[tr], y[tr]).predict(X[te])
        base[te] = ValueModel(BASELINE).fit(X[tr][:, cols], y[tr]).predict(X[te][:, cols])

    def scores(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return {"brier": round(float(np.mean((p - y) ** 2)), 4),
                "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
                "mae": round(float(np.mean(np.abs(p - y))), 4)}

    return {"n": int(len(y)), "folds": k, "model": scores(pred), "baseline": scores(base),
            "oof": pred.tolist(), "oof_baseline": base.tolist()}


# ---- training data ----------------------------------------------------------

def random_scenario(rng: random.Random, base: dict) -> dict:
    """A plausible endgame snapshot: base scenario's field, randomised state."""
    s = copy.deepcopy(base)
    s["description"] = "random snapshot for value-function training"
    s["time_remaining"] = round(rng.uniform(12, 30), 1)
    for r in s["robots"]:
        r["x"], r["y"] = round(rng.uniform(-1.6, 1.6), 2), round(rng.uniform(-1.6, 1.6), 2)
        own = "RY" if r["alliance"] == "red" else "BY"
        r["holding"] = rng.choice([None, None, own, "YY"])
        if r["holding"] is None:
            r.pop("holding")
    s["toggles"] = {q: rng.choice("RBN") for q in R.QUADRANT_ORDER}
    s["goals"] = {}
    for g, v in R.GOALS.items():
        kind = v[2]
        red = rng.randint(0, 3) if kind in ("red", "short") else 0
        blue = rng.randint(0, 3) if kind in ("blue", "short") else 0
        if kind == "tall":
            red, blue = 0, 0
        s["goals"][g] = [red, blue, rng.randint(0, 3)]
    for st in s.get("floor", {}).values():
        st["pins"] = {"YY": rng.randint(0, 2), "RY": rng.randint(0, 1), "BY": rng.randint(0, 1)}
    s["loader"] = {"red": {"RY": rng.randint(0, 3), "YY": rng.randint(0, 1)},
                   "blue": {"BY": rng.randint(0, 3), "YY": rng.randint(0, 1)}}
    auton = rng.choice([(12, 0), (0, 12), (6, 6)])
    s["base"] = [auton[0] + rng.randint(0, 40), auton[1] + rng.randint(0, 40)]
    return s


def label(scenario: dict, iters=12000, screen_episodes=60, cell_episodes=60, scan_episodes=40,
          shortlist=3, workers=4, seed=0, log=lambda m: None):
    """Equilibrium value of a snapshot: the same screening + double oracle as
    run_endgame.solve, at lower precision and without the line table."""
    game, state = sim.build(scenario)
    pg = plans.PlanGame(game)
    root = (state, None)
    n = len(scenario["robots"])
    screener = parallel.solve(pg, root, iters, workers=workers, seed=seed, gamma=0.1)
    forces = [{r: p} for r in range(n) for p in pg.legal_actions(root, r)]
    evals = iter(parallel.evaluate_many(screener, root, forces, screen_episodes, workers=workers, seed=seed + 1))
    short = {}
    for r in range(n):
        sign = 1 if game.alliance_of(state, r) == RED else -1
        rows = []
        for p in pg.legal_actions(root, r):
            ev = next(evals)
            pw = ev["p_red_win"] if sign == 1 else 1 - ev["p_red_win"]
            rows.append((pw, sign * ev["e_margin"], p))
        rows.sort(key=lambda x: (-x[0], -x[1]))
        short[r] = [p for _, _, p in rows[:shortlist]]
    pay = matrix.Payoffs(pg, root, workers=workers, seed=seed + 2, margin_weight=MARGIN_WEIGHT)
    red_all, blue_all = matrix.all_joints(pg, root, RED), matrix.all_joints(pg, root, BLUE)
    do = matrix.double_oracle(pay, matrix.all_joints(pg, root, RED, short), matrix.all_joints(pg, root, BLUE, short),
                              red_all, blue_all, cell_episodes, scan_episodes=scan_episodes, log=log)
    red_j, blue_j, x, y = do["red"], do["blue"], do["x"], do["y"]
    p_red = sum(x[i] * y[j] * pay.p(r, b) for i, r in enumerate(red_j) for j, b in enumerate(blue_j))
    m_red = sum(x[i] * y[j] * pay.margin(r, b) for i, r in enumerate(red_j) for j, b in enumerate(blue_j))
    return {"p_red_win": round(p_red, 4), "e_margin_red": round(m_red, 2),
            "gap": round(max(do["red_gap"], do["blue_gap"]), 4)}


def load_dataset(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows

"""Run independent solver copies on several cores and pool them."""
from __future__ import annotations

import multiprocessing as mp

from .smmcts import Solver

_JOB = {}


def _search(args):
    seed, iters = args
    sv = Solver(_JOB["game"], seed=seed, **_JOB["kw"])
    sv.search(_JOB["state"], iters)
    return sv.learners


def _eval(args):
    force, episodes, seed = args
    return _JOB["solver"].evaluate(_JOB["state"], episodes, force=force, seed=seed)


def solve(game, state, iterations: int, workers: int = 4, seed: int = 0, **kw) -> Solver:
    _JOB.update(game=game, state=state, kw=kw)
    per = max(1, iterations // workers)
    with mp.get_context("fork").Pool(workers) as pool:
        tables = pool.map(_search, [(seed * 1000 + w, per) for w in range(workers)])
    out = Solver(game, seed=seed, **kw)
    for t in tables:
        other = Solver(game, **kw)
        other.learners = t
        out.merge(other)
    return out


def evaluate_many(solver: Solver, state, forces, episodes: int, workers: int = 4, seed: int = 1):
    """[evaluate(force) for force in forces], spread across cores. The same seed
    is used for every force (common random numbers), so lines compare fairly."""
    _JOB.update(solver=solver, state=state)
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_eval, [(f, episodes, seed) for f in forces])

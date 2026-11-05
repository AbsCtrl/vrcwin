"""Solve an Override endgame scenario and print the engine's line table.

    python3 run_endgame.py scenarios/contested_top_25s.json --out results/contested_top_25s.json

Two stages:
1. Screening. Regret matching over every robot's whole-window plans
   (vrcwin/games/override/plans.py), then each plan is scored against the
   others' screening mix. The best few plans per robot go forward.
2. Exact. Each alliance's actions are the joint plans of its two robots;
   every (red joint, blue joint) cell is simulated with common random
   numbers and the zero-sum matrix game is solved. The equilibrium mix,
   its value and its exploitability come from this stage.

A robot's line value = its alliance's P(win) when it plays that plan, its
partner plays the best complementary shortlisted plan, and the opponents
play their equilibrium mix.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vrcwin.engine import matrix, parallel  # noqa: E402
from vrcwin.engine.base import ALLIANCE_NAMES, BLUE, RED
from vrcwin.engine.smmcts import MARGIN_WEIGHT
from vrcwin.games.override import plans, sim


def solve(scenario, iters=200000, screen_episodes=600, cell_episodes=300, shortlist=4,
          workers=4, seed=0, profiles=None, line_episodes=150, log=print):
    game, state = sim.build(scenario, profiles)
    pg = plans.PlanGame(game)
    root = (state, None)
    names = [r.get("name", f"robot {i}") for i, r in enumerate(scenario["robots"])]
    n = len(names)

    # Stage 1: screening.
    screener = parallel.solve(pg, root, iters, workers=workers, seed=seed, gamma=0.1)
    forces = [{r: p} for r in range(n) for p in pg.legal_actions(root, r)]
    evals = iter(parallel.evaluate_many(screener, root, forces, screen_episodes, workers=workers, seed=seed + 1))
    screen = {}
    for r in range(n):
        sign = 1 if game.alliance_of(state, r) == RED else -1
        rows = []
        for p in pg.legal_actions(root, r):
            ev = next(evals)
            pw = ev["p_red_win"] if sign == 1 else 1 - ev["p_red_win"]
            rows.append((pw, sign * ev["e_margin"], p))
        rows.sort(key=lambda x: (-x[0], -x[1]))
        screen[r] = rows
    short = {r: [p for _, _, p in screen[r][:shortlist]] for r in range(n)}

    # Stage 2: exact matrix game, grown by double oracle until no joint plan beats it.
    pay = matrix.Payoffs(pg, root, workers=workers, seed=seed + 2, margin_weight=MARGIN_WEIGHT)
    red_all, blue_all = matrix.all_joints(pg, root, RED), matrix.all_joints(pg, root, BLUE)
    do = matrix.double_oracle(pay, matrix.all_joints(pg, root, RED, short), matrix.all_joints(pg, root, BLUE, short),
                              red_all, blue_all, cell_episodes, log=log)
    red_j, blue_j, x, y = do["red"], do["blue"], do["x"], do["y"]
    p_red = sum(x[i] * y[j] * pay.p(r, b) for i, r in enumerate(red_j) for j, b in enumerate(blue_j))
    m_red = sum(x[i] * y[j] * pay.margin(r, b) for i, r in enumerate(red_j) for j, b in enumerate(blue_j))

    # Line table: every joint plan against the opponent's equilibrium mix.
    sup = {RED: [(w, j) for w, j in zip(x, red_j) if w > 0.005],
           BLUE: [(w, j) for w, j in zip(y, blue_j) if w > 0.005]}
    pay.fill([(r, b) for r in red_all for _, b in sup[BLUE]] + [(r, b) for _, r in sup[RED] for b in blue_all],
             line_episodes)

    def vs_mix(a, jt):
        opp = sup[BLUE if a == RED else RED]
        tot = sum(w for w, _ in opp)
        if a == RED:
            pw = sum(w * pay.p(jt, b) for w, b in opp) / tot
            m = sum(w * pay.margin(jt, b) for w, b in opp) / tot
            return pw, m
        pw = sum(w * pay.p(r, jt) for w, r in opp) / tot
        m = sum(w * pay.margin(r, jt) for w, r in opp) / tot
        return 1 - pw, -m

    lines = {}
    for r in range(n):
        a = game.alliance_of(state, r)
        own_all, own_set, w = (red_all, red_j, x) if a == RED else (blue_all, blue_j, y)
        weight = {}
        for wt, jt in zip(w, own_set):
            weight[dict(jt)[r]] = weight.get(dict(jt)[r], 0.0) + wt
        rows = []
        for p in pg.legal_actions(root, r):
            cands = [jt for jt in own_all if dict(jt)[r] == p]
            vals = [(vs_mix(a, jt), jt) for jt in cands]
            (pw, m), jt = max(vals, key=lambda v: (v[0][0], v[0][1]))
            partner = [q for rr, q in jt if rr != r]
            rows.append({"line": plans.describe(p), "plan": list(p), "p_win": round(pw, 4), "e_margin": round(m, 2),
                         "eq_weight": round(weight.get(p, 0.0), 3),
                         "best_partner": plans.describe(partner[0]) if partner else None})
        rows.sort(key=lambda z: (-z["p_win"], -z["e_margin"]))
        lines[names[r]] = {"alliance": ALLIANCE_NAMES[a], "lines": rows}

    def mix(joints, w):
        out = [(wt, jt) for wt, jt in zip(w, joints) if wt > 0.02]
        out.sort(key=lambda z: -z[0])
        return [{"p": round(wt, 3), "plan": {names[r]: plans.describe(p) for r, p in jt}} for wt, jt in out]

    top = dict(red_j[max(range(len(x)), key=x.__getitem__)])
    top.update(blue_j[max(range(len(y)), key=y.__getitem__)])
    events, out = pg.trace(root, top, random.Random(seed))
    trace = [{"t": t, "robot": names[i], "action": game.describe(a)} for t, i, a in events]

    return {"scenario": scenario.get("description", ""), "time_remaining": state.t,
            "overall": {"p_red_win": round(p_red, 4), "e_margin_red": round(m_red, 2),
                        "red_could_gain": round(do["red_gap"], 4), "blue_could_gain": round(do["blue_gap"], 4)},
            "equilibrium": {"red": mix(red_j, x), "blue": mix(blue_j, y)},
            "lines": lines, "sample_trace": trace, "sample_final": [out.red, out.blue],
            "settings": {"screen_iters": iters, "screen_episodes": screen_episodes,
                         "cell_episodes": cell_episodes, "line_episodes": line_episodes, "shortlist": shortlist,
                         "seed": seed, "oracle_rounds": do["rounds"]}}


def fmt(res):
    o = res["overall"]
    out = [f"{res['time_remaining']:.0f} s left. Value: P(red win) {o['p_red_win']:.3f}, "
           f"E[red margin] {o['e_margin_red']:+.1f}. Best deviation found: red {o['red_could_gain']:+.3f}, "
           f"blue {o['blue_could_gain']:+.3f}",
           "\nEquilibrium:"]
    for side, mx in res["equilibrium"].items():
        for m in mx:
            out.append(f"  {side} {m['p']:.2f}: " + "; ".join(f"{k}: {v}" for k, v in m["plan"].items()))
    for robot, d in res["lines"].items():
        side = d["alliance"]
        best = d["lines"][0]["p_win"]
        out.append(f"\n{robot} ({side}) lines: P(win) for {side}, margin, Δ vs best, equilibrium weight")
        for row in d["lines"]:
            out.append(f"  {row['p_win']:6.3f}  {row['e_margin']:+6.1f}  Δ{row['p_win'] - best:+.3f}  "
                       f"w={row['eq_weight']:.2f}  {row['line']}")
    out.append("\nSample window under the most likely plans:")
    for e in res["sample_trace"][:30]:
        out.append(f"  {e['t']:5.2f}s  {e['robot']}: {e['action']}")
    out.append(f"  final {res['sample_final'][0]}-{res['sample_final'][1]}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--screen-episodes", type=int, default=600)
    ap.add_argument("--cell-episodes", type=int, default=300)
    ap.add_argument("--shortlist", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    scen = json.load(open(args.scenario))
    t0 = time.time()
    res = solve(scen, args.iters, args.screen_episodes, args.cell_episodes, args.shortlist,
                args.workers, args.seed)
    res["seconds"] = round(time.time() - t0, 1)
    print(fmt(res))
    print(f"\n{res['seconds']} s")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)

"""Does the top line survive ±15% changes to the reference robot?

    python3 sensitivity.py scenarios/contested_top_25s.json --out results/sensitivity.json

Each stat is scaled for all four robots at once (the reference robot moves),
the scenario is re-solved, and each robot's best line is compared with the
baseline. If the top lines hold, flattening that stat is justified. A
capability check then removes one robot's Toggle ability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_endgame import solve
from vrcwin.robot import RobotProfile

PERTURB = {
    "speed": ("max_speed",),
    "cycle": ("intake_time", "place_time"),
    "toggle_time": ("toggle_time",),
    "push_success": ("p_push",),
}


def summary(res):
    return {"p_red_win": res["overall"]["p_red_win"], "e_margin_red": res["overall"]["e_margin_red"],
            "top": {k: d["lines"][0]["line"] for k, d in res["lines"].items()},
            "top_p": {k: d["lines"][0]["p_win"] for k, d in res["lines"].items()},
            "second_gap": {k: round(d["lines"][0]["p_win"] - d["lines"][1]["p_win"], 3)
                           for k, d in res["lines"].items()},
            "eq_top": {side: mx[0]["plan"] for side, mx in res["equilibrium"].items()},
            "eq_top_p": {side: mx[0]["p"] for side, mx in res["equilibrium"].items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--out")
    args = ap.parse_args()
    scen = json.load(open(args.scenario))
    n = len(scen["robots"])
    runs = {"baseline": summary(solve(scen, args.iters))}
    print("baseline", runs["baseline"], flush=True)
    for name, fields in PERTURB.items():
        for f in (0.85, 1.15):
            prof = RobotProfile().scaled(**{k: f for k in fields})
            if "p_push" in fields:
                prof = RobotProfile(p_push=min(1.0, RobotProfile().p_push * f))
            res = summary(solve(scen, args.iters, profiles=[prof] * n))
            res["changed"] = [k for k in res["top"] if res["top"][k] != runs["baseline"]["top"][k]]
            res["eq_changed"] = [a for a in res["eq_top"] if res["eq_top"][a] != runs["baseline"]["eq_top"][a]]
            runs[f"{name} x{f}"] = res
            print(name, f, res["changed"], res["eq_changed"], res["p_red_win"], flush=True)
    # Capability check: blue 2 cannot work Toggles.
    profs = [RobotProfile()] * n
    blue2 = next(i for i, r in enumerate(scen["robots"]) if r["alliance"] == "blue" and i == n - 1)
    profs[blue2] = RobotProfile().with_capabilities(remove=("toggle",))
    res = summary(solve(scen, args.iters, profiles=profs))
    res["changed"] = [k for k in res["top"] if res["top"][k] != runs["baseline"]["top"][k]]
    res["eq_changed"] = [a for a in res["eq_top"] if res["eq_top"][a] != runs["baseline"]["eq_top"][a]]
    runs[f"{scen['robots'][blue2]['name']} without toggle"] = res
    print("no toggle", res, flush=True)
    if args.out:
        json.dump(runs, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

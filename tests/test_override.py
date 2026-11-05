"""Rule checks for the Override model. Run: python3 -m pytest tests  (or python3 tests/test_override.py)"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vrcwin.games.override import rules as R
from vrcwin.games.override import sim
from vrcwin.games.override.plans import PlanGame
from vrcwin.robot import RobotProfile

EMPTY = {g: (0, 0, 0) for g in R.GOALS}
NEUTRAL = {q: "N" for q in R.QUADRANT_ORDER}


def test_manual_toggle_example():
    # Manual v2.1 <SC5a> example: 3 red, 1 blue, 3 yellow halves in a Quadrant Goal.
    goals = dict(EMPTY, L_N=(3, 1, 3))
    for tog, want in (("N", (15, 5)), ("B", (15, 35)), ("R", (45, 5))):
        assert R.final_score(goals, dict(NEUTRAL, L=tog), (0, 0), (0, 0)) == want


def test_midfield_goal_ownership_and_robots():
    goals = dict(EMPTY, MID=(0, 0, 2))
    assert R.final_score(goals, NEUTRAL, (2, 1), (0, 0)) == (16 + 20, 8)
    assert R.final_score(goals, NEUTRAL, (1, 1), (0, 0)) == (8, 8)      # tie: nobody owns <SC5b-i>


def test_travel_time_profile():
    p = RobotProfile()
    assert p.travel_time(0) == 0
    assert abs(p.travel_time(3.0) - (2 * 1.8 / 3.0 + (3.0 - 1.8 ** 2 / 3.0) / 1.8)) < 1e-9
    assert p.scaled(max_speed=1.15).travel_time(3.0) < p.travel_time(3.0)


def _scenario():
    path = os.path.join(os.path.dirname(__file__), "..", "scenarios", "contested_top_25s.json")
    return json.load(open(path))


def test_no_midfield_goal_placement_in_endgame():
    scen = _scenario()
    scen["time_remaining"] = 10.5
    game, s = sim.build(scen)
    for i in range(4):
        for a in game.legal_actions(s, i):
            assert not (a[0] == "SCORE" and a[1] == "MID"), a


def test_touched_toggle_is_neutral_at_buzzer():
    scen = _scenario()
    scen["time_remaining"] = 1.0
    scen["robots"][0].update(x=-1.4, y=0.0)          # right next to Toggle L
    game, s = sim.build(scen)
    s = game.step(s, {0: ("FLIP", "L"), 1: ("WAIT",), 2: ("WAIT",), 3: ("WAIT",)}, random.Random(0))
    while not game.is_terminal(s):
        s = game.step(s, {i: ("WAIT",) for a in (0, 1) for i in game.idle_robots(s, a)}, random.Random(0))
    base = game.outcome(s)
    # The same state with Toggle L red and nobody touching it would score more for red.
    scen["toggles"]["L"] = "R"
    g2, s2 = sim.build(scen)
    assert base.red < R.final_score(dict(zip(sim.GOAL_ORDER, s2.goals)), dict(zip(R.QUADRANT_ORDER, s2.toggles)),
                                    (0, 0), s2.base)[0]


def test_plan_window_runs_to_buzzer_and_conserves_pins():
    game, s = sim.build(_scenario())
    pg = PlanGame(game)
    start_pins = sum(map(sum, s.floor)) + sum(map(sum, s.loader)) + sum(1 for b in s.bots if b.holding)
    plans = {i: pg.legal_actions((s, None), i)[0] for i in range(4)}
    end, _ = pg.step((s, None), plans, random.Random(3))
    assert end.done and end.t == 0
    placed = sum(sum(g) for g in end.goals) - sum(sum(g) for g in s.goals)
    left = sum(map(sum, end.floor)) + sum(map(sum, end.loader)) + sum(1 for b in end.bots if b.holding)
    assert placed + left == start_pins


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

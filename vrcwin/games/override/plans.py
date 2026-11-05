"""Plan-level lines for the Override endgame.

A *line* is a whole-window plan for one robot, the analogue of a poker
solver's bet-size menu: a main job, plus an endgame stance that decides
whether and when it breaks off for the Midfield. Inside the plan the robot
picks its next macro-action with simple reactive rules (re-flip a Toggle
that was taken, push when out-numbered in the Midfield, and so on), so the
solver only has to choose between a few meaningfully different lines.

Main jobs
  score          cycle Pins; yellow up on a Goal whose Quadrant we own, else own-colour up
  score_mid      cycle Pins onto the Midfield Goal (yellow up) until the Endgame ban
  control:q      set Toggle q to our colour, then guard it and re-flip if it is lost
  midfield       take the Midfield now, hold it, push opponents out
Endgame stances
  stay           keep doing the main job to the buzzer
  mid@10         break for the Midfield when the Endgame starts (10 s left)
  mid@late       break for the Midfield at the last moment it can still get there
"""
from __future__ import annotations

from ...engine.base import BLUE, RED
from . import rules as R

STANCES = ("stay", "mid@10", "mid@late")
LATE_MARGIN = 1.0     # s of slack kept when breaking late for the Midfield


def plans_for(profile):
    mains = ["score"]
    if profile.can("tall_goal"):
        mains.append("score_mid")
    if profile.can("toggle"):
        mains += [f"control:{q}" for q in R.QUADRANT_ORDER]
    out = [(m, st) for m in mains for st in STANCES]
    out.append(("midfield", "stay"))
    return out


def describe(plan) -> str:
    main, stance = plan
    if main == "midfield":
        return "take and hold the Midfield all window"
    text = {
        "score": "score Pins (yellow where we own the Quadrant)",
        "score_mid": "score yellow on the Midfield Goal",
    }.get(main) or f"take and hold Toggle {main.split(':')[1]}"
    tail = {"stay": ", stay on it to the buzzer",
            "mid@10": ", Midfield at 10 s",
            "mid@late": ", Midfield at the last moment"}[stance]
    return text + tail


def next_action(game, s, i, plan, rng):
    """Macro-action robot i takes now under `plan`."""
    main, stance = plan
    b, prof = s.bots[i], game.profiles[i]
    acts = game.legal_actions(s, i)
    me = R.COLOR_OF_ALLIANCE[b.alliance]

    in_mid_job = main == "midfield"
    if stance == "mid@10" and s.t <= R.ENDGAME_SECONDS + 0.01:
        in_mid_job = True
    if stance == "mid@late":
        t_mid = prof.travel_time(R.dist(b.pos, R.midfield_entry(b.pos)))
        if s.t <= t_mid + LATE_MARGIN:
            in_mid_job = True
    if in_mid_job:
        return _midfield(game, s, i, acts)

    if main.startswith("control:"):
        q = main.split(":")[1]
        if ("FLIP", q) in acts:
            return ("FLIP", q)
        if ("GUARD", q) in acts:
            return ("GUARD", q)
        return ("WAIT",)

    # Scoring jobs.
    if b.holding is None:
        gets = [a for a in acts if a[0] == "GET"]
        if not gets:
            return _midfield(game, s, i, acts)
        return min(gets, key=lambda a: game._duration(s, i, a))
    scores = [a for a in acts if a[0] == "SCORE"]
    if main == "score_mid" and ("SCORE", "MID", "Y") in scores:
        return ("SCORE", "MID", "Y")
    owned = [a for a in scores if a[2] == "Y" and a[1] in R.QUADRANTS
             and s.toggles[R.QUADRANT_ORDER.index(a[1])] == me]
    if owned:
        return min(owned, key=lambda a: game._duration(s, i, a))
    own_half = [a for a in scores if a[2] == me]
    if own_half:
        return own_half[0]
    if scores:                       # only a yellow half left, nothing owned yet
        return min(scores, key=lambda a: game._duration(s, i, a))
    return ("WAIT",)


def _midfield(game, s, i, acts):
    b = s.bots[i]
    if not R.in_midfield(b.pos):
        return ("MID",)
    pushes = [a for a in acts if a[0] == "PUSH"]
    if pushes:
        mine = sum(1 for o in s.bots if o.alliance == b.alliance and R.in_midfield(game._target_loc(o)))
        theirs = len(pushes)
        if theirs >= mine:
            return pushes[0]
    return ("MID",)


class PlanGame:
    """Wraps OverrideEndgame so the solver's choice is one plan per robot at the
    start of the window; the window is then simulated with the plans' rules."""

    def __init__(self, game):
        self.game = game
        self.plans = [plans_for(p) for p in game.profiles]

    # State is (base_state, chosen plans or None).
    def key(self, st):
        return (st[1] is None, st[0].t)

    def infoset(self, st, r):
        return ("root",)

    def clock(self, st):
        return st[0].t

    def alliance_of(self, st, r):
        return st[0].bots[r].alliance

    def is_terminal(self, st):
        return st[1] is not None

    def outcome(self, st):
        return self.game.outcome(st[0])

    def idle_robots(self, st, alliance):
        if st[1] is not None:
            return ()
        return tuple(i for i, b in enumerate(st[0].bots) if b.alliance == alliance)

    def legal_actions(self, st, r):
        return self.plans[r]

    def rollout_action(self, st, r, rng):
        return ("score", "mid@late")

    def describe(self, plan):
        return describe(plan)

    def step(self, st, actions, rng):
        s = st[0]
        plans = tuple(actions[i] for i in range(len(s.bots)))
        g = self.game
        while not g.is_terminal(s):
            acts = {}
            for a in (RED, BLUE):
                for i in g.idle_robots(s, a):
                    acts[i] = next_action(g, s, i, plans[i], rng)
            s = g.step(s, acts, rng)
        return (s, plans)

    def trace(self, st, plans, rng):
        """Replay one window under fixed plans, returning (time, robot, action) events."""
        s = st[0]
        g = self.game
        events = []
        while not g.is_terminal(s):
            acts = {}
            for a in (RED, BLUE):
                for i in g.idle_robots(s, a):
                    acts[i] = next_action(g, s, i, plans[i], rng)
                    events.append((s.t, i, acts[i]))
            s = g.step(s, acts, rng)
        return events, g.outcome(s)

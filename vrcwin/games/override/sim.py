"""Override endgame as a simultaneous-move semi-Markov game (implements engine.base.Game).

Macro-actions ("lines") for one robot:
  GET src             drive to a Pin source (floor stack or own Loader) and intake one Pin
  SCORE goal half     drive to a Goal and Place the held Pin with `half` ("R"/"B"/"Y") on top
  FLIP q              drive to Quadrant q's Toggle and set it to our colour
  GUARD q             park in front of Toggle q for GUARD_SECONDS; opposing FLIPs there mostly fail
  MID                 drive into the Midfield and hold it for MID_HOLD_SECONDS
  PUSH r              drive at opposing robot r in the Midfield and try to shove it out
  WAIT                hold position for WAIT_SECONDS
Durations come from the robot profile (travel time + task time); success is
sampled when the action finishes.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

from ...engine.base import BLUE, RED, Outcome
from ...robot import RobotProfile
from . import rules as R

TICK = 0.05                 # time grid, s
GOAL_STANDOFF = 0.30        # robot centre to Goal centre when placing, m
TOGGLE_STANDOFF = 0.25
GUARD_STANDOFF = 0.55
GUARD_SECONDS = 3.0
GUARD_FLIP_FACTOR = 0.25    # an opposing FLIP into a guarded Toggle keeps this share of its success chance
MID_HOLD_SECONDS = 2.0
WAIT_SECONDS = 1.0


def _q(t: float) -> float:
    return round(round(t / TICK) * TICK, 2)


def _p(p) -> tuple:
    return (round(p[0], 2), round(p[1], 2))


class Bot(NamedTuple):
    alliance: int
    pos: tuple
    holding: Optional[str] = None       # pin type, e.g. "RY"
    action: Optional[tuple] = None
    end_t: float = 0.0                  # time remaining when the action resolves
    dest: Optional[tuple] = None
    arrive_t: float = 0.0               # time remaining when it reaches dest


class State(NamedTuple):
    t: float                            # seconds remaining
    bots: tuple
    goals: tuple                        # per GOAL_ORDER: (red, blue, yellow) visible halves
    toggles: tuple                      # per QUADRANT_ORDER: "R"/"B"/"N"
    floor: tuple                        # per floor source: counts per PIN_TYPES
    loader: tuple                       # per alliance: counts per PIN_TYPES
    base: tuple                         # (red, blue) points fixed before the window
    done: bool = False


GOAL_ORDER = tuple(R.GOALS)


class OverrideEndgame:
    """Game definition the solver drives. Holds the static parts of a scenario."""

    def __init__(self, profiles, floor_sources: dict):
        self.profiles = tuple(profiles)                 # RobotProfile per robot index
        self.floor_names = tuple(floor_sources)
        self.floor_pos = tuple(tuple(floor_sources[n]) for n in self.floor_names)
        self.loader_pos = {
            RED: [v[:2] for v in R.LOADERS.values() if v[2] == "red"],
            BLUE: [v[:2] for v in R.LOADERS.values() if v[2] == "blue"],
        }

    # ---- protocol ---------------------------------------------------------
    def key(self, s: State):
        return (s.t, s.bots, s.goals, s.toggles, s.floor, s.loader)

    def clock(self, s: State) -> float:
        return s.t

    def infoset(self, s: State, i: int):
        """Coarse abstraction a robot decides on: time (2.5 s buckets), its zone
        (Midfield or one of the four Quadrants), whether it holds a Pin, the
        Toggles relative to its alliance, the projected margin if the match ended
        now (bucketed), and Midfield occupancy by target (own, opposing)."""
        b = s.bots[i]
        me = R.COLOR_OF_ALLIANCE[b.alliance]
        tog = tuple(0 if c == "N" else 1 if c == me else -1 for c in s.toggles)
        mid = [0, 0]
        for o in s.bots:
            if R.in_midfield(self._target_loc(o)):
                mid[0 if o.alliance == b.alliance else 1] += 1
        margin = self.outcome(s).diff * (1 if b.alliance == RED else -1)
        mb = -2 if margin < -15 else -1 if margin < -5 else 0 if margin <= 5 else 1 if margin <= 15 else 2
        return (int(s.t / 2.5), _zone(b.pos), b.holding is not None, tog, mb, tuple(mid))

    def is_terminal(self, s: State) -> bool:
        return s.done

    def idle_robots(self, s: State, alliance: int):
        if s.done:
            return ()
        return tuple(i for i, b in enumerate(s.bots) if b.alliance == alliance and b.action is None)

    def outcome(self, s: State) -> Outcome:
        goals = dict(zip(GOAL_ORDER, s.goals))
        toggles = dict(zip(R.QUADRANT_ORDER, s.toggles))
        for b in s.bots:                                 # <SC4>: a touched Toggle is not set
            q = self._touching(b, 0.0)
            if q:
                toggles[q] = "N"
        counts = [0, 0]
        for b in s.bots:
            if R.in_midfield(self._loc(b, 0.0)):
                counts[b.alliance] += 1
        red, blue = R.final_score(goals, toggles, tuple(counts), s.base)
        return Outcome(red, blue)

    def alliance_of(self, s: State, i: int) -> int:
        return s.bots[i].alliance

    def legal_actions(self, s: State, i: int):
        """Canonical action set: it depends on what is possible, not on exact
        positions, so learners in the same abstract situation line up."""
        b, prof = s.bots[i], self.profiles[i]
        me = R.COLOR_OF_ALLIANCE[b.alliance]
        acts = [("WAIT",), ("MID",)]
        if b.holding is None:
            acts += [("GET", n) for n, c in zip(self.floor_names, s.floor) if sum(c) > 0]
            if sum(s.loader[b.alliance]) > 0:
                acts.append(("GET", "LOADER"))
        else:
            acts += self._score_actions(s, i, me)
        if prof.can("toggle"):
            for q, col in zip(R.QUADRANT_ORDER, s.toggles):
                acts.append(("GUARD", q) if col == me else ("FLIP", q))
        if prof.can("push"):
            for j, o in enumerate(s.bots):
                if o.alliance != b.alliance and R.in_midfield(self._target_loc(o)):
                    acts.append(("PUSH", j))
        return acts

    def step(self, s: State, actions: dict, rng) -> State:
        bots = list(s.bots)
        for i, a in actions.items():
            bots[i] = self._commit(s, i, a)
        s = s._replace(bots=tuple(bots))
        nxt = max(b.end_t for b in s.bots)
        if nxt < 0:
            return s._replace(t=0.0, done=True)
        s = s._replace(t=nxt)
        for i in sorted(range(len(s.bots)), key=lambda k: rng.random()):
            if s.bots[i].action is not None and s.bots[i].end_t >= nxt - 1e-9:
                s = self._resolve(s, i, rng)
        if nxt <= 0:
            s = s._replace(t=0.0, done=True)
        return s

    def rollout_action(self, s: State, i: int, rng):
        acts = self.legal_actions(s, i)
        if rng.random() < 0.2:
            return acts[rng.randrange(len(acts))]
        b, prof = s.bots[i], self.profiles[i]
        t_mid = prof.travel_time(R.dist(b.pos, R.midfield_entry(b.pos)))
        if s.t <= t_mid + 1.5:
            push = [a for a in acts if a[0] == "PUSH"]
            if push and R.in_midfield(b.pos) and rng.random() < 0.5:
                return push[0]
            return ("MID",)
        me = R.COLOR_OF_ALLIANCE[b.alliance]
        flips = [a for a in acts if a[0] == "FLIP"]
        if flips and rng.random() < 0.25:
            return min(flips, key=lambda a: self._duration(s, i, a))
        if b.holding is not None:
            best = max((a for a in acts if a[0] == "SCORE"),
                       key=lambda a: self._place_value(s, a, me) / (1.0 + self._duration(s, i, a)),
                       default=None)
            if best is not None:
                return best
        gets = [a for a in acts if a[0] == "GET"]
        if gets:
            return min(gets, key=lambda a: self._duration(s, i, a))
        return ("MID",)

    def describe(self, a) -> str:
        kind = a[0]
        if kind == "GET":
            return "intake Pin from " + ("own Loader" if a[1] == "LOADER" else f"floor stack {a[1]}")
        if kind == "SCORE":
            half = {"R": "red", "B": "blue", "Y": "yellow"}[a[2]]
            where = {"NEAREST": "nearest Goal", "MID": "Midfield Goal"}.get(a[1], f"Quadrant {a[1]}")
            return f"score {half} half on {where}"
        if kind == "FLIP":
            return f"flip Toggle {a[1]}"
        if kind == "GUARD":
            return f"guard Toggle {a[1]}"
        if kind == "MID":
            return "take/hold Midfield"
        if kind == "PUSH":
            return f"push robot {a[1]} out of Midfield"
        return "wait"

    # ---- mechanics --------------------------------------------------------
    def _score_actions(self, s, i, me):
        """("SCORE", where, half): where is "NEAREST", a Quadrant, or "MID"."""
        b, prof = s.bots[i], self.profiles[i]
        out = []
        if me in b.holding:
            out.append(("SCORE", "NEAREST", me))
        if "Y" in b.holding:
            for q, col in zip(R.QUADRANT_ORDER, s.toggles):
                if col != _opp(me):
                    out.append(("SCORE", q, "Y"))
            if prof.can("tall_goal"):
                out.append(("SCORE", "MID", "Y"))
        # <SG12-1>: no Placing on the Midfield Goal in the Endgame.
        return [a for a in out if not (a[1] == "MID" and s.t - self._duration(s, i, a) <= R.ENDGAME_SECONDS)]

    def _score_goal(self, s, i, a):
        b = s.bots[i]
        if a[1] == "MID":
            return "MID"
        cands = [g for g, v in R.GOALS.items()
                 if R.goal_alliance(g) in (None, b.alliance) and g != "MID"
                 and (a[1] == "NEAREST" or v[3] == a[1])]
        return min(cands, key=lambda g: R.dist(b.pos, R.GOALS[g][:2]))

    def _place_value(self, s, a, me):
        if a[2] != "Y":
            return R.PTS_ALLIANCE_HALF
        q = a[1]
        if q == "MID":
            return R.PTS_YELLOW_HALF * 0.5
        return R.PTS_YELLOW_HALF if s.toggles[R.QUADRANT_ORDER.index(q)] == me else R.PTS_YELLOW_HALF * 0.3

    def _target(self, s, i, a):
        """Where the action takes the robot, and the task time once there."""
        b, prof = s.bots[i], self.profiles[i]
        kind = a[0]
        if kind == "GET":
            if a[1] == "LOADER":
                p = min(self.loader_pos[b.alliance], key=lambda p: R.dist(b.pos, p))
            else:
                p = self.floor_pos[self.floor_names.index(a[1])]
            return _approach(b.pos, p, 0.2), prof.intake_time
        if kind == "SCORE":
            return _approach(b.pos, R.GOALS[self._score_goal(s, i, a)][:2], GOAL_STANDOFF), prof.place_time
        if kind == "FLIP":
            return _approach(b.pos, R.QUADRANTS[a[1]][:2], TOGGLE_STANDOFF), prof.toggle_time
        if kind == "GUARD":
            return _approach(b.pos, R.QUADRANTS[a[1]][:2], GUARD_STANDOFF), GUARD_SECONDS
        if kind == "MID":
            return R.midfield_entry(b.pos), MID_HOLD_SECONDS
        if kind == "PUSH":
            return R.midfield_entry(self._target_loc(s.bots[a[1]])), prof.push_time
        return b.pos, WAIT_SECONDS

    def _duration(self, s, i, a):
        dest, task = self._target(s, i, a)
        return self.profiles[i].travel_time(R.dist(s.bots[i].pos, dest)) + task

    def _commit(self, s, i, a):
        b = s.bots[i]
        dest, task = self._target(s, i, a)
        if a[0] == "SCORE":
            a = ("SCORE", self._score_goal(s, i, a), a[2])      # commit to a concrete Goal
        travel = self.profiles[i].travel_time(R.dist(b.pos, dest))
        arrive = _q(s.t - travel)
        end = _q(s.t - travel - task)
        if end >= s.t:
            end = _q(s.t - TICK)
        return b._replace(action=a, dest=_p(dest), arrive_t=arrive, end_t=end)

    def _loc(self, b: Bot, now: float):
        """Robot position at time-remaining `now` (in transit counts as nowhere useful)."""
        if b.action is None or b.dest is None:
            return b.pos
        if b.arrive_t >= now - 1e-9:
            return b.dest
        if R.in_midfield(b.pos) and R.in_midfield(b.dest):
            return b.pos
        return (99.0, 99.0)                              # in transit

    def _target_loc(self, b: Bot):
        return b.dest if b.action is not None and b.dest is not None else b.pos

    def _touching(self, b: Bot, now: float):
        if b.action is not None and b.action[0] == "FLIP" and b.arrive_t >= now - 1e-9 and b.end_t < now:
            return b.action[1]
        return None

    def _resolve(self, s: State, i: int, rng) -> State:
        b, prof = s.bots[i], self.profiles[i]
        a = b.action
        kind = a[0]
        me = R.COLOR_OF_ALLIANCE[b.alliance]
        nb = b._replace(pos=b.dest, action=None, dest=None)
        if kind == "GET":
            if a[1] == "LOADER":
                pool = s.loader[b.alliance]
                pick = _best_pin(pool, me)
                if pick is not None and rng.random() < prof.p_intake:
                    loader = list(s.loader)
                    loader[b.alliance] = _take(pool, pick)
                    s = s._replace(loader=tuple(loader))
                    nb = nb._replace(holding=R.PIN_TYPES[pick])
            else:
                k = self.floor_names.index(a[1])
                pick = _best_pin(s.floor[k], me)
                if pick is not None and rng.random() < prof.p_intake:
                    floor = list(s.floor)
                    floor[k] = _take(s.floor[k], pick)
                    s = s._replace(floor=tuple(floor))
                    nb = nb._replace(holding=R.PIN_TYPES[pick])
        elif kind == "SCORE":
            if b.holding is not None and rng.random() < prof.p_place:
                gi = GOAL_ORDER.index(a[1])
                r, bl, y = s.goals[gi]
                half = a[2]
                if half == "Y":
                    y += 1
                elif half == "R":
                    r += 1
                else:
                    bl += 1
                goals = list(s.goals)
                goals[gi] = (r, bl, y)
                s = s._replace(goals=tuple(goals))
                nb = nb._replace(holding=None)
        elif kind == "FLIP":
            p = prof.p_toggle
            qpos = R.QUADRANTS[a[1]][:2]
            for o in s.bots:
                if (o.alliance != b.alliance and o.action is not None and o.action[0] == "GUARD"
                        and o.action[1] == a[1] and o.arrive_t >= s.t - 1e-9):
                    p *= GUARD_FLIP_FACTOR
                    break
            if rng.random() < p:
                tg = list(s.toggles)
                tg[R.QUADRANT_ORDER.index(a[1])] = me
                s = s._replace(toggles=tuple(tg))
            nb = nb._replace(pos=_p(_approach(qpos, (0.0, 0.0), R.dist(qpos, (0.0, 0.0)) - 0.45)))   # back off the Toggle
        elif kind == "PUSH":
            j = a[1]
            o = s.bots[j]
            if R.in_midfield(self._loc(o, s.t)) and rng.random() < prof.p_push:
                where = self._loc(o, s.t)
                bots = list(s.bots)
                bots[j] = o._replace(pos=_p(R.midfield_exit(where)), action=None, dest=None)
                s = s._replace(bots=tuple(bots))
        bots = list(s.bots)
        bots[i] = nb
        return s._replace(bots=tuple(bots))


def _zone(p):
    if R.in_midfield(p):
        return "M"
    x, y = p
    if abs(x) >= abs(y):
        return "L" if x < 0 else "R"
    return "B" if y < 0 else "T"


def _opp(c):
    return "B" if c == "R" else "R"


def _approach(frm, to, standoff):
    d = R.dist(frm, to)
    if d <= standoff or d == 0:
        return frm
    f = (d - standoff) / d
    return (frm[0] + (to[0] - frm[0]) * f, frm[1] + (to[1] - frm[1]) * f)


def _best_pin(counts, me):
    """Index of the Pin type this alliance would pick up from a source."""
    pref = ("RY", "YY", "RB", "BY") if me == "R" else ("BY", "YY", "RB", "RY")
    for t in pref:
        k = R.PIN_TYPES.index(t)
        if counts[k] > 0:
            return k
    return None


def _take(counts, k):
    c = list(counts)
    c[k] -= 1
    return tuple(c)


def build(scenario: dict, profiles=None):
    """Make (game, initial_state) from a scenario dict (see scenarios/*.json)."""
    robots = scenario["robots"]
    if profiles is None:
        profiles = []
        for r in robots:
            p = RobotProfile()
            if "remove_capabilities" in r:
                p = p.with_capabilities(remove=tuple(r["remove_capabilities"]))
            if "scale" in r:
                p = p.scaled(**r["scale"])
            profiles.append(p)
    floor = {k: (v["x"], v["y"]) for k, v in scenario.get("floor", {}).items()}
    game = OverrideEndgame(profiles, floor)
    bots = tuple(Bot(alliance=RED if r["alliance"] == "red" else BLUE,
                     pos=_p((r["x"], r["y"])), holding=r.get("holding")) for r in robots)
    goals = tuple(tuple(scenario["goals"].get(g, (0, 0, 0))) for g in GOAL_ORDER)
    toggles = tuple(scenario["toggles"][q] for q in R.QUADRANT_ORDER)
    floor_counts = tuple(tuple(scenario["floor"][n]["pins"].get(t, 0) for t in R.PIN_TYPES)
                         for n in game.floor_names)
    loader = tuple(tuple(scenario["loader"][a].get(t, 0) for t in R.PIN_TYPES) for a in ("red", "blue"))
    state = State(t=_q(scenario["time_remaining"]), bots=bots, goals=goals, toggles=toggles,
                  floor=floor_counts, loader=loader, base=tuple(scenario.get("base", (0, 0))))
    return game, state

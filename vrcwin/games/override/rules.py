"""V5RC 2026-27 "Override": field geometry, constants and end-of-match scoring.

Sources
- Game manual v2.1 (2026-10-08), section 2 Scoring <SC1>-<SC8>, <SG6>,
  <SG9>, <SG10>, <SG12>: https://link.vex.com/docs/26-27/v5rc/game-manual
- Goal / Toggle / Loader coordinates: VEXcode VR Override playground GPS
  reference (field centre = (0, 0), millimetres, red Alliance Station on
  the -x wall): https://api.vex.com/vr/home/playgrounds/v5rc_override.html

Assumptions (marked ASSUMPTION below, all easy to change here):
1. Quadrants are the four triangles cut by the field diagonals (left,
   bottom, right, top). Evidence: one Toggle per wall, each wall-side
   triangle then holds exactly one Alliance Goal and one short neutral
   Goal, and the Skills rules name a single "Quadrant adjacent to the red
   Alliance Station". Left and bottom are red Quadrants, right and top blue.
2. Yellow Pin Halves on an Alliance Goal follow that Quadrant's Toggle the
   same way as the neutral Quadrant Goal (<SC5a> says "Quadrant Goals").
3. Each placed Pin (with its Cup) shows exactly one Scored Pin Half, the one
   the placing robot puts on top. Cup tricks that expose both halves or
   cover an opponent's half are not modelled.
4. The Midfield is a disc of MIDFIELD_RADIUS around the tall Goal.
"""
from __future__ import annotations

import math

# ---- Match timing and point values (manual v2.1) --------------------------
MATCH_DRIVER_SECONDS = 105
ENDGAME_SECONDS = 10            # <SG12>: no Placing on the Midfield Goal in the final 10 s
PTS_ALLIANCE_HALF = 5           # Scored red/blue Pin Half
PTS_YELLOW_HALF = 10            # Scored and Owned yellow Pin Half
PTS_MIDFIELD_ROBOT = 8          # each Robot ending in the Midfield
AUTON_BONUS = 12                # decided before the endgame; enters as base points

# ---- Geometry (metres) -----------------------------------------------------
MM = 0.001
MIDFIELD_RADIUS = 0.45          # ASSUMPTION 4

# name -> (x, y, kind, quadrant). kind: "red"/"blue" Alliance Goal, "short"
# neutral Quadrant Goal, "tall" neutral Midfield Goal.
GOALS = {
    "MID": (0, 0, "tall", None),
    "L_N": (-1200 * MM, 600 * MM, "short", "L"),
    "L_R": (-1200 * MM, -600 * MM, "red", "L"),
    "B_R": (-600 * MM, -1200 * MM, "red", "B"),
    "B_N": (600 * MM, -1200 * MM, "short", "B"),
    "R_N": (1200 * MM, -600 * MM, "short", "R"),
    "R_B": (1200 * MM, 600 * MM, "blue", "R"),
    "T_B": (600 * MM, 1200 * MM, "blue", "T"),
    "T_N": (-600 * MM, 1200 * MM, "short", "T"),
}

# Quadrant -> (toggle x, y, home colour). ASSUMPTION 1.
QUADRANTS = {
    "L": (-1780 * MM, 0.0, "R"),
    "B": (0.0, -1780 * MM, "R"),
    "R": (1780 * MM, 0.0, "B"),
    "T": (0.0, 1780 * MM, "B"),
}
QUADRANT_ORDER = ("L", "B", "R", "T")

LOADERS = {
    "LD_R1": (-1740 * MM, 1490 * MM, "red"),
    "LD_R2": (-1740 * MM, -1490 * MM, "red"),
    "LD_B1": (1740 * MM, 1490 * MM, "blue"),
    "LD_B2": (1740 * MM, -1490 * MM, "blue"),
}

# Pin types: the two halves. "RY" = red/yellow, etc.
PIN_TYPES = ("RB", "RY", "BY", "YY")
COLOR_OF_ALLIANCE = ("R", "B")


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def in_midfield(p) -> bool:
    return math.hypot(p[0], p[1]) <= MIDFIELD_RADIUS


def midfield_entry(p):
    """Nearest point inside the Midfield from p (robot centre just inside)."""
    r = math.hypot(p[0], p[1])
    inner = MIDFIELD_RADIUS * 0.7
    if r <= inner:
        return p
    return (p[0] * inner / r, p[1] * inner / r)


def midfield_exit(p):
    """Where a robot ends up after being shoved out of the Midfield."""
    r = math.hypot(p[0], p[1]) or 1e-9
    out = MIDFIELD_RADIUS + 0.35
    if r < 1e-6:
        return (-out, 0.0)
    return (p[0] * out / r, p[1] * out / r)


def goal_alliance(goal: str):
    """0/1 for an Alliance Goal, None for neutral."""
    kind = GOALS[goal][2]
    return 0 if kind == "red" else 1 if kind == "blue" else None


def final_score(goals: dict, toggles: dict, midfield_counts: tuple, base: tuple):
    """End-of-match score.

    goals: name -> (red_halves, blue_halves, yellow_halves) visible
    toggles: quadrant -> "R" / "B" / "N" (neutral, i.e. not set or touched)
    midfield_counts: (red robots in Midfield, blue robots in Midfield)
    base: (red, blue) points fixed before the window being solved
    """
    red, blue = base
    mid_owner = None
    if midfield_counts[0] > midfield_counts[1]:
        mid_owner = "R"
    elif midfield_counts[1] > midfield_counts[0]:
        mid_owner = "B"
    for name, (r, b, y) in goals.items():
        red += PTS_ALLIANCE_HALF * r
        blue += PTS_ALLIANCE_HALF * b
        quadrant = GOALS[name][3]
        owner = mid_owner if quadrant is None else toggles[quadrant]   # <SC5b> / <SC5a>
        if owner == "R":
            red += PTS_YELLOW_HALF * y
        elif owner == "B":
            blue += PTS_YELLOW_HALF * y
    red += PTS_MIDFIELD_ROBOT * midfield_counts[0]
    blue += PTS_MIDFIELD_ROBOT * midfield_counts[1]
    return red, blue

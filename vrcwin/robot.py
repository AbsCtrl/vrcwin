"""Reference robot profile.

One "optimal" high-level robot with fixed stats, plus capability toggles.
Travel speed stays a real parameter (it moves the who-gets-there-first
boundary directly); the rest are flattened to a reference value and
justified by the sensitivity pass in sensitivity.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class RobotProfile:
    name: str = "reference"
    # Motion (straight-line trapezoidal profile).
    max_speed: float = 1.8        # m/s, roughly a 450 rpm / 3.25" drive
    max_accel: float = 3.0        # m/s^2
    # Manipulation times, seconds.
    intake_time: float = 1.2      # pick up one Pin from the floor or a Loader
    place_time: float = 0.8       # place a held Pin (with its Cup) on a Goal
    toggle_time: float = 1.0      # rotate a wall Toggle and back off it
    push_time: float = 2.0        # one attempt to shove an opponent out of the Midfield
    # Success probabilities for one attempt.
    p_intake: float = 0.95
    p_place: float = 0.95
    p_toggle: float = 0.9
    p_push: float = 0.35          # vs. an equal robot holding its ground
    # Capability toggles (season-specific keys; the game decides what they gate).
    capabilities: frozenset = field(default_factory=lambda: frozenset(
        {"tall_goal", "toggle", "push"}))

    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    def travel_time(self, distance_m: float) -> float:
        """Time to cover distance_m from rest to rest."""
        if distance_m <= 0:
            return 0.0
        v, a = self.max_speed, self.max_accel
        d_ramp = v * v / a                     # accel + decel distance at full speed
        if distance_m <= d_ramp:
            return 2.0 * math.sqrt(distance_m / a)
        return 2.0 * v / a + (distance_m - d_ramp) / v

    def scaled(self, **factors: float) -> "RobotProfile":
        """Copy with numeric stats multiplied, e.g. scaled(max_speed=1.15)."""
        return replace(self, **{k: getattr(self, k) * f for k, f in factors.items()})

    def with_capabilities(self, *add: str, remove: tuple = ()) -> "RobotProfile":
        caps = (set(self.capabilities) | set(add)) - set(remove)
        return replace(self, capabilities=frozenset(caps))

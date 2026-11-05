"""Checks for the interface's back end and the value model. Run: python3 tests/test_ui.py"""
import json
import os
import random
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ui"))

import server  # noqa: E402
from vrcwin.games.override import value as V  # noqa: E402


def _scenario():
    return json.load(open(os.path.join(HERE, "..", "scenarios", "contested_top_25s.json")))


def test_simulate_and_replay_reach_buzzer():
    scen = _scenario()
    menu = server.plan_menu(scen)
    out = server.simulate(scen, [m[0]["id"] for m in menu], episodes=30)
    assert 0 <= out["p_red_win"] <= 1
    rp = out["replay"]
    assert rp["frames"][0]["t"] == scen["time_remaining"] and rp["frames"][-1]["t"] == 0
    assert all(s["t0"] >= s["arrive"] - 1e-9 for s in rp["segments"])
    assert rp["final"] == [rp["frames"][-1]["red"], rp["frames"][-1]["blue"]]


def test_features_track_the_scenario():
    scen = _scenario()
    x = V.features(scen)
    assert len(x) == len(V.FEATURES)
    scen["base"] = [scen["base"][0] + 50, scen["base"][1]]
    assert V.features(scen)[0] > x[0]                     # margin_now goes up with red's fixed points


def test_value_model_learns_a_margin_signal():
    rng = random.Random(0)
    base = _scenario()
    X, y = [], []
    for _ in range(40):
        s = V.random_scenario(rng, base)
        f = V.features(s)
        X.append(f)
        y.append(1 / (1 + 2.718 ** (-3 * f[0])))           # synthetic label driven by the margin
    cv = V.cross_validate(X, y)
    assert cv["model"]["brier"] < 0.02
    m = V.ValueModel().fit(X, y)
    again = V.ValueModel.from_json(json.loads(json.dumps(m.to_json())))
    assert abs(again.predict([X[0]])[0] - m.predict([X[0]])[0]) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

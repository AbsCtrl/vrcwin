"""VRCWin interface: a local browser app to set up, test, solve, train and examine.

    python3 ui/server.py            # then open http://localhost:8765
    python3 ui/server.py --port 9000

Standard library plus numpy, nothing to install. Long jobs (solves,
sensitivity, training) run as separate processes via ui/worker.py, so the
page stays responsive and several can be queued.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from vrcwin.engine.base import BLUE, RED  # noqa: E402
from vrcwin.games.override import plans, rules as R, sim  # noqa: E402
from vrcwin.games.override import value as V  # noqa: E402
from vrcwin.robot import RobotProfile  # noqa: E402

SCEN_DIR = os.path.join(ROOT, "scenarios")
RES_DIR = os.path.join(ROOT, "results")
RUN_DIR = os.path.join(HERE, "runs")
STATIC = os.path.join(HERE, "static")
DATASET = os.path.join(ROOT, "data", "value_dataset.jsonl")
MODEL = os.path.join(ROOT, "data", "value_model.json")
SAFE = re.compile(r"^[A-Za-z0-9_.\- ]+$")


# ---- static field / model metadata ------------------------------------------

def meta():
    d = RobotProfile()
    stats = ["max_speed", "max_accel", "intake_time", "place_time", "toggle_time", "push_time",
             "p_intake", "p_place", "p_toggle", "p_push"]
    return {
        "goals": {k: {"x": v[0], "y": v[1], "kind": v[2], "quadrant": v[3]} for k, v in R.GOALS.items()},
        "quadrants": {k: {"x": v[0], "y": v[1], "home": v[2]} for k, v in R.QUADRANTS.items()},
        "loaders": {k: {"x": v[0], "y": v[1], "alliance": v[2]} for k, v in R.LOADERS.items()},
        "midfield_radius": R.MIDFIELD_RADIUS,
        "pin_types": list(R.PIN_TYPES),
        "profile": {k: getattr(d, k) for k in stats},
        "capabilities": sorted(d.capabilities),
        "points": {"alliance_half": R.PTS_ALLIANCE_HALF, "yellow_half": R.PTS_YELLOW_HALF,
                   "midfield_robot": R.PTS_MIDFIELD_ROBOT},
        "features": list(V.FEATURES),
    }


def plan_menu(scenario):
    game, s = sim.build(scenario)
    out = []
    for i, prof in enumerate(game.profiles):
        out.append([{"id": f"{m}|{st}", "text": plans.describe((m, st))} for m, st in plans.plans_for(prof)])
    return out


def _plan(pid):
    m, st = pid.split("|")
    return (m, st)


# ---- quick test: fixed plans, many rollouts, one replay ---------------------

def simulate(scenario, plan_ids, episodes=500, seed=0):
    game, s0 = sim.build(scenario)
    names = [r.get("name", f"robot {i}") for i, r in enumerate(scenario["robots"])]
    chosen = tuple(_plan(p) for p in plan_ids)
    pg = plans.PlanGame(game)
    wins = 0.0
    margins = []
    for e in range(episodes):
        end, _ = pg.step((s0, None), dict(enumerate(chosen)), random.Random(seed * 1_000_003 + e))
        o = game.outcome(end)
        wins += o.red_win
        margins.append(o.diff)
    margins.sort()
    hist = {}
    for m in margins:
        b = int(m // 10) * 10
        hist[b] = hist.get(b, 0) + 1
    return {"p_red_win": round(wins / episodes, 4), "e_margin_red": round(sum(margins) / episodes, 2),
            "margin_p10": margins[len(margins) // 10], "margin_p90": margins[len(margins) * 9 // 10],
            "histogram": sorted(hist.items()), "episodes": episodes,
            "replay": replay(game, s0, chosen, names, random.Random(seed))}


def replay(game, s, chosen, names, rng):
    """One window under fixed plans, recorded for animation: per-robot motion
    segments plus a snapshot of the scoreboard after every decision point."""
    segs, frames = [], []

    def snap(st):
        o = game.outcome(st)
        frames.append({"t": st.t, "red": o.red, "blue": o.blue, "toggles": list(st.toggles),
                       "goals": {g: list(v) for g, v in zip(sim.GOAL_ORDER, st.goals)},
                       "holding": [b.holding for b in st.bots],
                       "floor": {n: sum(c) for n, c in zip(game.floor_names, st.floor)},
                       "loader": {side: {"all": sum(c)} for side, c in zip(("red", "blue"), st.loader)}})

    snap(s)
    while not game.is_terminal(s):
        acts = {}
        for a in (RED, BLUE):
            for i in game.idle_robots(s, a):
                acts[i] = plans.next_action(game, s, i, chosen[i], rng)
        before = s
        s = game.step(s, acts, rng)
        for i, a in acts.items():
            committed = game._commit(before, i, a)
            segs.append({"robot": i, "name": names[i], "t0": before.t, "from": list(before.bots[i].pos),
                         "to": list(committed.dest), "arrive": committed.arrive_t, "end": committed.end_t,
                         "action": game.describe(committed.action)})
        # Robots shoved out of the Midfield jump; start a new resting segment there.
        mid = before._replace(bots=tuple(game._commit(before, i, acts[i]) if i in acts else b
                                         for i, b in enumerate(before.bots)))
        for i, (b0, b1) in enumerate(zip(mid.bots, s.bots)):
            if b1.action is None and b0.action is not None and b0.dest is not None and tuple(b1.pos) != tuple(b0.dest):
                segs.append({"robot": i, "name": names[i], "t0": s.t, "from": list(b1.pos), "to": list(b1.pos),
                             "arrive": s.t, "end": s.t, "action": "pushed out of the Midfield"})
        snap(s)
    o = game.outcome(s)
    return {"segments": segs, "frames": frames, "final": [o.red, o.blue], "start_t": frames[0]["t"]}


# ---- value model ------------------------------------------------------------

def load_model():
    try:
        return V.ValueModel.from_json(json.load(open(MODEL)))
    except FileNotFoundError:
        return None


def evaluate(scenario, heat_robot=None, grid=21):
    m = load_model()
    x = V.features(scenario)
    game, st = sim.build(scenario)
    o = game.outcome(st)
    out = {"features": dict(zip(V.FEATURES, x)), "model": None, "score_now": [o.red, o.blue]}
    if m is None:
        return out
    out["model"] = {"p_red_win": float(m.predict([x])[0]), "contributions": m.contributions(x), "stats": {
        k: m.stats.get(k) for k in ("n", "model", "baseline", "fitted_at")}}
    if heat_robot is not None:
        xs = [-1.65 + 3.3 * k / (grid - 1) for k in range(grid)]
        cells = []
        rows = []
        for yy in xs:
            for xx in xs:
                sc = json.loads(json.dumps(scenario))
                sc["robots"][heat_robot]["x"], sc["robots"][heat_robot]["y"] = xx, yy
                rows.append(V.features(sc))
                cells.append((xx, yy))
        ps = m.predict(rows)
        out["heat"] = {"robot": heat_robot, "cells": [[c[0], c[1], float(p)] for c, p in zip(cells, ps)],
                       "step": 3.3 / (grid - 1)}
    return out


def model_info():
    m = load_model()
    rows = V.load_dataset(DATASET)
    return {"samples": len(rows), "model": m.stats if m else None,
            "weights": dict(zip(m.names, m.w.tolist())) if m else None,
            "recent": [{"t": r["scenario"]["time_remaining"], "p_red_win": r["p_red_win"],
                        "e_margin_red": r["e_margin_red"], "seconds": r.get("seconds")} for r in rows[-12:]]}


# ---- jobs -------------------------------------------------------------------

class Jobs:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        os.makedirs(RUN_DIR, exist_ok=True)

    def start(self, spec):
        jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        path = os.path.join(RUN_DIR, f"{jid}.json")
        json.dump(spec, open(path, "w"))
        logf = open(os.path.join(RUN_DIR, f"{jid}.log"), "w")
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "worker.py"), path], cwd=ROOT,
                                stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        job = {"id": jid, "kind": spec["kind"], "label": spec.get("label") or spec["kind"], "proc": proc,
               "started": time.time(), "log": os.path.join(RUN_DIR, f"{jid}.log"), "status": "running"}
        with self.lock:
            self.jobs[jid] = job
        return self.view(job)

    def _refresh(self, job):
        if job["status"] == "running" and job["proc"].poll() is not None:
            job["status"] = "done" if job["proc"].returncode == 0 else (
                "cancelled" if job.get("cancelled") else "failed")
            job["ended"] = time.time()

    def view(self, job, with_log=False):
        self._refresh(job)
        out = {k: job[k] for k in ("id", "kind", "label", "status", "started")}
        out["elapsed"] = round((job.get("ended") or time.time()) - job["started"], 1)
        text = open(job["log"]).read() if os.path.exists(job["log"]) else ""
        res = [ln for ln in text.splitlines() if ln.startswith("RESULT ")]
        if res:
            out["result"] = json.loads(res[-1][7:])
        lines = [ln for ln in text.splitlines() if not ln.startswith("RESULT ")]
        out["last"] = lines[-1] if lines else ""
        if with_log:
            out["log"] = "\n".join(lines[-400:])
        return out

    def list(self):
        with self.lock:
            js = list(self.jobs.values())
        return [self.view(j) for j in sorted(js, key=lambda j: -j["started"])]

    def get(self, jid):
        return self.view(self.jobs[jid], with_log=True)

    def cancel(self, jid):
        j = self.jobs[jid]
        if j["proc"].poll() is None:
            j["cancelled"] = True
            os.killpg(j["proc"].pid, signal.SIGTERM)
        return self.view(j)


JOBS = Jobs()


# ---- files ------------------------------------------------------------------

def _safe(name):
    if not name or not SAFE.match(name) or ".." in name:
        raise ValueError("bad name")
    return name


def list_dir(d, ext=".json"):
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(ext):
            p = os.path.join(d, f)
            out.append({"name": f, "modified": os.path.getmtime(p), "size": os.path.getsize(p)})
    return sorted(out, key=lambda r: -r["modified"])


# ---- HTTP -------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC, **kw)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/meta":
                return self._json(meta())
            if u.path == "/api/scenarios":
                return self._json(list_dir(SCEN_DIR))
            if u.path == "/api/scenario":
                return self._json(json.load(open(os.path.join(SCEN_DIR, _safe(q["name"])))))
            if u.path == "/api/results":
                return self._json(list_dir(RES_DIR))
            if u.path == "/api/result":
                return self._json(json.load(open(os.path.join(RES_DIR, _safe(q["name"])))))
            if u.path == "/api/jobs":
                return self._json(JOBS.list())
            if u.path == "/api/job":
                return self._json(JOBS.get(q["id"]))
            if u.path == "/api/model":
                return self._json(model_info())
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 400)
        if u.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self._body()
            if u.path == "/api/scenario":
                name = _safe(b["name"])
                if not name.endswith(".json"):
                    name += ".json"
                json.dump(b["scenario"], open(os.path.join(SCEN_DIR, name), "w"), indent=2)
                return self._json({"saved": name})
            if u.path == "/api/plans":
                return self._json(plan_menu(b["scenario"]))
            if u.path == "/api/simulate":
                return self._json(simulate(b["scenario"], b["plans"], int(b.get("episodes", 500)),
                                           int(b.get("seed", 0))))
            if u.path == "/api/evaluate":
                return self._json(evaluate(b["scenario"], b.get("heat_robot")))
            if u.path == "/api/jobs":
                if b.get("name"):
                    _safe(b["name"])
                return self._json(JOBS.start(b))
            if u.path == "/api/cancel":
                return self._json(JOBS.cancel(b["id"]))
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        return self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"VRCWin interface on http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

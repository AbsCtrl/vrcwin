"""Long-running jobs for the interface, one process per job.

    python3 -u ui/worker.py <job.json>

job.json: {"kind": "solve" | "sensitivity" | "tests" | "dataset" | "fit", ...}
Progress goes to stdout (the server streams it); the final line is
`RESULT <json>`.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET = os.path.join(ROOT, "data", "value_dataset.jsonl")
MODEL = os.path.join(ROOT, "data", "value_model.json")

PRESETS = {
    # ~2-3 min on 4 cores; matched the full solve to 0.002 P(win) on the sample scenario.
    "quick": dict(iters=20000, screen_episodes=100, cell_episodes=80, line_episodes=40, shortlist=4),
    # run_endgame.py defaults, ~6 min.
    "full": dict(iters=200000, screen_episodes=600, cell_episodes=300, line_episodes=150, shortlist=4),
}


def log(msg):
    print(msg, flush=True)


def run_solve(job):
    from run_endgame import solve
    scen = job["scenario"]
    kw = dict(PRESETS[job.get("preset", "quick")])
    kw.update({k: v for k, v in job.get("overrides", {}).items() if v is not None})
    log(f"Solving with preset '{job.get('preset', 'quick')}': {kw}")
    log("Stage 1: screening every robot's plans with regret matching...")
    t0 = time.time()
    res = solve(scen, workers=job.get("workers", 4), seed=job.get("seed", 0), log=log, **kw)
    res["seconds"] = round(time.time() - t0, 1)
    res["preset"] = job.get("preset", "quick")
    res["scenario_input"] = scen
    name = job.get("name") or "ui_result"
    path = os.path.join(ROOT, "results", f"{name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    log(f"Done in {res['seconds']} s. P(red win) {res['overall']['p_red_win']:.3f}, "
        f"E[red margin] {res['overall']['e_margin_red']:+.1f}. Saved results/{name}.json")
    return {"result": f"{name}.json"}


def run_sensitivity(job):
    from run_endgame import solve
    from sensitivity import PERTURB, summary
    from vrcwin.robot import RobotProfile
    scen = job["scenario"]
    kw = dict(PRESETS[job.get("preset", "quick")])
    n = len(scen["robots"])
    pct = job.get("pct", 15) / 100
    runs = {}
    log("Baseline solve...")
    runs["baseline"] = summary(solve(scen, log=log, **kw))
    for name, fields in PERTURB.items():
        for f in (1 - pct, 1 + pct):
            log(f"{name} x{f:.2f}...")
            if "p_push" in fields:
                prof = RobotProfile(p_push=min(1.0, RobotProfile().p_push * f))
            else:
                prof = RobotProfile().scaled(**{k: f for k in fields})
            r = summary(solve(scen, profiles=[prof] * n, log=log, **kw))
            r["changed"] = [k for k in r["top"] if r["top"][k] != runs["baseline"]["top"][k]]
            r["eq_changed"] = [a for a in r["eq_top"] if r["eq_top"][a] != runs["baseline"]["eq_top"][a]]
            runs[f"{name} x{f:.2f}"] = r
            log(f"  P(red win) {r['p_red_win']:.3f}; top line changed for: {', '.join(r['changed']) or 'nobody'}")
    name = job.get("name") or "ui_sensitivity"
    with open(os.path.join(ROOT, "results", f"{name}.sensitivity.json"), "w") as f:
        json.dump(runs, f, indent=1)
    log(f"Saved results/{name}.sensitivity.json")
    return {"result": f"{name}.sensitivity.json"}


def run_tests(job):
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    sys.path.insert(0, os.path.join(ROOT, "ui"))
    passed, failed = [], []
    for mod in sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "tests")) if f.startswith("test_") and f.endswith(".py")):
        t = importlib.import_module(mod)
        for name, fn in sorted(vars(t).items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                    passed.append(name)
                    log(f"PASS {mod}.{name}")
                except Exception:
                    failed.append(name)
                    log(f"FAIL {mod}.{name}\n{traceback.format_exc()}")
    log(f"{len(passed)} passed, {len(failed)} failed")
    return {"passed": passed, "failed": failed}


def run_dataset(job):
    from vrcwin.games.override import value as V
    base = job["scenario"]
    n = int(job.get("samples", 10))
    rng = random.Random(job.get("seed", int(time.time())))
    os.makedirs(os.path.dirname(DATASET), exist_ok=True)
    for k in range(n):
        scen = V.random_scenario(rng, base)
        t0 = time.time()
        lab = V.label(scen, workers=job.get("workers", 4), seed=k)
        row = {"scenario": scen, **lab, "seconds": round(time.time() - t0, 1)}
        with open(DATASET, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"sample {k + 1}/{n}: {scen['time_remaining']} s left, P(red win) {lab['p_red_win']:.3f}, "
            f"margin {lab['e_margin_red']:+.1f} ({row['seconds']} s)")
    total = len(V.load_dataset(DATASET))
    log(f"Dataset now has {total} labelled snapshots.")
    if job.get("fit_after", True) and total >= 8:
        return run_fit({})
    return {"samples": total}


def run_fit(job):
    from vrcwin.games.override import value as V
    rows = V.load_dataset(DATASET)
    if len(rows) < 8:
        raise SystemExit(f"Need at least 8 labelled snapshots to fit, have {len(rows)}.")
    X = [V.features(r["scenario"]) for r in rows]
    y = [r["p_red_win"] for r in rows]
    cv = V.cross_validate(X, y)
    model = V.ValueModel().fit(X, y)
    model.stats = {"n": cv["n"], "folds": cv["folds"], "model": cv["model"], "baseline": cv["baseline"],
                   "fitted_at": time.strftime("%Y-%m-%d %H:%M"),
                   "calibration": list(zip(cv["oof"], y)), "calibration_baseline": list(zip(cv["oof_baseline"], y))}
    with open(MODEL, "w") as f:
        json.dump(model.to_json(), f, indent=1)
    log(f"Fitted on {cv['n']} snapshots. Out-of-fold Brier: model {cv['model']['brier']:.4f}, "
        f"margin+time baseline {cv['baseline']['brier']:.4f}")
    return {"stats": {k: model.stats[k] for k in ("n", "model", "baseline")}}


KINDS = {"solve": run_solve, "sensitivity": run_sensitivity, "tests": run_tests,
         "dataset": run_dataset, "fit": run_fit}


if __name__ == "__main__":
    job = json.load(open(sys.argv[1]))
    try:
        out = KINDS[job["kind"]](job)
        print("RESULT " + json.dumps(out), flush=True)
    except SystemExit as e:
        print(f"ERROR {e}", flush=True)
        sys.exit(1)
    except Exception:
        print("ERROR " + traceback.format_exc(), flush=True)
        sys.exit(1)

# VRCWin: endgame solver (Override, V5RC 2026-27)

First piece of the strategy engine: given a snapshot of the last 20-30 seconds of a match, it finds an equilibrium of whole-window plans for all four robots and scores every robot's alternative lines by **P(win)** (the decision metric) with **expected score margin** beside it.

## Run it

```
python3 run_endgame.py scenarios/contested_top_25s.json --out results/contested_top_25s.json
python3 sensitivity.py scenarios/contested_top_25s.json --out results/sensitivity.json
python3 tests/test_override.py
python3 tests/test_ui.py
```

Python 3 plus numpy. One solve takes about 6 minutes on 4 cores.

## Interface (test, solve, train, examine)

```
python3 ui/server.py        # then open http://localhost:8765
```

No extra installs. Tabs:

- **Scenario**: drag robots and floor stacks on the field, click Toggles to cycle them, edit Goal halves, Loaders, fixed points and time left; per-robot speed / cycle / Toggle / push multipliers and capability toggles. Load and save `scenarios/*.json`.
- **Test**: pick a line for each robot and simulate the window hundreds of times instantly (P(win), margin spread) with an animated replay of one run; button to run everything in `tests/`.
- **Solve**: full or quick solve (quick ≈ 2-3 min, matched the full solve to 0.002 P(win) on the sample) and the ±15% sensitivity pass, as background jobs with live logs. Results land in `results/`.
- **Results**: equilibrium plan mixes, every robot's line table, sensitivity tables; load a result's scenario or replay its equilibrium in Test.
- **Train**: the learned value function (step two of the build order). Random snapshots are labelled by a reduced solve (about 2 min each on 4 cores) into `data/value_dataset.jsonl`, and a logistic model on hand-made features (score, Toggle ownership, who reaches the Midfield and each Toggle first) is fitted to them, with out-of-fold Brier against a margin + time-left baseline. Code: `vrcwin/games/override/value.py`.
- **Examine**: the value model's instant rating of the current scenario, which features drive it, and a positioning heatmap for any robot (the model's P(win) if that robot stood elsewhere).

## Layout

| Path | What it is |
|---|---|
| `vrcwin/engine/base.py` | Season-independent `Game` protocol the solver drives |
| `vrcwin/engine/smmcts.py` | Solver: decoupled regret matching+ per robot, variance-reduced, with state abstraction |
| `vrcwin/engine/parallel.py` | Runs solver copies on several cores and pools them |
| `vrcwin/engine/matrix.py` | Exact stage: zero-sum matrix game over shortlisted joint plans |
| `vrcwin/robot.py` | Reference robot profile: speed, accel, cycle times, success rates, capability toggles |
| `vrcwin/games/override/rules.py` | Override point values, field geometry, end-of-match scoring |
| `vrcwin/games/override/sim.py` | Override endgame as a simultaneous-move semi-Markov game over macro-actions |
| `vrcwin/games/override/plans.py` | The menu of lines (plans) per robot and their reactive rules |
| `scenarios/*.json` | Endgame snapshots |
| `results/` | Solver output |
| `vrcwin/games/override/value.py` | Learned value function: features, training labels, logistic model |
| `ui/` | Browser interface (`server.py`), its job runner (`worker.py`) and page (`static/`) |
| `data/` | Value-function training set and fitted model |

A new season means a new `vrcwin/games/<season>/` that implements `engine/base.py`; the engine does not change.

## How it models the endgame

**Macro-actions.** Each robot runs one action at a time: intake a Pin (floor stack or own Loader), score it on a Goal with a chosen half up, flip a Toggle, guard a Toggle, take/hold the Midfield, push an opponent out of the Midfield, wait. Duration = trapezoidal travel time + task time; success is sampled when the action finishes. Both alliances decide at the same instant whenever any robot is idle.

**Lines.** The solver chooses between whole-window plans, which is the abstraction that makes the values stable (the same move as a poker solver's bet-size menu). A plan is a main job (`score`, `score on Midfield Goal`, `take and hold Toggle q`, `hold Midfield all window`) plus an endgame stance (`stay to the buzzer`, `Midfield at 10 s`, `Midfield at the last moment`). The macro-action level solver (`Solver` on `OverrideEndgame` directly) is in the code but did not converge in useful time, see "Known gaps".

**Solve.** (1) Screening: regret matching over every plan for every robot; each plan is scored against the others' mix and the top 4 per robot go forward. (2) Matrix game: each alliance's actions are joint plans of its two robots, every red-vs-blue cell is simulated 300 times with common random numbers, and the zero-sum game is solved. (3) Double oracle: every one of the 19x19 joint plans per alliance is scanned against the opponent's mix; any that beats the equilibrium by more than 0.01 P(win) is added and the game re-solved, until nothing does. Step 3 matters: on the sample scenario screening missed red's best pairing, and the value moved from 0.61 to 0.84 once it was added.

**Line values.** A robot's line = its alliance's P(win) when it plays that plan, its partner plays the best complementary plan, and the opponents play their equilibrium mix (150 simulations per cell, so the top of each table is slightly optimistic).

## Rules modelled (manual v2.1, 2026-10-08)

- 5 points per Scored red/blue Pin Half, 10 per Scored yellow Pin Half that the alliance Owns, 8 per robot ending in the Midfield.
- Quadrant yellow halves belong to whoever's colour the Quadrant's Toggle is set to; a Toggle that a robot is touching at the buzzer counts as neutral (<SC4>, <SC5a>).
- Midfield Goal yellow halves belong to the alliance with more robots in the Midfield; a tie means nobody (<SC5b>).
- No Placing on the Midfield Goal in the final 10 s (<SG12-1>). Opposing Alliance Goals cannot be touched (<SG9>); Placed objects cannot be removed from neutral Goals (<SG10>). One Pin held at a time (<SG6>).
- Goal, Toggle and Loader coordinates come from the VEXcode VR Override playground's GPS reference.

## Assumptions to check

1. **Quadrants are the four diagonal triangles** (left, bottom, right, top), each with one Alliance Goal, one short neutral Goal and one wall Toggle; left/bottom are red, right/top blue. The manual's text didn't give the geometry, so this is inferred from the coordinates.
2. **Yellow halves on an Alliance Goal follow that Quadrant's Toggle** like the neutral Goal does.
3. **One visible half per placed Pin.** Cup tricks (showing both halves, covering an opponent's half) aren't modelled.
4. **Midfield is a 0.45 m-radius disc** around the tall Goal.
5. Reference robot: 1.8 m/s, 3 m/s², 1.2 s intake, 0.8 s place, 1.0 s Toggle, 95% intake/place, 90% Toggle, 35% push success, 25% of normal Toggle success when the Toggle is guarded.

## Known gaps

- Robots are points: no collisions, no blocking paths except the Toggle guard and Midfield push abstractions.
- The macro-action-level search (every 1-3 s decision) doesn't converge at Python speed; the plan menu is the working abstraction. A learned value function (the next step in the plan) is what lets the finer level work.
- No continuum EV yet (time-to-reach maps / positioning heatmap).
- Simulator stats are guesses until calibrated against tracker footage.
